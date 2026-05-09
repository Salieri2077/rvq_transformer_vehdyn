import argparse
import csv
import hashlib
import json
import os
import pickle
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from train_tfm import (
        TrajRVQTransformer,
        train_rvq_taae,
        signed_velocity_loss_from_dxdydyaw,
        signed_acceleration_loss_from_dxdydyaw,
        turn_global_yaw_loss,
    )
    from eval_tokenizer_by_scenario import (
        load_norm_params,
        normalize_trajs,
        reconstruct_trajs,
        scenario_metrics,
    )
    from eval_tokenizer_health import evaluate_tokenizer_health
    from utils import load_sampled_datas, preprocess_and_save_norm_params
except ImportError:
    from rvq_transformer_vehdyn.train_tfm import (
        TrajRVQTransformer,
        train_rvq_taae,
        signed_velocity_loss_from_dxdydyaw,
        signed_acceleration_loss_from_dxdydyaw,
        turn_global_yaw_loss,
    )
    from rvq_transformer_vehdyn.eval_tokenizer_by_scenario import (
        load_norm_params,
        normalize_trajs,
        reconstruct_trajs,
        scenario_metrics,
    )
    from rvq_transformer_vehdyn.eval_tokenizer_health import evaluate_tokenizer_health
    from rvq_transformer_vehdyn.utils import load_sampled_datas, preprocess_and_save_norm_params


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_py(obj):
    """递归转成 Python 原生类型，确保 json.dump 不报错。"""
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return _to_py(obj.detach().cpu().numpy())
    return obj


def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {k: 0.0 for k in ["p0", "p25", "p50", "p75", "p95", "p99", "max"]}
    q = np.percentile(arr, [0, 25, 50, 75, 95, 99])
    return {
        "p0": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(np.max(arr)),
    }


def _write_json(path: str, data: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_py(data), f, indent=2)


def _write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float_str(v: float) -> str:
    return f"{float(v):.8f}"


def _build_grouping_cache_key(
    *,
    data_path: Optional[str],
    data_type: str,
    n_total: int,
    num_steps: int,
    requested_num_groups: int,
    grouping_method: str,
    group_feature: str,
    shape_downsample_steps: int,
    feature_xy_weight: float,
    feature_yaw_weight: float,
    dt: float,
    kmeans_batch_size: int,
    kmeans_max_iter: int,
    kmeans_random_state: int,
    seed: int,
) -> str:
    """
    为分组结果生成稳定 cache key。
    不依赖大数组哈希，避免额外开销；使用数据路径 + 关键参数 + 数据规模组合。
    """
    payload = {
        "data_path": str(data_path),
        "data_type": str(data_type),
        "n_total": int(n_total),
        "num_steps": int(num_steps),
        "requested_num_groups": int(requested_num_groups),
        "grouping_method": str(grouping_method),
        "group_feature": str(group_feature),
        "shape_downsample_steps": int(shape_downsample_steps),
        "feature_xy_weight": _safe_float_str(feature_xy_weight),
        "feature_yaw_weight": _safe_float_str(feature_yaw_weight),
        "dt": _safe_float_str(dt),
        "kmeans_batch_size": int(kmeans_batch_size),
        "kmeans_max_iter": int(kmeans_max_iter),
        "kmeans_random_state": int(kmeans_random_state),
        "seed": int(seed),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _try_load_grouping_from_cache(
    cache_dir: str,
    n_total: int,
) -> Optional[Dict[str, object]]:
    rep_path = os.path.join(cache_dir, "representative_indices.npy")
    gid_path = os.path.join(cache_dir, "group_id_per_sample.npy")
    gsz_path = os.path.join(cache_dir, "group_sizes.npy")
    repd_path = os.path.join(cache_dir, "representative_distance_to_center.npy")
    meta_path = os.path.join(cache_dir, "group_cache_meta.json")
    if not (os.path.exists(rep_path) and os.path.exists(gid_path) and os.path.exists(gsz_path)):
        return None

    try:
        representative_indices = np.load(rep_path).astype(np.int64)
        group_id_per_sample = np.load(gid_path).astype(np.int32)
        group_sizes = np.load(gsz_path).astype(np.int64)
        if os.path.exists(repd_path):
            rep_dist = np.load(repd_path).astype(np.float32)
        else:
            rep_dist = np.zeros((representative_indices.shape[0],), dtype=np.float32)
    except Exception:
        return None

    if group_id_per_sample.shape[0] != int(n_total):
        return None
    if representative_indices.shape[0] != group_sizes.shape[0]:
        return None

    actual_num_groups = int(group_sizes.shape[0])
    if actual_num_groups <= 0:
        return None

    group_to_indices = _build_group_to_indices(group_id_per_sample, actual_num_groups)

    grouping_method_used = "cache"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            grouping_method_used = str(meta.get("grouping_method_used", grouping_method_used))
        except Exception:
            pass

    return {
        "group_id_per_sample": group_id_per_sample,
        "representative_indices": representative_indices,
        "group_sizes": group_sizes,
        "representative_distance_to_center": rep_dist,
        "group_to_indices": group_to_indices,
        "actual_num_groups": actual_num_groups,
        "grouping_method_used": grouping_method_used,
    }


def _save_grouping_cache(
    cache_dir: str,
    grouping: Dict[str, object],
    meta: Dict[str, object],
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "representative_indices.npy"), np.asarray(grouping["representative_indices"], dtype=np.int64))
    np.save(os.path.join(cache_dir, "group_id_per_sample.npy"), np.asarray(grouping["group_id_per_sample"], dtype=np.int32))
    np.save(os.path.join(cache_dir, "group_sizes.npy"), np.asarray(grouping["group_sizes"], dtype=np.int64))
    np.save(
        os.path.join(cache_dir, "representative_distance_to_center.npy"),
        np.asarray(grouping["representative_distance_to_center"], dtype=np.float32),
    )
    _write_json(os.path.join(cache_dir, "group_cache_meta.json"), meta)


def compute_motion_stats(trajs: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """统计运动学分布，只用于分析与分组，不用于过滤。"""
    x = np.asarray(trajs, dtype=np.float32)
    dx = x[..., 0]
    dy = x[..., 1]
    dyaw = x[..., 2]

    step_dist = np.sqrt(dx * dx + dy * dy)
    total_path_length = np.sum(step_dist, axis=1)
    mean_speed = np.mean(step_dist, axis=1) / max(float(dt), 1e-6)
    max_speed = np.max(step_dist, axis=1) / max(float(dt), 1e-6)
    abs_yaw_sum = np.sum(np.abs(dyaw), axis=1)
    final_yaw_abs = np.abs(np.sum(dyaw, axis=1))
    lateral_abs_sum = np.sum(np.abs(dy), axis=1)

    stationary_like = (total_path_length < 0.5) | (mean_speed < 0.1)

    return {
        "total_path_length": total_path_length.astype(np.float32),
        "mean_speed": mean_speed.astype(np.float32),
        "max_speed": max_speed.astype(np.float32),
        "abs_yaw_sum": abs_yaw_sum.astype(np.float32),
        "final_yaw_abs": final_yaw_abs.astype(np.float32),
        "lateral_abs_sum": lateral_abs_sum.astype(np.float32),
        "stationary_like": stationary_like.astype(bool),
    }


def _uniform_downsample_indices(num_steps: int, target_steps: int) -> np.ndarray:
    if target_steps <= 1:
        return np.asarray([0], dtype=np.int64)
    if target_steps >= num_steps:
        return np.arange(num_steps, dtype=np.int64)
    return np.linspace(0, num_steps - 1, target_steps).round().astype(np.int64)


def _robust_scale(feature: np.ndarray) -> np.ndarray:
    med = np.median(feature, axis=0, keepdims=True)
    q75 = np.percentile(feature, 75, axis=0, keepdims=True)
    q25 = np.percentile(feature, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    return (feature - med) / (iqr + 1e-6)


def build_group_features(
    trajs: np.ndarray,
    dt: float,
    group_feature: str,
    shape_downsample_steps: int,
    xy_weight: float,
    yaw_weight: float,
    motion_stats: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """构造分组特征：kinematic / shape / kinematic_plus_shape。"""
    x = np.asarray(trajs, dtype=np.float32)
    n, t, _ = x.shape

    if motion_stats is None:
        motion_stats = compute_motion_stats(x, dt=dt)

    dx = x[..., 0]
    dy = x[..., 1]
    dyaw = x[..., 2]

    mean_abs_dx = np.mean(np.abs(dx), axis=1)
    mean_abs_dy = np.mean(np.abs(dy), axis=1)
    mean_abs_dyaw = np.mean(np.abs(dyaw), axis=1)
    std_dx = np.std(dx, axis=1)
    std_dy = np.std(dy, axis=1)
    std_dyaw = np.std(dyaw, axis=1)

    kin = np.stack(
        [
            motion_stats["total_path_length"],
            motion_stats["mean_speed"],
            motion_stats["max_speed"],
            motion_stats["abs_yaw_sum"],
            motion_stats["final_yaw_abs"],
            motion_stats["lateral_abs_sum"],
            mean_abs_dx,
            mean_abs_dy,
            mean_abs_dyaw,
            std_dx,
            std_dy,
            std_dyaw,
        ],
        axis=1,
    ).astype(np.float32)

    # 正值统计做 log1p，缓解长尾。
    kin = np.log1p(np.maximum(kin, 0.0))
    kin = _robust_scale(kin).astype(np.float32)

    feat_parts: List[np.ndarray] = []
    if group_feature in ("kinematic", "kinematic_plus_shape"):
        feat_parts.append(kin)

    if group_feature in ("shape", "kinematic_plus_shape"):
        ds_idx = _uniform_downsample_indices(t, max(1, int(shape_downsample_steps)))
        shape = x[:, ds_idx, :].copy()
        shape[..., 0] *= float(xy_weight)
        shape[..., 1] *= float(xy_weight)
        shape[..., 2] *= float(yaw_weight)
        shape = shape.reshape(n, -1).astype(np.float32)
        feat_parts.append(shape)

    if not feat_parts:
        raise ValueError(f"Unsupported group_feature: {group_feature}")

    feat = np.concatenate(feat_parts, axis=1).astype(np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    feat = (feat - mean) / (std + 1e-6)
    feat = feat.astype(np.float32)

    print(f"Grouping feature shape: {feat.shape}")
    return feat


def _build_group_to_indices(group_id_per_sample: np.ndarray, num_groups: int) -> List[np.ndarray]:
    order = np.argsort(group_id_per_sample, kind="mergesort")
    labels_sorted = group_id_per_sample[order]
    group_to_indices: List[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(num_groups)]
    if order.size == 0:
        return group_to_indices

    boundaries = np.flatnonzero(np.diff(labels_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [order.size]])
    for s, e in zip(starts.tolist(), ends.tolist()):
        gid = int(labels_sorted[s])
        group_to_indices[gid] = order[s:e].astype(np.int64)
    return group_to_indices


def _compute_representatives_from_groups(
    features: np.ndarray,
    group_to_indices: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    reps = np.zeros((len(group_to_indices),), dtype=np.int64)
    rep_dist = np.zeros((len(group_to_indices),), dtype=np.float32)
    for gid, idxs in enumerate(group_to_indices):
        if idxs.size == 0:
            reps[gid] = 0
            rep_dist[gid] = 0.0
            continue
        feat = features[idxs]
        center = feat.mean(axis=0, keepdims=True)
        d2 = np.sum((feat - center) ** 2, axis=1)
        j = int(np.argmin(d2))
        reps[gid] = int(idxs[j])
        rep_dist[gid] = float(np.sqrt(max(float(d2[j]), 0.0)))
    return reps, rep_dist


def _split_groups_to_target(
    groups: List[np.ndarray],
    features: np.ndarray,
    num_groups: int,
    seed: int,
) -> List[np.ndarray]:
    """将初始分组数调整到接近期望组数。"""
    n = int(sum(int(g.size) for g in groups))
    target = int(max(1, min(num_groups, n)))
    if len(groups) == target:
        return groups

    # 初始组过多：做平衡合并。
    if len(groups) > target:
        groups_sorted = sorted(groups, key=lambda g: int(g.size), reverse=True)
        buckets: List[List[np.ndarray]] = [[] for _ in range(target)]
        bucket_sizes = np.zeros((target,), dtype=np.int64)
        for g in groups_sorted:
            bid = int(np.argmin(bucket_sizes))
            buckets[bid].append(g)
            bucket_sizes[bid] += int(g.size)
        merged: List[np.ndarray] = []
        for bucket in buckets:
            merged.append(np.concatenate(bucket, axis=0).astype(np.int64) if bucket else np.zeros((0,), dtype=np.int64))
        return merged

    # 初始组不足：按组内特征一维排序后切块细分。
    avg_target_size = float(n) / float(target)
    base_split = []
    split_remainder = []
    for g in groups:
        desired = float(g.size) / max(avg_target_size, 1e-6)
        k = int(np.floor(desired))
        k = max(1, min(int(g.size), k))
        base_split.append(k)
        split_remainder.append(desired - np.floor(desired))

    current = int(np.sum(base_split))
    order_add = np.argsort(-np.asarray(split_remainder))
    order_sub = np.argsort(np.asarray(split_remainder))

    ptr = 0
    while current < target and ptr < len(order_add):
        i = int(order_add[ptr])
        if base_split[i] < int(groups[i].size):
            base_split[i] += 1
            current += 1
        else:
            ptr += 1

    ptr = 0
    while current > target and ptr < len(order_sub):
        i = int(order_sub[ptr])
        if base_split[i] > 1:
            base_split[i] -= 1
            current -= 1
        else:
            ptr += 1

    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for i, g in enumerate(groups):
        k = int(max(1, min(base_split[i], int(g.size))))
        if k == 1:
            out.append(g.astype(np.int64))
            continue
        feat = features[g]
        var = np.var(feat, axis=0)
        dim = int(np.argmax(var))
        # 若该维完全无方差，退化成随机投影，避免无法切分。
        if float(var[dim]) < 1e-12:
            proj = rng.standard_normal((feat.shape[0],), dtype=np.float32)
        else:
            proj = feat[:, dim]
        order = np.argsort(proj)
        splits = np.array_split(order, k)
        for s in splits:
            if s.size > 0:
                out.append(g[s].astype(np.int64))

    if len(out) > target:
        out = sorted(out, key=lambda a: int(a.size), reverse=True)[:target]
    elif len(out) < target:
        # 极端情况下补齐空组前，尽量再次拆大组。
        out = sorted(out, key=lambda a: int(a.size), reverse=True)
        i = 0
        while len(out) < target and i < len(out):
            g = out[i]
            if g.size >= 2:
                half = g.size // 2
                out[i] = g[:half]
                out.append(g[half:])
            else:
                i += 1

    return out


def find_representative_groups_kinematic_bins(
    trajs: np.ndarray,
    features: np.ndarray,
    motion_stats: Dict[str, np.ndarray],
    num_groups: int,
    seed: int,
) -> Dict[str, object]:
    """运动学分箱分组（fallback）：先分箱，再按特征细分/合并到目标组数。"""
    n = int(trajs.shape[0])
    target = int(max(1, min(num_groups, n)))

    stats_for_bins = [
        motion_stats["mean_speed"],
        motion_stats["abs_yaw_sum"],
        motion_stats["total_path_length"],
        motion_stats["lateral_abs_sum"],
    ]

    binned_cols = []
    for arr in stats_for_bins:
        arr = np.asarray(arr, dtype=np.float32)
        q = np.percentile(arr, np.linspace(0, 100, 11))
        q = np.unique(q)
        if q.size <= 2:
            bins = np.zeros_like(arr, dtype=np.int32)
        else:
            bins = np.searchsorted(q[1:-1], arr, side="right").astype(np.int32)
        binned_cols.append(bins)

    b0, b1, b2, b3 = binned_cols
    key = b0 + 16 * b1 + 256 * b2 + 4096 * b3

    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    boundaries = np.flatnonzero(np.diff(key_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [order.size]])

    groups: List[np.ndarray] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        groups.append(order[s:e].astype(np.int64))

    groups = _split_groups_to_target(groups, features=features, num_groups=target, seed=seed)

    group_id = np.zeros((n,), dtype=np.int32)
    for gid, idxs in enumerate(groups):
        group_id[idxs] = int(gid)

    group_sizes = np.asarray([int(g.size) for g in groups], dtype=np.int64)
    reps, rep_dist = _compute_representatives_from_groups(features, groups)

    return {
        "group_id_per_sample": group_id,
        "representative_indices": reps,
        "group_sizes": group_sizes,
        "representative_distance_to_center": rep_dist,
        "group_to_indices": groups,
        "actual_num_groups": int(len(groups)),
        "grouping_method_used": "kinematic_bins",
    }


def find_representative_groups_minibatch_kmeans(
    trajs: np.ndarray,
    features: np.ndarray,
    num_groups: int,
    batch_size: int,
    max_iter: int,
    seed: int,
    motion_stats: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """MiniBatchKMeans 分组；若 sklearn 不可用则自动回退 kinematic_bins。"""
    n = int(trajs.shape[0])
    target = int(max(1, min(num_groups, n)))

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception:
        print("[warning] sklearn 不可用，fallback 到 kinematic_bins 分组。")
        return find_representative_groups_kinematic_bins(
            trajs=trajs,
            features=features,
            motion_stats=motion_stats,
            num_groups=target,
            seed=seed,
        )

    # sklearn 版本兼容：
    # 有些版本在构造时接受 n_init='auto'，但 fit 时才报类型错误。
    # 因此这里在 fit 阶段也做一次兜底回退。
    def _build_kmeans(n_init_value):
        return MiniBatchKMeans(
            n_clusters=target,
            batch_size=max(256, int(batch_size)),
            max_iter=max(10, int(max_iter)),
            random_state=seed,
            n_init=n_init_value,
            reassignment_ratio=0.01,
            verbose=0,
        )

    kmeans = _build_kmeans("auto")
    try:
        labels_raw = kmeans.fit_predict(features).astype(np.int32)
        centers_raw = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    except Exception:
        # 兼容老 sklearn：有的版本在 __init__ 不报错，但 fit 时才因 n_init='auto' 失败。
        # 只要当前 n_init 是字符串，就回退到整数 n_init 再试一次。
        if not isinstance(getattr(kmeans, "n_init", None), str):
            raise
        print("[warning] sklearn 版本不支持 n_init='auto'，自动回退到 n_init=3。")
        kmeans = _build_kmeans(3)
        labels_raw = kmeans.fit_predict(features).astype(np.int32)
        centers_raw = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    unique_labels, group_id_per_sample = np.unique(labels_raw, return_inverse=True)
    centers = centers_raw[unique_labels]
    num_actual = int(unique_labels.shape[0])

    group_to_indices = _build_group_to_indices(group_id_per_sample.astype(np.int32), num_actual)

    reps = np.zeros((num_actual,), dtype=np.int64)
    rep_dist = np.zeros((num_actual,), dtype=np.float32)
    group_sizes = np.zeros((num_actual,), dtype=np.int64)

    for gid, idxs in enumerate(group_to_indices):
        group_sizes[gid] = int(idxs.size)
        if idxs.size == 0:
            reps[gid] = 0
            rep_dist[gid] = 0.0
            continue
        feat = features[idxs]
        c = centers[gid]
        d2 = np.sum((feat - c[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        reps[gid] = int(idxs[j])
        rep_dist[gid] = float(np.sqrt(max(float(d2[j]), 0.0)))

    return {
        "group_id_per_sample": group_id_per_sample.astype(np.int32),
        "representative_indices": reps,
        "group_sizes": group_sizes,
        "representative_distance_to_center": rep_dist,
        "group_to_indices": group_to_indices,
        "actual_num_groups": num_actual,
        "grouping_method_used": "minibatch_kmeans",
    }


def get_primary_codebook_weight(model: TrajRVQTransformer) -> Optional[torch.Tensor]:
    """尽力提取 RVQ 第一层 codebook（兼容不同实现字段名）。"""
    rvq = getattr(model, "rvq", None)
    if rvq is None:
        return None

    def _extract_weight(obj) -> Optional[torch.Tensor]:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, "weight") and isinstance(getattr(obj, "weight"), torch.Tensor):
            return getattr(obj, "weight")
        return None

    def _to_2d(w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if w is None:
            return None
        if w.ndim == 2:
            return w
        if w.ndim == 3:
            return w[0]
        return None

    if hasattr(rvq, "layers") and len(rvq.layers) > 0:
        layer0 = rvq.layers[0]
        for name in ["embedding", "embeddings", "codebook", "codebooks"]:
            if hasattr(layer0, name):
                w = _to_2d(_extract_weight(getattr(layer0, name)))
                if w is not None:
                    return w
        w = _to_2d(_extract_weight(layer0))
        if w is not None:
            return w

    for name in ["codebooks", "codebook", "embedding", "embeddings"]:
        if hasattr(rvq, name):
            obj = getattr(rvq, name)
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                w = _to_2d(_extract_weight(obj[0]))
            else:
                w = _to_2d(_extract_weight(obj))
            if w is not None:
                return w
    return None


def flatten_or_pool_latent(z: torch.Tensor) -> torch.Tensor:
    """将 encoder 输出统一为 [B, D]，便于 consistency 计算。"""
    if z.ndim == 2:
        return z
    if z.ndim == 3:
        return z.mean(dim=1)
    return z.view(z.shape[0], -1)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """监督式对比损失：同 label 为正样本，不同 label 为负样本。"""
    if features.shape[0] <= 1:
        return features.new_zeros(())

    z = F.normalize(features, dim=1)
    sim = torch.matmul(z, z.t()) / max(float(temperature), 1e-6)
    n = int(sim.shape[0])

    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    pos_mask = (labels[:, None] == labels[None, :]) & (~eye)
    den_mask = ~eye

    has_pos = pos_mask.any(dim=1)
    if not torch.any(has_pos):
        return sim.new_zeros(())

    losses = []
    for i in torch.where(has_pos)[0].tolist():
        sim_i = sim[i]
        pos_lse = torch.logsumexp(sim_i[pos_mask[i]], dim=0)
        den_lse = torch.logsumexp(sim_i[den_mask[i]], dim=0)
        losses.append(-(pos_lse - den_lse))
    return torch.stack(losses).mean() if losses else sim.new_zeros(())


def sample_consistency_batch(
    trajs: np.ndarray,
    group_id_per_sample: np.ndarray,
    group_to_indices: List[np.ndarray],
    positive_groups_per_step: int,
    positives_per_group: int,
    negative_groups_per_step: int,
    seed_or_rng,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态采样 consistency batch，避免在 108 万条上构造全量标签矩阵。
    返回:
      consistency_trajs: [B, T, 3]
      labels: [B]
    """
    if isinstance(seed_or_rng, np.random.Generator):
        rng = seed_or_rng
    else:
        rng = np.random.default_rng(int(seed_or_rng))

    valid_positive_groups = [gid for gid, idxs in enumerate(group_to_indices) if idxs.size >= 2]
    if len(valid_positive_groups) == 0:
        return np.zeros((0,) + trajs.shape[1:], dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n_pos_groups = min(max(1, int(positive_groups_per_step)), len(valid_positive_groups))
    pos_groups = rng.choice(valid_positive_groups, size=n_pos_groups, replace=False)

    traj_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    next_label = 0
    selected_pos_group_set = set(int(g) for g in pos_groups.tolist())
    for gid in pos_groups.tolist():
        idxs = group_to_indices[int(gid)]
        take = min(max(1, int(positives_per_group)), int(idxs.size))
        picked = rng.choice(idxs, size=take, replace=False)
        traj_parts.append(trajs[picked].astype(np.float32))
        label_parts.append(np.full((take,), next_label, dtype=np.int64))
        next_label += 1

    # negatives: 其它 group 各取 1 条，作为分母项。
    candidate_neg_groups = [gid for gid, idxs in enumerate(group_to_indices) if idxs.size > 0 and gid not in selected_pos_group_set]
    if len(candidate_neg_groups) > 0 and int(negative_groups_per_step) > 0:
        n_neg = min(int(negative_groups_per_step), len(candidate_neg_groups))
        neg_groups = rng.choice(candidate_neg_groups, size=n_neg, replace=False)
        for gid in neg_groups.tolist():
            idxs = group_to_indices[int(gid)]
            picked = int(rng.choice(idxs, size=1, replace=False)[0])
            traj_parts.append(trajs[picked : picked + 1].astype(np.float32))
            label_parts.append(np.full((1,), next_label, dtype=np.int64))
            next_label += 1

    if not traj_parts:
        return np.zeros((0,) + trajs.shape[1:], dtype=np.float32), np.zeros((0,), dtype=np.int64)

    x = np.concatenate(traj_parts, axis=0).astype(np.float32)
    y = np.concatenate(label_parts, axis=0).astype(np.int64)
    return x, y


def _normalize_by_saved_params(
    trajs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: Optional[torch.Tensor],
) -> torch.Tensor:
    x = (trajs - mean) / (std + 1e-8)
    if clip_limit is not None:
        x = torch.clamp(x, -clip_limit, clip_limit)
    return x / scale_factor


def _base_taae_loss(
    model: TrajRVQTransformer,
    x_norm: torch.Tensor,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """保持 train_rvq_taae 的 base loss 风格。"""
    x_recon, vq_loss, _, v, kappa = model(x_norm)

    mse_dxdy = F.mse_loss(x_recon[..., :2], x_norm[..., :2])
    mse_dyaw = F.mse_loss(x_recon[..., 2], x_norm[..., 2])
    recon_loss = mse_dxdy + 14.0 * mse_dyaw

    pred_phys = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
    gt_phys = (x_norm * model.norm_scale * model.norm_std) + model.norm_mean

    vel_loss = signed_velocity_loss_from_dxdydyaw(pred_phys, gt_phys, dt=model.dt)
    acc_loss = signed_acceleration_loss_from_dxdydyaw(pred_phys, gt_phys, dt=model.dt)
    turn_global_loss, turn_yaw_loss, _ = turn_global_yaw_loss(
        pred_phys,
        gt_phys,
        turn_threshold=0.35,
    )

    acc = (v[:, 1:] - v[:, :-1]) / model.dt
    kappa_rate = (kappa[:, 1:] - kappa[:, :-1]) / model.dt
    kin_smooth_loss = acc.pow(2).mean() + kappa_rate.pow(2).mean()

    recon_loss_weight = 10.0
    vq_loss_weight = 5.0
    vel_loss_weight = 0.5
    acc_loss_weight = 0.05
    kin_smooth_weight = 1e-2 if epoch > 30 else 0.0
    turn_global_weight = 1.0
    turn_yaw_weight = 2.0

    base_loss = (
        recon_loss_weight * recon_loss
        + vq_loss_weight * vq_loss
        + vel_loss_weight * vel_loss
        + acc_loss_weight * acc_loss
        + kin_smooth_weight * kin_smooth_loss
        + turn_global_weight * turn_global_loss
        + turn_yaw_weight * turn_yaw_loss
    )

    terms = {
        "recon_loss": recon_loss,
        "vq_loss": vq_loss,
        "vel_loss": vel_loss,
        "acc_loss": acc_loss,
        "turn_global_loss": turn_global_loss,
        "turn_yaw_loss": turn_yaw_loss,
        "kin_smooth_loss": kin_smooth_loss,
    }
    return base_loss, terms, x_recon


def _latent_consistency_loss(z_pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for lab in torch.unique(labels).tolist():
        mask = labels == int(lab)
        if int(mask.sum().item()) < 2:
            continue
        z_group = F.normalize(z_pooled[mask], dim=1)
        proto = F.normalize(z_group.detach().mean(dim=0, keepdim=True), dim=1)
        losses.append((1.0 - torch.sum(z_group * proto, dim=1)).mean())
    if not losses:
        return z_pooled.new_zeros(())
    return torch.stack(losses).mean()


def _soft_primary_code_consistency_loss(
    z_pooled: torch.Tensor,
    labels: torch.Tensor,
    codebook_weight: Optional[torch.Tensor],
    temperature: float,
    warn_state: Dict[str, bool],
) -> torch.Tensor:
    if codebook_weight is None:
        return z_pooled.new_zeros(())

    cb = codebook_weight
    if cb.ndim != 2:
        if not warn_state.get("warned_codebook_shape", False):
            print("[warning] primary codebook 维度异常，soft_primary_code consistency 已跳过。")
            warn_state["warned_codebook_shape"] = True
        return z_pooled.new_zeros(())

    if cb.shape[1] != z_pooled.shape[1] and cb.shape[0] == z_pooled.shape[1]:
        cb = cb.t()

    if cb.shape[1] != z_pooled.shape[1]:
        if not warn_state.get("warned_codebook_dim", False):
            print("[warning] primary codebook 与 latent 维度不匹配，soft_primary_code consistency 已跳过。")
            warn_state["warned_codebook_dim"] = True
        return z_pooled.new_zeros(())

    z_sq = torch.sum(z_pooled * z_pooled, dim=1, keepdim=True)
    cb_sq = torch.sum(cb * cb, dim=1, keepdim=True).t()
    dist = z_sq + cb_sq - 2.0 * torch.matmul(z_pooled, cb.t())
    prob = torch.softmax(-dist / max(float(temperature), 1e-6), dim=1)

    losses = []
    eps = 1e-8
    for lab in torch.unique(labels).tolist():
        mask = labels == int(lab)
        if int(mask.sum().item()) < 2:
            continue
        p_group = prob[mask]
        p_proto = p_group.detach().mean(dim=0, keepdim=True)

        p_group_safe = torch.clamp(p_group, min=eps)
        p_proto_safe = torch.clamp(p_proto, min=eps)
        p_group_safe = p_group_safe / p_group_safe.sum(dim=1, keepdim=True)
        p_proto_safe = p_proto_safe / p_proto_safe.sum(dim=1, keepdim=True)

        kl_proto_to_group = (p_proto_safe * (torch.log(p_proto_safe) - torch.log(p_group_safe))).sum(dim=1)
        kl_group_to_proto = (p_group_safe * (torch.log(p_group_safe) - torch.log(p_proto_safe))).sum(dim=1)
        losses.append(0.5 * (kl_proto_to_group + kl_group_to_proto).mean())

    if not losses:
        return z_pooled.new_zeros(())
    return torch.stack(losses).mean()


def train_rvq_taae_with_group_consistency(
    base_train_trajs: np.ndarray,
    all_trajs: np.ndarray,
    group_id_per_sample: np.ndarray,
    group_to_indices: List[np.ndarray],
    save_dir: str,
    data_type: str,
    batch_size: int,
    num_layers: int,
    num_transformer_layers: int,
    epochs: int,
    positive_groups_per_step: int,
    positives_per_group: int,
    negative_groups_per_step: int,
    lambda_latent_consistency: float,
    lambda_soft_code_consistency: float,
    lambda_supcon: float,
    contrastive_temperature: float,
    soft_code_temperature: float,
    consistency_warmup_epochs: int,
    consistency_target: str,
    seed: int,
) -> List[Dict[str, float]]:
    """similar_consistency 实验训练：base loss + 动态采样 consistency loss。"""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_normalized = preprocess_and_save_norm_params(base_train_trajs, save_dir, data_type)
    dataset = TensorDataset(torch.tensor(data_normalized, dtype=torch.float32))
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = TrajRVQTransformer(
        input_steps=base_train_trajs.shape[1],
        input_dim=base_train_trajs.shape[2],
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=num_transformer_layers,
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        use_amp = True
        amp_dtype = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        print("Using BF16 mixed precision training")
    elif torch.cuda.is_available():
        use_amp = True
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print("Using FP16 mixed precision training")
    else:
        use_amp = False
        amp_dtype = torch.float32
        scaler = None
        print("Using FP32 training")

    initial_lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-5 / initial_lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs) - warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")
    with open(norm_path, "rb") as f:
        norm_params = pickle.load(f)

    mean = torch.tensor(norm_params["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm_params["std"], dtype=torch.float32, device=device)
    scale_factor = torch.tensor(norm_params["scale_factor"], dtype=torch.float32, device=device)
    clip_limit = torch.tensor(norm_params["clip_limit"], dtype=torch.float32, device=device) if "clip_limit" in norm_params else None
    model.set_norm_params(mean, std, scale_factor)

    run_name = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))

    use_latent = consistency_target in ("latent", "latent_plus_soft_code")
    use_soft = consistency_target in ("soft_primary_code", "latent_plus_soft_code")

    codebook_weight = get_primary_codebook_weight(model) if use_soft else None
    if use_soft and codebook_weight is None:
        print("[warning] 未找到 primary codebook，soft_primary_code consistency 将跳过。")

    rng = np.random.default_rng(seed)
    warn_state = {
        "warned_codebook_shape": False,
        "warned_codebook_dim": False,
    }

    history: List[Dict[str, float]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        if epoch > epochs * 0.8:
            model.rvq.dropout = 0.0

        warmup_weight = 1.0
        if int(consistency_warmup_epochs) > 0:
            warmup_weight = min(1.0, float(epoch) / float(consistency_warmup_epochs))

        total_loss_sum = 0.0
        recon_sum = 0.0
        vq_sum = 0.0
        latent_sum = 0.0
        soft_sum = 0.0
        supcon_sum = 0.0
        n_batches = 0

        for batch in dataloader:
            x_norm = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                base_loss, base_terms, _ = _base_taae_loss(model=model, x_norm=x_norm, epoch=epoch)

                # 动态采样 consistency batch（原始物理空间），再按保存参数归一化。
                c_trajs_np, c_labels_np = sample_consistency_batch(
                    trajs=all_trajs,
                    group_id_per_sample=group_id_per_sample,
                    group_to_indices=group_to_indices,
                    positive_groups_per_step=positive_groups_per_step,
                    positives_per_group=positives_per_group,
                    negative_groups_per_step=negative_groups_per_step,
                    seed_or_rng=rng,
                )

                latent_loss = x_norm.new_zeros(())
                soft_loss = x_norm.new_zeros(())
                supcon_loss = x_norm.new_zeros(())

                if c_trajs_np.shape[0] > 1:
                    c_x_phys = torch.tensor(c_trajs_np, dtype=torch.float32, device=device)
                    c_labels = torch.tensor(c_labels_np, dtype=torch.long, device=device)
                    c_x_norm = _normalize_by_saved_params(
                        trajs=c_x_phys,
                        mean=mean,
                        std=std,
                        scale_factor=scale_factor,
                        clip_limit=clip_limit,
                    )
                    z = model.encode(c_x_norm)
                    z_pooled = flatten_or_pool_latent(z)

                    if use_latent:
                        latent_loss = _latent_consistency_loss(z_pooled=z_pooled, labels=c_labels)
                    if use_soft:
                        soft_loss = _soft_primary_code_consistency_loss(
                            z_pooled=z_pooled,
                            labels=c_labels,
                            codebook_weight=codebook_weight,
                            temperature=soft_code_temperature,
                            warn_state=warn_state,
                        )
                    supcon_loss = supervised_contrastive_loss(
                        features=z_pooled,
                        labels=c_labels,
                        temperature=contrastive_temperature,
                    )

                consistency_term = (
                    float(lambda_latent_consistency) * latent_loss
                    + float(lambda_soft_code_consistency) * soft_loss
                    + float(lambda_supcon) * supcon_loss
                )
                total_loss = base_loss + warmup_weight * consistency_term

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += float(total_loss.item())
            recon_sum += float(base_terms["recon_loss"].item())
            vq_sum += float(base_terms["vq_loss"].item())
            latent_sum += float(latent_loss.item())
            soft_sum += float(soft_loss.item())
            supcon_sum += float(supcon_loss.item())
            n_batches += 1

        scheduler.step()

        denom = max(1, n_batches)
        row = {
            "epoch": int(epoch + 1),
            "total_loss": total_loss_sum / denom,
            "recon_loss": recon_sum / denom,
            "vq_loss": vq_sum / denom,
            "latent_consistency_loss": latent_sum / denom,
            "soft_code_consistency_loss": soft_sum / denom,
            "supcon_loss": supcon_sum / denom,
            "warmup_weight": float(warmup_weight),
        }
        history.append(row)

        writer.add_scalar("loss/total", row["total_loss"], epoch + 1)
        writer.add_scalar("loss/recon", row["recon_loss"], epoch + 1)
        writer.add_scalar("loss/vq", row["vq_loss"], epoch + 1)
        writer.add_scalar("loss/latent_consistency", row["latent_consistency_loss"], epoch + 1)
        writer.add_scalar("loss/soft_code_consistency", row["soft_code_consistency_loss"], epoch + 1)
        writer.add_scalar("loss/supcon", row["supcon_loss"], epoch + 1)
        writer.add_scalar("loss/warmup_weight", row["warmup_weight"], epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(
                f"[SimilarConsistency] Epoch {epoch+1:03d} | total={row['total_loss']:.5f} | "
                f"recon={row['recon_loss']:.5f} | vq={row['vq_loss']:.5f} | "
                f"latent={row['latent_consistency_loss']:.5f} | "
                f"soft={row['soft_code_consistency_loss']:.5f} | "
                f"supcon={row['supcon_loss']:.5f} | warmup={row['warmup_weight']:.3f}"
            )

    torch.save(model.state_dict(), os.path.join(save_dir, f"{data_type}_rvq_taae_model.pth"))
    writer.close()
    print(f"Similar-consistency Training Done. Model saved to {save_dir}")
    return history


def build_model_and_norm(
    save_dir: str,
    data_type: str,
    input_steps: int,
    num_layers: int,
    num_transformer_layers: int,
    device: torch.device,
) -> Tuple[TrajRVQTransformer, Dict[str, torch.Tensor], str]:
    model_path = os.path.join(save_dir, f"{data_type}_rvq_taae_model.pth")
    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")

    model = TrajRVQTransformer(
        input_steps=input_steps,
        input_dim=3,
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=num_transformer_layers,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    norm_params = load_norm_params(norm_path, device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])
    model.eval()
    return model, norm_params, model_path


def _encode_codes_for_indices(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    indices: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
) -> np.ndarray:
    subset = np.asarray(trajs[indices], dtype=np.float32)
    x_norm = normalize_trajs(
        subset,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
    )
    loader = DataLoader(
        TensorDataset(x_norm),
        batch_size=min(max(1, int(batch_size)), max(1, int(x_norm.shape[0]))),
        shuffle=False,
    )
    all_codes: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(norm_params["mean"].device)
            z = model.encode(x)
            _, _, codes = model.rvq(z)
            all_codes.append(codes.cpu().numpy())
    return np.concatenate(all_codes, axis=0).astype(np.int64)


def evaluate_group_level_similar_or(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    group_id_per_sample: np.ndarray,
    representative_indices: np.ndarray,
    group_to_indices: List[np.ndarray],
    norm_params: Dict[str, torch.Tensor],
    max_groups_eval: int,
    max_members_per_group: int,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    """按 group 评估 Similar-Traj OR（核心指标）。"""
    group_sizes = np.asarray([idxs.size for idxs in group_to_indices], dtype=np.int64)
    valid_groups = np.where(group_sizes >= 2)[0]
    if valid_groups.size == 0:
        return {
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }

    rng = np.random.default_rng(seed)
    max_g = min(int(max_groups_eval), int(valid_groups.size))

    # 覆盖大小分布：按 group size 排序后均匀抽取。
    ordered = valid_groups[np.argsort(group_sizes[valid_groups])]
    if ordered.size <= max_g:
        chosen_groups = ordered
    else:
        pick = np.linspace(0, ordered.size - 1, max_g).round().astype(np.int64)
        chosen_groups = ordered[pick]
        rng.shuffle(chosen_groups)

    eval_indices = []
    group_entries: List[Dict[str, object]] = []
    for gid in chosen_groups.tolist():
        members = group_to_indices[int(gid)]
        anchor = int(representative_indices[int(gid)])
        others = members[members != anchor]
        if others.size == 0:
            continue
        take = min(int(max_members_per_group), int(others.size))
        # 采样组内成员（不包含 anchor）。
        if take < others.size:
            chosen_members = rng.choice(others, size=take, replace=False)
        else:
            chosen_members = others

        local = np.concatenate([[anchor], chosen_members.astype(np.int64)], axis=0)
        start = len(eval_indices)
        eval_indices.extend(local.tolist())
        end = len(eval_indices)
        group_entries.append(
            {
                "group_id": int(gid),
                "group_size": int(members.size),
                "anchor_idx": int(anchor),
                "slice": (start, end),
            }
        )

    if len(eval_indices) == 0:
        return {
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }

    eval_indices_np = np.asarray(eval_indices, dtype=np.int64)
    codes_all = _encode_codes_for_indices(
        model=model,
        trajs=trajs,
        indices=eval_indices_np,
        norm_params=norm_params,
        batch_size=batch_size,
    )

    per_group_or = []
    layer_or_collect = []
    or1_collect = []
    or3_collect = []
    orall_collect = []
    exact_collect = []

    for g in group_entries:
        s, e = g["slice"]
        codes = codes_all[s:e]
        if codes.shape[0] <= 1:
            continue
        anchor = codes[0]
        members = codes[1:]
        same = (members == anchor[None, :]).astype(np.float32)
        layer_or = same.mean(axis=0) * 100.0
        num_layers = int(layer_or.shape[0])

        or1 = float(layer_or[0]) if num_layers >= 1 else 0.0
        or3 = float(np.mean(layer_or[: min(3, num_layers)])) if num_layers > 0 else 0.0
        orall = float(np.mean(layer_or)) if num_layers > 0 else 0.0
        exact = float((same.sum(axis=1) == num_layers).mean() * 100.0)

        or1_collect.append(or1)
        or3_collect.append(or3)
        orall_collect.append(orall)
        exact_collect.append(exact)
        layer_or_collect.append(layer_or)

        per_group_or.append(
            {
                "group_id": int(g["group_id"]),
                "group_size": int(g["group_size"]),
                "anchor_idx": int(g["anchor_idx"]),
                "num_members_eval": int(members.shape[0]),
                "or_at_1": or1,
                "or_at_3": or3,
                "or_at_all": orall,
                "exact_all_layers_match_pct": exact,
                "layer_or": [float(v) for v in layer_or.tolist()],
            }
        )

    if not per_group_or:
        return {
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }

    layer_or_mean = np.mean(np.stack(layer_or_collect, axis=0), axis=0)
    return {
        "num_groups_evaluated": int(len(per_group_or)),
        "or_at_1": float(np.mean(or1_collect)),
        "or_at_3": float(np.mean(or3_collect)),
        "or_at_all": float(np.mean(orall_collect)),
        "exact_all_layers_match_pct": float(np.mean(exact_collect)),
        "layer_or_mean": [float(v) for v in layer_or_mean.tolist()],
        "per_group_or": per_group_or,
    }


def evaluate_selected_group_or(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    representative_indices: np.ndarray,
    group_to_indices: List[np.ndarray],
    norm_params: Dict[str, torch.Tensor],
    group_size: int,
    batch_size: int,
    seed: int,
) -> Optional[Dict[str, object]]:
    """单个诊断 group 的 OR（可视化/排障用，不是主指标）。"""
    rng = np.random.default_rng(seed)
    group_sizes = np.asarray([idxs.size for idxs in group_to_indices], dtype=np.int64)
    candidates = np.where(group_sizes >= 2)[0]
    if candidates.size == 0:
        return None

    # 优先找 size 足够大的组，否则选最大组。
    enough = candidates[group_sizes[candidates] >= max(2, int(group_size))]
    if enough.size > 0:
        gid = int(enough[np.argmax(group_sizes[enough])])
    else:
        gid = int(candidates[np.argmax(group_sizes[candidates])])

    members = group_to_indices[gid]
    anchor = int(representative_indices[gid])
    others = members[members != anchor]
    if others.size == 0:
        return None

    take = min(max(1, int(group_size) - 1), int(others.size))
    chosen = rng.choice(others, size=take, replace=False) if take < others.size else others
    eval_indices = np.concatenate([[anchor], chosen.astype(np.int64)], axis=0)

    codes = _encode_codes_for_indices(
        model=model,
        trajs=trajs,
        indices=eval_indices,
        norm_params=norm_params,
        batch_size=batch_size,
    )
    anchor_codes = codes[0]
    member_codes = codes[1:]
    same = (member_codes == anchor_codes[None, :]).astype(np.float32)
    layer_or = same.mean(axis=0) * 100.0
    l = int(layer_or.shape[0])

    return {
        "group_id": int(gid),
        "group_size": int(members.size),
        "anchor_idx": int(anchor),
        "sample_indices": [int(v) for v in eval_indices.tolist()],
        "or_at_1": float(layer_or[0]) if l >= 1 else 0.0,
        "or_at_3": float(np.mean(layer_or[: min(3, l)])) if l > 0 else 0.0,
        "or_at_all": float(np.mean(layer_or)) if l > 0 else 0.0,
        "exact_all_layers_match_pct": float((same.sum(axis=1) == l).mean() * 100.0),
        "layer_or": [float(v) for v in layer_or.tolist()],
        "codes": [[int(x) for x in row.tolist()] for row in codes],
    }


def _print_dataset_summary(stats: Dict[str, np.ndarray], num_steps: int) -> None:
    raw_count = int(stats["total_path_length"].shape[0])
    stationary_like_count = int(np.sum(stats["stationary_like"]))
    stationary_like_ratio = float(stationary_like_count / max(1, raw_count))

    print("=" * 80)
    print("Dataset Summary")
    print(f"raw_count: {raw_count}")
    print(f"stationary_like_count: {stationary_like_count}")
    print(f"stationary_like_ratio: {stationary_like_ratio:.6f}")
    print(f"trajectory_length: {num_steps}")
    print("=" * 80)


def _print_grouping_summary(
    grouping_method: str,
    group_feature: str,
    requested_num_groups: int,
    actual_num_groups: int,
    representative_count: int,
    group_sizes: np.ndarray,
) -> None:
    p = _percentiles(group_sizes.astype(np.float32))
    print("Grouping Summary")
    print(f"grouping_method: {grouping_method}")
    print(f"group_feature: {group_feature}")
    print(f"requested_num_groups: {requested_num_groups}")
    print(f"actual_num_groups: {actual_num_groups}")
    print(f"representative_count: {representative_count}")
    print(
        "group_size p0/p25/p50/p75/p95/p99/max: "
        f"{p['p0']:.2f}/{p['p25']:.2f}/{p['p50']:.2f}/{p['p75']:.2f}/{p['p95']:.2f}/{p['p99']:.2f}/{p['max']:.2f}"
    )
    print("=" * 80)


def _print_training_summary(
    experiment_mode: str,
    train_unique_count: int,
    train_data_source: str,
    epochs: int,
    batch_size: int,
) -> None:
    print("Training Summary")
    print(f"experiment_mode: {experiment_mode}")
    print(f"train_unique_count: {train_unique_count}")
    print(f"train_data_source: {train_data_source}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative/full/consistency tokenizer experiments.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])

    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="representative_group",
        choices=["representative_group", "full_data", "similar_consistency"],
    )

    parser.add_argument("--num-representative-groups", type=int, default=80000)
    parser.add_argument(
        "--grouping-method",
        type=str,
        default="minibatch_kmeans",
        choices=["minibatch_kmeans", "kinematic_bins"],
    )
    parser.add_argument(
        "--group-feature",
        type=str,
        default="kinematic_plus_shape",
        choices=["kinematic", "shape", "kinematic_plus_shape"],
    )
    parser.add_argument("--shape-downsample-steps", type=int, default=10)
    parser.add_argument("--feature-xy-weight", type=float, default=1.0)
    parser.add_argument("--feature-yaw-weight", type=float, default=3.0)

    parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-random-state", type=int, default=42)

    parser.add_argument("--full-train-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--consistency-train-base",
        type=str,
        default="representative_group",
        choices=["representative_group", "full_data"],
    )
    parser.add_argument("--positive-groups-per-step", type=int, default=8)
    parser.add_argument("--positives-per-group", type=int, default=4)
    parser.add_argument("--negative-groups-per-step", type=int, default=16)

    parser.add_argument("--lambda-latent-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-soft-code-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-supcon", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--soft-code-temperature", type=float, default=0.2)
    parser.add_argument("--consistency-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--consistency-target",
        type=str,
        default="latent_plus_soft_code",
        choices=["latent", "soft_primary_code", "latent_plus_soft_code"],
    )

    parser.add_argument("--report-similar-or", action="store_true")
    parser.add_argument("--report-noise-or", action="store_true")
    parser.add_argument("--noise-std-xy", type=float, default=0.05)
    parser.add_argument("--noise-std-yaw", type=float, default=0.01)

    parser.add_argument("--max-groups-eval", type=int, default=1000)
    parser.add_argument("--max-members-per-group", type=int, default=8)

    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--report-selected-group-or", action="store_true")

    parser.add_argument("--save-root", type=str, default="./work_dirs/tokenizer/similar_single_train")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--grouping-cache-dir", type=str, default=None)
    parser.add_argument("--disable-grouping-cache", action="store_true")
    parser.add_argument("--grouping-cache-key", type=str, default="")

    args = parser.parse_args()

    _set_seed(int(args.seed))

    run_name = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(args.save_root, f"{args.data_type}_{args.experiment_mode}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    trajs = load_sampled_datas(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    trajs = np.asarray(trajs, dtype=np.float32)

    n_total = int(trajs.shape[0])
    requested_num_groups = int(max(1, min(args.num_representative_groups, n_total)))

    motion_stats = compute_motion_stats(trajs, dt=float(args.dt))
    _print_dataset_summary(motion_stats, num_steps=int(trajs.shape[1]))

    grouping = None
    grouping_loaded_from_cache = False

    # 分组 cache：优先读 output_dir 下已有结果；其次读全局 grouping_cache。
    cache_key = str(args.grouping_cache_key).strip()
    if not cache_key:
        cache_key = _build_grouping_cache_key(
            data_path=args.data_path,
            data_type=args.data_type,
            n_total=n_total,
            num_steps=int(trajs.shape[1]),
            requested_num_groups=requested_num_groups,
            grouping_method=args.grouping_method,
            group_feature=args.group_feature,
            shape_downsample_steps=int(args.shape_downsample_steps),
            feature_xy_weight=float(args.feature_xy_weight),
            feature_yaw_weight=float(args.feature_yaw_weight),
            dt=float(args.dt),
            kmeans_batch_size=int(args.kmeans_batch_size),
            kmeans_max_iter=int(args.kmeans_max_iter),
            kmeans_random_state=int(args.kmeans_random_state),
            seed=int(args.seed),
        )

    grouping_cache_root = args.grouping_cache_dir or os.path.join(args.save_root, "grouping_cache")
    grouping_cache_dir = os.path.join(grouping_cache_root, cache_key)

    if not args.disable_grouping_cache:
        grouping = _try_load_grouping_from_cache(output_dir, n_total=n_total)
        if grouping is not None:
            grouping_loaded_from_cache = True
            print(f"[cache] Loaded grouping from output_dir: {output_dir}")
        else:
            grouping = _try_load_grouping_from_cache(grouping_cache_dir, n_total=n_total)
            if grouping is not None:
                grouping_loaded_from_cache = True
                print(f"[cache] Loaded grouping from cache_dir: {grouping_cache_dir}")

    if grouping is None:
        features = build_group_features(
            trajs=trajs,
            dt=float(args.dt),
            group_feature=args.group_feature,
            shape_downsample_steps=int(args.shape_downsample_steps),
            xy_weight=float(args.feature_xy_weight),
            yaw_weight=float(args.feature_yaw_weight),
            motion_stats=motion_stats,
        )

        if args.grouping_method == "minibatch_kmeans":
            grouping = find_representative_groups_minibatch_kmeans(
                trajs=trajs,
                features=features,
                num_groups=requested_num_groups,
                batch_size=int(args.kmeans_batch_size),
                max_iter=int(args.kmeans_max_iter),
                seed=int(args.kmeans_random_state),
                motion_stats=motion_stats,
            )
        else:
            grouping = find_representative_groups_kinematic_bins(
                trajs=trajs,
                features=features,
                motion_stats=motion_stats,
                num_groups=requested_num_groups,
                seed=int(args.seed),
            )

        if not args.disable_grouping_cache:
            cache_meta = {
                "grouping_method_used": grouping["grouping_method_used"],
                "grouping_method_requested": args.grouping_method,
                "group_feature": args.group_feature,
                "cache_key": cache_key,
                "raw_count": int(n_total),
                "num_steps": int(trajs.shape[1]),
                "requested_num_groups": int(requested_num_groups),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_grouping_cache(output_dir, grouping, meta=cache_meta)
            _save_grouping_cache(grouping_cache_dir, grouping, meta=cache_meta)
            print(f"[cache] Saved grouping cache to: {grouping_cache_dir}")
    else:
        # 缓存命中时也同步一份到当前 output_dir，保证本次实验目录完整。
        if not args.disable_grouping_cache:
            cache_meta = {
                "grouping_method_used": grouping["grouping_method_used"],
                "grouping_method_requested": args.grouping_method,
                "group_feature": args.group_feature,
                "cache_key": cache_key,
                "raw_count": int(n_total),
                "num_steps": int(trajs.shape[1]),
                "requested_num_groups": int(requested_num_groups),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "loaded_from_cache": True,
            }
            _save_grouping_cache(output_dir, grouping, meta=cache_meta)

    group_id_per_sample = np.asarray(grouping["group_id_per_sample"], dtype=np.int32)
    representative_indices = np.asarray(grouping["representative_indices"], dtype=np.int64)
    group_sizes = np.asarray(grouping["group_sizes"], dtype=np.int64)
    rep_dist = np.asarray(grouping["representative_distance_to_center"], dtype=np.float32)
    group_to_indices: List[np.ndarray] = grouping["group_to_indices"]
    actual_num_groups = int(grouping["actual_num_groups"])
    grouping_method_used = str(grouping["grouping_method_used"])

    if representative_indices.shape[0] < int(0.8 * requested_num_groups):
        print(
            "[warning] representative_count 低于 requested_num_groups 的 80%，"
            f"requested={requested_num_groups}, actual={representative_indices.shape[0]}"
        )

    _print_grouping_summary(
        grouping_method=grouping_method_used,
        group_feature=args.group_feature,
        requested_num_groups=requested_num_groups,
        actual_num_groups=actual_num_groups,
        representative_count=int(representative_indices.shape[0]),
        group_sizes=group_sizes,
    )

    representative_indices_path = os.path.join(output_dir, "representative_indices.npy")
    group_id_per_sample_path = os.path.join(output_dir, "group_id_per_sample.npy")
    group_sizes_path = os.path.join(output_dir, "group_sizes.npy")

    np.save(representative_indices_path, representative_indices)
    np.save(group_id_per_sample_path, group_id_per_sample)
    np.save(group_sizes_path, group_sizes)

    group_summary = {
        "grouping_method": grouping_method_used,
        "group_feature": args.group_feature,
        "requested_num_groups": int(requested_num_groups),
        "actual_num_groups": int(actual_num_groups),
        "representative_count": int(representative_indices.shape[0]),
        "loaded_from_cache": bool(grouping_loaded_from_cache),
        "cache_key": cache_key,
        "grouping_cache_dir": grouping_cache_dir,
        "group_size_percentiles": _percentiles(group_sizes.astype(np.float32)),
        "representative_distance_to_center_percentiles": _percentiles(rep_dist.astype(np.float32)),
        "representative_indices_path": representative_indices_path,
        "group_id_per_sample_path": group_id_per_sample_path,
        "group_sizes_path": group_sizes_path,
    }
    _write_json(os.path.join(output_dir, "group_summary.json"), group_summary)

    if args.experiment_mode == "representative_group":
        train_indices = representative_indices.copy()
        train_trajs = trajs[train_indices]
        train_data_source = "representative_group"
        train_batch_size = int(args.batch_size)
    elif args.experiment_mode == "full_data":
        train_indices = np.arange(n_total, dtype=np.int64)
        train_trajs = trajs
        train_data_source = "full_data"
        train_batch_size = int(args.full_train_batch_size)
    elif args.experiment_mode == "similar_consistency":
        if args.consistency_train_base == "representative_group":
            train_indices = representative_indices.copy()
            train_trajs = trajs[train_indices]
            train_data_source = "representative_group"
            train_batch_size = int(args.batch_size)
        else:
            train_indices = np.arange(n_total, dtype=np.int64)
            train_trajs = trajs
            train_data_source = "full_data"
            train_batch_size = int(args.full_train_batch_size)
    else:
        raise ValueError(f"Unsupported experiment_mode: {args.experiment_mode}")

    train_indices_path = os.path.join(output_dir, "train_indices.npy")
    np.save(train_indices_path, train_indices)

    _print_training_summary(
        experiment_mode=args.experiment_mode,
        train_unique_count=int(train_indices.shape[0]),
        train_data_source=train_data_source,
        epochs=int(args.epochs),
        batch_size=int(train_batch_size),
    )

    robust_training_losses = None
    if args.experiment_mode in ("representative_group", "full_data"):
        train_rvq_taae(
            data_array=train_trajs,
            save_dir=output_dir,
            data_type=args.data_type,
            batch_size=train_batch_size,
            num_layers=int(args.num_layers),
            num_transformer_layers=int(args.num_transformer_layers),
            epochs=int(args.epochs),
        )
    else:
        robust_training_losses = train_rvq_taae_with_group_consistency(
            base_train_trajs=train_trajs,
            all_trajs=trajs,
            group_id_per_sample=group_id_per_sample,
            group_to_indices=group_to_indices,
            save_dir=output_dir,
            data_type=args.data_type,
            batch_size=train_batch_size,
            num_layers=int(args.num_layers),
            num_transformer_layers=int(args.num_transformer_layers),
            epochs=int(args.epochs),
            positive_groups_per_step=int(args.positive_groups_per_step),
            positives_per_group=int(args.positives_per_group),
            negative_groups_per_step=int(args.negative_groups_per_step),
            lambda_latent_consistency=float(args.lambda_latent_consistency),
            lambda_soft_code_consistency=float(args.lambda_soft_code_consistency),
            lambda_supcon=float(args.lambda_supcon),
            contrastive_temperature=float(args.contrastive_temperature),
            soft_code_temperature=float(args.soft_code_temperature),
            consistency_warmup_epochs=int(args.consistency_warmup_epochs),
            consistency_target=args.consistency_target,
            seed=int(args.seed),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_params, model_path = build_model_and_norm(
        save_dir=output_dir,
        data_type=args.data_type,
        input_steps=int(trajs.shape[1]),
        num_layers=int(args.num_layers),
        num_transformer_layers=int(args.num_transformer_layers),
        device=device,
    )

    # reconstruction 指标：在训练数据上评估。
    recon_trajs, _ = reconstruct_trajs(
        model=model,
        trajs=train_trajs,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=max(1, int(args.eval_batch_size)),
    )
    reconstruction_metrics = scenario_metrics(train_trajs, recon_trajs, dt=float(args.dt))

    group_level_similar_or = None
    group_level_similar_or_path = os.path.join(output_dir, "group_level_similar_or_summary.json")
    if args.report_similar_or:
        group_level_similar_or = evaluate_group_level_similar_or(
            model=model,
            trajs=trajs,
            group_id_per_sample=group_id_per_sample,
            representative_indices=representative_indices,
            group_to_indices=group_to_indices,
            norm_params=norm_params,
            max_groups_eval=int(args.max_groups_eval),
            max_members_per_group=int(args.max_members_per_group),
            batch_size=max(1, int(args.eval_batch_size)),
            seed=int(args.seed),
        )
    else:
        group_level_similar_or = {
            "enabled": False,
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }
    _write_json(group_level_similar_or_path, group_level_similar_or)

    selected_group_or = None
    if args.report_selected_group_or:
        selected_group_or = evaluate_selected_group_or(
            model=model,
            trajs=trajs,
            representative_indices=representative_indices,
            group_to_indices=group_to_indices,
            norm_params=norm_params,
            group_size=int(args.group_size),
            batch_size=max(1, int(args.eval_batch_size)),
            seed=int(args.seed),
        )

    tokenizer_health = None
    if args.report_noise_or:
        eval_norm = normalize_trajs(
            train_trajs,
            mean=norm_params["mean"],
            std=norm_params["std"],
            scale_factor=norm_params["scale_factor"],
            clip_limit=norm_params["clip_limit"],
        )
        eval_loader = DataLoader(
            TensorDataset(eval_norm),
            batch_size=min(max(1, int(args.eval_batch_size)), max(1, int(train_trajs.shape[0]))),
            shuffle=False,
        )
        util, overlap = evaluate_tokenizer_health(
            model=model,
            dataloader=eval_loader,
            device=device,
            noise_std_xy=float(args.noise_std_xy),
            noise_std_yaw=float(args.noise_std_yaw),
            clip_limit=norm_params["clip_limit"],
            vis_dt=float(args.dt),
        )
        tokenizer_health = {
            "avg_codebook_utilization_pct": float(util),
            "avg_overlap_rate_pct": float(overlap),
            "noise_std_xy": float(args.noise_std_xy),
            "noise_std_yaw": float(args.noise_std_yaw),
        }

    if robust_training_losses is not None:
        _write_csv(robust_training_losses, os.path.join(output_dir, "train_loss_history.csv"))

    dataset_summary = {
        "raw_count": int(n_total),
        "stationary_like_count": int(np.sum(motion_stats["stationary_like"])),
        "stationary_like_ratio": float(np.mean(motion_stats["stationary_like"].astype(np.float32))),
        "motion_stats_percentiles": {
            "total_path_length": _percentiles(motion_stats["total_path_length"]),
            "mean_speed": _percentiles(motion_stats["mean_speed"]),
            "abs_yaw_sum": _percentiles(motion_stats["abs_yaw_sum"]),
        },
    }

    training_summary = {
        "experiment_mode": args.experiment_mode,
        "train_unique_count": int(train_indices.shape[0]),
        "train_indices_path": train_indices_path,
        "train_data_source": train_data_source,
        "epochs": int(args.epochs),
        "batch_size": int(train_batch_size),
    }

    summary = {
        "dataset": dataset_summary,
        "grouping": group_summary,
        "training": training_summary,
        "metrics": {
            "reconstruction_metrics": reconstruction_metrics,
            "group_level_similar_or": group_level_similar_or,
            "selected_group_or": selected_group_or,
            "tokenizer_health": tokenizer_health,
            "robust_training_losses": robust_training_losses,
        },
    }

    _write_json(os.path.join(output_dir, "dataset_training_summary.json"), {
        "dataset": dataset_summary,
        "training": training_summary,
    })
    summary_path = os.path.join(output_dir, "similar_single_train_summary.json")
    _write_json(summary_path, summary)

    print("=" * 80)
    print("Experiment Done")
    print(f"model_path: {model_path}")
    print(f"summary_path: {summary_path}")

    if group_level_similar_or is not None:
        print("Group-level Similar-Traj OR:")
        print(f"  OR@1: {float(group_level_similar_or.get('or_at_1', 0.0)):.4f}")
        print(f"  OR@3: {float(group_level_similar_or.get('or_at_3', 0.0)):.4f}")
        print(f"  OR@All: {float(group_level_similar_or.get('or_at_all', 0.0)):.4f}")
        print(f"  exact_all_layers_match_pct: {float(group_level_similar_or.get('exact_all_layers_match_pct', 0.0)):.4f}")

    if tokenizer_health is not None:
        print("Noise OR:")
        print(f"  avg_codebook_utilization_pct: {float(tokenizer_health['avg_codebook_utilization_pct']):.4f}")
        print(f"  avg_overlap_rate_pct: {float(tokenizer_health['avg_overlap_rate_pct']):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
