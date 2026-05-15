import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from utils import build_scenario_masks
except ImportError:
    from rvq_transformer_vehdyn.utils import build_scenario_masks


def _to_py(obj):
    """递归转成 Python 原生类型，确保 json.dump 稳定。"""
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
    return obj


def _write_json(path: str, data: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_py(data), f, indent=2)


def _safe_float_str(v: float) -> str:
    return f"{float(v):.8f}"


def build_grouping_cache_key(
    *,
    data_path: Optional[str],
    data_type: str,
    n_total: int,
    num_steps: int,
    requested_num_groups: int,
    grouping_method: str,
    grouping_stage: str,
    group_feature: str,
    shape_downsample_steps: int,
    feature_xy_weight: float,
    feature_yaw_weight: float,
    dt: float,
    kmeans_batch_size: int,
    kmeans_max_iter: int,
    kmeans_n_init: int,
    kmeans_random_state: int,
    seed: int,
) -> str:
    """为分组结果生成稳定 cache key。"""
    payload = {
        "data_path": str(data_path),
        "data_type": str(data_type),
        "n_total": int(n_total),
        "num_steps": int(num_steps),
        "requested_num_groups": int(requested_num_groups),
        "grouping_method": str(grouping_method),
        "grouping_stage": str(grouping_stage),
        "group_feature": str(group_feature),
        "shape_downsample_steps": int(shape_downsample_steps),
        "feature_xy_weight": _safe_float_str(feature_xy_weight),
        "feature_yaw_weight": _safe_float_str(feature_yaw_weight),
        "dt": _safe_float_str(dt),
        "kmeans_batch_size": int(kmeans_batch_size),
        "kmeans_max_iter": int(kmeans_max_iter),
        "kmeans_n_init": int(kmeans_n_init),
        "kmeans_random_state": int(kmeans_random_state),
        "seed": int(seed),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


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


def try_load_grouping_from_cache(
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
    grouping_stage_used = "unknown"
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            grouping_method_used = str(meta.get("grouping_method_used", grouping_method_used))
            grouping_stage_used = str(meta.get("grouping_stage_used", grouping_stage_used))
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
        "grouping_stage_used": grouping_stage_used,
        "cache_key": str(meta.get("cache_key", "")),
        "cache_meta": meta,
        "scenario_partition_summary": [],
    }


def save_grouping_cache(
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
    n_init: int,
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

    def _fit_with_progress(kmeans_obj, x: np.ndarray, mb_size: int, n_epoch: int, seed0: int):
        if tqdm is None:
            kmeans_obj.fit(x)
            return
        n_samples = int(x.shape[0])
        effective_mb_size = max(int(mb_size), int(target))
        if effective_mb_size != int(mb_size):
            print(
                f"[warning] kmeans_batch_size({int(mb_size)}) < n_clusters({int(target)}), "
                f"auto bump to {effective_mb_size} for partial_fit."
            )
        rng = np.random.default_rng(seed0)
        steps_per_epoch = (n_samples + effective_mb_size - 1) // effective_mb_size
        total_steps = max(1, int(n_epoch)) * max(1, int(steps_per_epoch))
        with tqdm(total=total_steps, desc="KMeans partial_fit", leave=True) as pbar:
            for _ in range(max(1, int(n_epoch))):
                perm = rng.permutation(n_samples)
                for start in range(0, n_samples, effective_mb_size):
                    end = min(start + effective_mb_size, n_samples)
                    batch_x = x[perm[start:end]]
                    if batch_x.shape[0] < int(target):
                        need = int(target) - int(batch_x.shape[0])
                        extra_idx = rng.choice(n_samples, size=need, replace=False)
                        batch_x = np.concatenate([batch_x, x[extra_idx]], axis=0)
                    kmeans_obj.partial_fit(batch_x)
                    pbar.update(1)

    def _predict_with_progress(kmeans_obj, x: np.ndarray, chunk_size: int) -> np.ndarray:
        n_samples = int(x.shape[0])
        labels = np.empty((n_samples,), dtype=np.int32)
        if tqdm is None:
            return kmeans_obj.predict(x).astype(np.int32)
        with tqdm(total=n_samples, desc="KMeans predict", leave=True) as pbar:
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                labels[start:end] = kmeans_obj.predict(x[start:end]).astype(np.int32)
                pbar.update(end - start)
        return labels

    if int(n_init) > 0:
        kmeans = _build_kmeans(int(n_init))
    else:
        kmeans = _build_kmeans("auto")
    try:
        if tqdm is None:
            labels_raw = kmeans.fit_predict(features).astype(np.int32)
        else:
            _fit_with_progress(
                kmeans_obj=kmeans,
                x=features,
                mb_size=max(256, int(batch_size)),
                n_epoch=max(10, int(max_iter)),
                seed0=seed,
            )
            labels_raw = _predict_with_progress(
                kmeans_obj=kmeans,
                x=features,
                chunk_size=max(8192, int(batch_size)),
            )
        centers_raw = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    except Exception:
        if not isinstance(getattr(kmeans, "n_init", None), str):
            raise
        fallback_n_init = max(1, int(n_init)) if int(n_init) > 0 else 1
        print(f"[warning] sklearn 版本不支持 n_init='auto'，自动回退到 n_init={fallback_n_init}。")
        kmeans = _build_kmeans(fallback_n_init)
        if tqdm is None:
            labels_raw = kmeans.fit_predict(features).astype(np.int32)
        else:
            _fit_with_progress(
                kmeans_obj=kmeans,
                x=features,
                mb_size=max(256, int(batch_size)),
                n_epoch=max(10, int(max_iter)),
                seed0=seed,
            )
            labels_raw = _predict_with_progress(
                kmeans_obj=kmeans,
                x=features,
                chunk_size=max(8192, int(batch_size)),
            )
        centers_raw = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    unique_labels, group_id_per_sample = np.unique(labels_raw, return_inverse=True)
    centers = centers_raw[unique_labels]
    num_actual = int(unique_labels.shape[0])

    group_to_indices = _build_group_to_indices(group_id_per_sample.astype(np.int32), num_actual)

    reps = np.zeros((num_actual,), dtype=np.int64)
    rep_dist = np.zeros((num_actual,), dtype=np.float32)
    group_sizes = np.zeros((num_actual,), dtype=np.int64)

    gid_iter = range(num_actual)
    if tqdm is not None:
        gid_iter = tqdm(gid_iter, total=num_actual, desc="Select representatives", leave=True)
    for gid in gid_iter:
        idxs = group_to_indices[gid]
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


def _scenario_partition_indices(trajs: np.ndarray, dt: float) -> List[Tuple[str, np.ndarray]]:
    """先按场景切分：复用 scenario 评估脚本共用的场景判别逻辑。"""
    fps = 1.0 / max(float(dt), 1e-6)
    categories, _ = build_scenario_masks(trajs, fps=fps)
    parts: List[Tuple[str, np.ndarray]] = []

    used = np.zeros((trajs.shape[0],), dtype=bool)
    for name, mask in categories.items():
        m = np.asarray(mask, dtype=bool)
        idx = np.flatnonzero(m & (~used)).astype(np.int64)
        if idx.size > 0:
            parts.append((str(name), idx))
            used[idx] = True

    other_idx = np.flatnonzero(~used).astype(np.int64)
    if other_idx.size > 0:
        parts.append(("Other", other_idx))
    return parts


def _allocate_group_counts(parts: List[Tuple[str, np.ndarray]], target: int) -> List[int]:
    sizes = np.asarray([int(idx.size) for _, idx in parts], dtype=np.int64)
    non_empty = np.flatnonzero(sizes > 0)
    if target <= 0:
        return [0 for _ in parts]
    if non_empty.size == 0:
        return [0 for _ in parts]
    if target < non_empty.size:
        raise ValueError(
            f"requested_num_groups({target}) < non_empty_scenarios({int(non_empty.size)}), "
            "请增大组数或改用 global 分组。"
        )

    alloc = np.zeros_like(sizes)
    alloc[non_empty] = 1
    remain = int(target - non_empty.size)
    if remain <= 0:
        return alloc.tolist()

    weights = sizes.astype(np.float64)
    wsum = float(np.sum(weights))
    share = (weights / max(wsum, 1e-12)) * remain
    add_floor = np.floor(share).astype(np.int64)
    alloc += add_floor
    left = int(remain - int(np.sum(add_floor)))

    frac = share - add_floor
    order = np.argsort(-frac)
    ptr = 0
    while left > 0 and ptr < order.size:
        i = int(order[ptr])
        alloc[i] += 1
        left -= 1
        ptr += 1

    cap = sizes.copy()
    overflow = np.maximum(alloc - cap, 0)
    overflow_total = int(np.sum(overflow))
    if overflow_total > 0:
        alloc = np.minimum(alloc, cap)
        spare = cap - alloc
        spare_order = np.argsort(-spare)
        for i in spare_order.tolist():
            if overflow_total <= 0:
                break
            add = int(min(spare[i], overflow_total))
            if add > 0:
                alloc[i] += add
                overflow_total -= add

    return alloc.astype(np.int64).tolist()


def find_representative_groups(
    trajs: np.ndarray,
    dt: float,
    num_groups: int,
    grouping_method: str,
    grouping_stage: str,
    group_feature: str,
    shape_downsample_steps: int,
    feature_xy_weight: float,
    feature_yaw_weight: float,
    kmeans_batch_size: int,
    kmeans_max_iter: int,
    kmeans_n_init: int,
    kmeans_random_state: int,
    seed: int,
    motion_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """
    统一分组入口：
    - global: 全量一次聚类
    - scenario_first: 先按场景切分，再场景内聚类，最后合并。
    """
    x = np.asarray(trajs, dtype=np.float32)
    n = int(x.shape[0])
    target = int(max(1, min(int(num_groups), n)))
    if motion_stats is None:
        motion_stats = compute_motion_stats(x, dt=float(dt))

    features = build_group_features(
        trajs=x,
        dt=float(dt),
        group_feature=group_feature,
        shape_downsample_steps=int(shape_downsample_steps),
        xy_weight=float(feature_xy_weight),
        yaw_weight=float(feature_yaw_weight),
        motion_stats=motion_stats,
    )

    def _group_subset(sub_idx: np.ndarray, sub_target: int, sub_seed: int) -> Dict[str, object]:
        sub_trajs = x[sub_idx]
        sub_feat = features[sub_idx]
        sub_stats = {k: np.asarray(v)[sub_idx] for k, v in motion_stats.items()}
        if grouping_method == "minibatch_kmeans":
            return find_representative_groups_minibatch_kmeans(
                trajs=sub_trajs,
                features=sub_feat,
                num_groups=int(sub_target),
                batch_size=int(kmeans_batch_size),
                max_iter=int(kmeans_max_iter),
                n_init=int(kmeans_n_init),
                seed=int(sub_seed),
                motion_stats=sub_stats,
            )
        return find_representative_groups_kinematic_bins(
            trajs=sub_trajs,
            features=sub_feat,
            motion_stats=sub_stats,
            num_groups=int(sub_target),
            seed=int(sub_seed),
        )

    if grouping_stage == "global":
        out = _group_subset(np.arange(n, dtype=np.int64), target, int(kmeans_random_state))
        out["grouping_stage_used"] = "global"
        out["scenario_partition_summary"] = []
        return out

    parts = _scenario_partition_indices(x, dt=float(dt))
    alloc = _allocate_group_counts(parts, target)

    global_gid = np.full((n,), -1, dtype=np.int32)
    representatives: List[int] = []
    group_sizes: List[int] = []
    rep_dist: List[float] = []
    group_to_indices: List[np.ndarray] = []
    scenario_partition_summary: List[Dict[str, object]] = []

    gid_offset = 0
    for i, (name, idx) in enumerate(parts):
        sub_target = int(alloc[i])
        if idx.size == 0 or sub_target <= 0:
            scenario_partition_summary.append({
                "scenario": str(name),
                "sample_count": int(idx.size),
                "allocated_groups": int(sub_target),
                "actual_groups": 0,
            })
            continue

        sub = _group_subset(idx, min(sub_target, int(idx.size)), int(seed + i * 17))
        sub_gid = np.asarray(sub["group_id_per_sample"], dtype=np.int32)
        sub_rep_local = np.asarray(sub["representative_indices"], dtype=np.int64)
        sub_gsz = np.asarray(sub["group_sizes"], dtype=np.int64)
        sub_rep_dist = np.asarray(sub["representative_distance_to_center"], dtype=np.float32)
        sub_groups: List[np.ndarray] = sub["group_to_indices"]
        sub_k = int(sub["actual_num_groups"])

        global_gid[idx] = (sub_gid + gid_offset).astype(np.int32)

        for g in range(sub_k):
            rep_global = int(idx[sub_rep_local[g]])
            representatives.append(rep_global)
            group_sizes.append(int(sub_gsz[g]))
            rep_dist.append(float(sub_rep_dist[g]))
            group_to_indices.append(idx[sub_groups[g]].astype(np.int64))

        scenario_partition_summary.append({
            "scenario": str(name),
            "sample_count": int(idx.size),
            "allocated_groups": int(sub_target),
            "actual_groups": int(sub_k),
        })
        gid_offset += sub_k

    if np.any(global_gid < 0):
        miss = np.flatnonzero(global_gid < 0)
        # 理论上不会走到这里；兜底用最接近代表组的方式分配到最后一个组。
        if gid_offset <= 0:
            raise RuntimeError("No valid groups were created.")
        global_gid[miss] = int(gid_offset - 1)
        group_to_indices[-1] = np.concatenate([group_to_indices[-1], miss.astype(np.int64)], axis=0)
        group_sizes[-1] = int(group_to_indices[-1].size)

    return {
        "group_id_per_sample": global_gid.astype(np.int32),
        "representative_indices": np.asarray(representatives, dtype=np.int64),
        "group_sizes": np.asarray(group_sizes, dtype=np.int64),
        "representative_distance_to_center": np.asarray(rep_dist, dtype=np.float32),
        "group_to_indices": group_to_indices,
        "actual_num_groups": int(gid_offset),
        "grouping_method_used": str(grouping_method),
        "grouping_stage_used": "scenario_first",
        "scenario_partition_summary": scenario_partition_summary,
    }
