import argparse
import csv
import json
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_tfm import TrajRVQTransformer


def load_trajs(data_path: str) -> np.ndarray:
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if isinstance(data, dict) and "trajs" in data:
        return np.asarray(data["trajs"], dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def load_norm_params(norm_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(norm_path, "rb") as f:
        norm_params = pickle.load(f)
    out = {
        "mean": torch.tensor(norm_params["mean"], dtype=torch.float32, device=device),
        "std": torch.tensor(norm_params["std"], dtype=torch.float32, device=device),
        "scale_factor": torch.tensor(norm_params["scale_factor"], dtype=torch.float32, device=device),
    }
    out["clip_limit"] = (
        torch.tensor(norm_params["clip_limit"], dtype=torch.float32, device=device)
        if "clip_limit" in norm_params
        else None
    )
    return out


def normalize_trajs(trajs: np.ndarray, norm_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = norm_params["mean"]
    x = torch.tensor(trajs, dtype=torch.float32, device=mean.device)
    x_norm = (x - mean) / (norm_params["std"] + 1e-8)
    if norm_params["clip_limit"] is not None:
        x_norm = torch.clamp(x_norm, -norm_params["clip_limit"], norm_params["clip_limit"])
    return x_norm / norm_params["scale_factor"]


def build_model(model_path: str, input_steps: int, device: torch.device) -> TrajRVQTransformer:
    model = TrajRVQTransformer(
        input_steps=input_steps,
        input_dim=3,
        num_layers=15,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def encode_codes(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
) -> np.ndarray:
    dataset = TensorDataset(normalize_trajs(trajs, norm_params).cpu())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    codes_all = []
    with torch.no_grad():
        for (x_cpu,) in dataloader:
            x = x_cpu.to(next(model.parameters()).device)
            z = model.encode(x)
            _, _, codes = model.rvq(z)
            codes_all.append(codes.cpu().numpy())
    return np.concatenate(codes_all, axis=0)


def integrate_to_global(trajs: np.ndarray) -> np.ndarray:
    dx = trajs[:, :, 0]
    dy = trajs[:, :, 1]
    dyaw = trajs[:, :, 2]
    yaw = np.cumsum(dyaw, axis=1)
    prev_yaw = np.zeros_like(yaw)
    prev_yaw[:, 1:] = yaw[:, :-1]
    dx_global = dx * np.cos(prev_yaw) - dy * np.sin(prev_yaw)
    dy_global = dx * np.sin(prev_yaw) + dy * np.cos(prev_yaw)
    return np.stack([np.cumsum(dx_global, axis=1), np.cumsum(dy_global, axis=1)], axis=-1)


def compute_features(trajs: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    dx = trajs[:, :, 0]
    dy = trajs[:, :, 1]
    dyaw = trajs[:, :, 2]
    speed = np.sqrt(dx * dx + dy * dy) / dt
    acc = np.diff(speed, axis=1) / dt
    xy = integrate_to_global(trajs)
    step_dist = np.sqrt(np.sum(np.diff(np.pad(xy, ((0, 0), (1, 0), (0, 0))), axis=1) ** 2, axis=-1))
    return {
        "avg_speed_mps": speed.mean(axis=1),
        "max_speed_mps": speed.max(axis=1),
        "speed_std_mps": speed.std(axis=1),
        "acc_mean_mps2": acc.mean(axis=1) if acc.shape[1] > 0 else np.zeros(len(trajs)),
        "acc_std_mps2": acc.std(axis=1) if acc.shape[1] > 0 else np.zeros(len(trajs)),
        "net_yaw_rad": dyaw.sum(axis=1),
        "gross_yaw_rad": np.abs(dyaw).sum(axis=1),
        "total_dist_m": step_dist.sum(axis=1),
        "final_x_m": xy[:, -1, 0],
        "final_y_m": xy[:, -1, 1],
    }


def feature_label(feature_name: str) -> str:
    labels = {
        "avg_speed_mps": "平均前进速度",
        "max_speed_mps": "最高速度",
        "speed_std_mps": "速度变化/加减速幅度",
        "acc_mean_mps2": "整体加速或减速趋势",
        "acc_std_mps2": "加速度变化/速度抖动",
        "net_yaw_rad": "左转/右转方向和净转角",
        "gross_yaw_rad": "转弯幅度/曲率强度",
        "total_dist_m": "总前进距离",
        "final_x_m": "终点纵向位置",
        "final_y_m": "终点横向偏移",
    }
    return labels.get(feature_name, feature_name)

def association_ratio(token_ids: np.ndarray, values: np.ndarray, min_count: int) -> float:
    total_var = float(np.var(values)) + 1e-12
    global_mean = float(np.mean(values))
    between = 0.0
    for token_id in np.unique(token_ids):
        mask = token_ids == token_id
        if int(mask.sum()) < min_count:
            continue
        diff = float(np.mean(values[mask])) - global_mean
        # 方差 = 组内方差 + 组间方差(between)；这里的association_ratio计算的就是between占总方差的比例，越高说明这个token位置对区分这个特征越重要
        between += float(mask.mean()) * diff * diff
    return between / total_var


def summarize_token_usage(
    codes: np.ndarray,
    features: Dict[str, np.ndarray],
    vocab_size: int,
    min_count: int,
    topk: int,
) -> Tuple[List[Dict], List[Dict]]:
    layer_rows = []
    top_code_rows = []
    num_layers = codes.shape[1]
    feature_names = list(features.keys())

    for layer in range(num_layers):
        layer_codes = codes[:, layer]
        counts = np.bincount(layer_codes, minlength=vocab_size)
        used_ids = np.where(counts > 0)[0]
        probs = counts / max(counts.sum(), 1)
        nz_probs = probs[probs > 0]
        entropy = -np.sum(nz_probs * np.log(nz_probs))
        perplexity = float(np.exp(entropy))

        ratios = {
            name: association_ratio(layer_codes, values, min_count=min_count)
            for name, values in features.items()
        }
        strongest_feature = max(ratios, key=ratios.get)

        top_ids = np.argsort(-counts)[:topk]
        layer_rows.append(
            {
                "layer": layer,
                "used_codes": int(len(used_ids)),
                "vocab_size": int(vocab_size),
                "utilization_pct": float(len(used_ids) / vocab_size * 100),
                "top1_pct": float(probs[top_ids[0]] * 100) if len(top_ids) > 0 else 0.0,
                "top5_pct": float(probs[top_ids[:5]].sum() * 100),
                "perplexity": perplexity,
                "strongest_feature": strongest_feature,
                "meaning_guess": feature_label(strongest_feature),
                "association_score": float(ratios[strongest_feature]),
                **{f"assoc_{name}": float(score) for name, score in ratios.items()},
            }
        )

        for token_id in top_ids:
            if counts[token_id] == 0:
                continue
            mask = layer_codes == token_id
            row = {
                "layer": layer,
                "code_id": int(token_id),
                "count": int(counts[token_id]),
                "rate_pct": float(probs[token_id] * 100),
            }
            for name, values in features.items():
                row[f"mean_{name}"] = float(values[mask].mean())
            top_code_rows.append(row)

    return layer_rows, top_code_rows


def decode_from_codes_phys(
    model: TrajRVQTransformer,
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
) -> np.ndarray:
    out = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(codes), batch_size):
            batch_codes = torch.tensor(codes[start:start + batch_size], dtype=torch.long, device=device)
            x_norm = model.decode_from_codes(batch_codes)
            x_phys = x_norm * norm_params["scale_factor"] * (norm_params["std"] + 1e-8) + norm_params["mean"]
            out.append(x_phys.cpu().numpy())
    return np.concatenate(out, axis=0)


def interventional_summary(
    model: TrajRVQTransformer,
    base_codes: np.ndarray,
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    dt: float,
    topk: int,
    batch_size: int,
) -> List[Dict]:
    rows = []
    vocab_size = model.vocab_size
    base_traj = decode_from_codes_phys(model, base_codes[None, :], norm_params, batch_size=1)
    base_feat = compute_features(base_traj, dt)
    feature_names = list(base_feat.keys())

    for layer in range(model.num_layers):
        counts = np.bincount(codes[:, layer], minlength=vocab_size)
        sweep_ids = np.argsort(-counts)[:topk]
        sweep_codes = np.repeat(base_codes[None, :], len(sweep_ids), axis=0)
        sweep_codes[:, layer] = sweep_ids
        sweep_trajs = decode_from_codes_phys(model, sweep_codes, norm_params, batch_size=batch_size)
        sweep_feat = compute_features(sweep_trajs, dt)

        delta_mean_abs = {
            name: float(np.mean(np.abs(sweep_feat[name] - base_feat[name][0])))
            for name in feature_names
        }
        strongest = max(delta_mean_abs, key=delta_mean_abs.get)
        rows.append(
            {
                "layer": layer,
                "swept_topk_codes": int(len(sweep_ids)),
                "strongest_intervention_feature": strongest,
                "intervention_meaning_guess": feature_label(strongest),
                **{f"mean_abs_delta_{name}": value for name, value in delta_mean_abs.items()},
            }
        )
    return rows


def build_scenario_masks(trajs: np.ndarray, dt: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fps = 1.0 / dt
    feat = compute_features(trajs, dt=dt)
    net_yaw = feat["net_yaw_rad"]
    gross_yaw = feat["gross_yaw_rad"]
    total_dist = feat["total_dist_m"]
    avg_speed = feat["avg_speed_mps"]

    th_dist_static = 1.0
    th_straight_net = 0.10
    th_straight_gross = 0.20
    v_10 = 10.0 / 3.6
    v_80 = 80.0 / 3.6
    v_120 = 120.0 / 3.6
    th_turn = 0.35
    th_uturn = 2.35

    sign_changes = np.zeros(len(trajs), dtype=np.int64)
    dyaw_sign = np.sign(trajs[:, :, 2])
    dyaw_sign[dyaw_sign == 0] = 1
    if trajs.shape[1] > 1:
        sign_changes = np.sum(dyaw_sign[:, 1:] != dyaw_sign[:, :-1], axis=1)

    mask_static = total_dist < th_dist_static
    mask_straight = (np.abs(net_yaw) < th_straight_net) & (gross_yaw < th_straight_gross) & (~mask_static)
    categories = {
        "Stationary": mask_static,
        "LowSpeedStraight_10kmh": mask_straight & (avg_speed >= (v_10 - 2 / 3.6)) & (avg_speed <= (v_10 + 2 / 3.6)),
        "HighSpeedStraight_80kmh": mask_straight & (avg_speed >= (v_80 - 10 / 3.6)) & (avg_speed <= (v_80 + 10 / 3.6)),
        "HighSpeedStraight_120kmh": mask_straight & (avg_speed >= (v_120 - 15 / 3.6)) & (avg_speed <= (v_120 + 15 / 3.6)),
        "LeftTurn": (net_yaw >= th_turn) & (np.abs(net_yaw) < th_uturn) & (~mask_static),
        "RightTurn": (net_yaw <= -th_turn) & (np.abs(net_yaw) < th_uturn) & (~mask_static),
        "Detour": (np.abs(net_yaw) < 0.20) & (gross_yaw >= 0.80) & (sign_changes >= 2) & (~mask_static),
        "UTurn": (np.abs(net_yaw) >= th_uturn) & (~mask_static),
    }
    feat["sign_changes"] = sign_changes
    feat["fps"] = np.full(len(trajs), fps)
    return categories, feat


def select_scenario_base_indices(categories: Dict[str, np.ndarray], features: Dict[str, np.ndarray]) -> Dict[str, int]:
    net_yaw = features["net_yaw_rad"]
    gross_yaw = features["gross_yaw_rad"]
    total_dist = features["total_dist_m"]
    avg_speed = features["avg_speed_mps"]
    sign_changes = features["sign_changes"]

    v_10 = 10.0 / 3.6
    v_80 = 80.0 / 3.6
    v_120 = 120.0 / 3.6
    out = {}
    for name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        if name == "Stationary":
            score = total_dist[idxs]
        elif name == "LowSpeedStraight_10kmh":
            score = np.abs(avg_speed[idxs] - v_10) + np.abs(net_yaw[idxs])
        elif name == "HighSpeedStraight_80kmh":
            score = np.abs(avg_speed[idxs] - v_80) + np.abs(net_yaw[idxs])
        elif name == "HighSpeedStraight_120kmh":
            score = np.abs(avg_speed[idxs] - v_120) + np.abs(net_yaw[idxs])
        elif name == "LeftTurn":
            score = np.abs(net_yaw[idxs] - 1.0) + 0.05 * np.abs(avg_speed[idxs] - v_10)
        elif name == "RightTurn":
            score = np.abs(net_yaw[idxs] + 1.0) + 0.05 * np.abs(avg_speed[idxs] - v_10)
        elif name == "Detour":
            score = np.abs(net_yaw[idxs]) + 1.0 / (gross_yaw[idxs] + 1e-6) + 1.0 / (sign_changes[idxs] + 1e-6)
        elif name == "UTurn":
            score = np.abs(np.abs(net_yaw[idxs]) - np.pi)
        else:
            score = np.zeros(len(idxs))
        out[name] = int(idxs[np.argmin(score)])
    return out


def select_local_neighbor_code_ids(codes: np.ndarray, layer: int, base_code: int, n_each_side: int = 3) -> np.ndarray:
    used_ids = np.unique(codes[:, layer]).astype(np.int64)
    used_ids.sort()
    base_code = int(base_code)

    lower = used_ids[used_ids < base_code]
    upper = used_ids[used_ids > base_code]

    def uniform_pick(pool: np.ndarray, count: int) -> List[int]:
        if len(pool) == 0 or count <= 0:
            return []
        if len(pool) <= count:
            return pool.tolist()
        positions = np.linspace(0, len(pool) - 1, count).round().astype(np.int64)
        return pool[positions].tolist()

    left = uniform_pick(lower, n_each_side)
    right = uniform_pick(upper, n_each_side)

    return np.asarray(sorted(set(left + [base_code] + right)), dtype=np.int64)


def _neighbor_rank(code_id: int, selected_ids: np.ndarray, base_code: int) -> str:
    code_id = int(code_id)
    base_code = int(base_code)
    selected = [int(x) for x in selected_ids]
    if code_id == base_code:
        return "base"
    left = [x for x in selected if x < base_code]
    right = [x for x in selected if x > base_code]
    if code_id < base_code:
        return f"-{len(left) - left.index(code_id)}"
    return f"+{right.index(code_id) + 1}"


def _build_neighbor_color_map(selected_ids: np.ndarray, base_code: int):
    blues = plt.get_cmap("Blues")
    oranges = plt.get_cmap("Oranges")
    selected = [int(x) for x in selected_ids]
    left = [x for x in selected if x < int(base_code)]
    right = [x for x in selected if x > int(base_code)]
    color_map = {int(base_code): "black"}
    for i, code_id in enumerate(left):
        denom = max(len(left) - 1, 1)
        color_map[code_id] = blues(0.45 + 0.45 * (i / denom))
    for i, code_id in enumerate(right):
        denom = max(len(right) - 1, 1)
        color_map[code_id] = oranges(0.45 + 0.45 * (i / denom))
    return color_map


def _smart_set_axis_limits(ax, xy: np.ndarray, pad_ratio: float = 0.08, min_span: float = 1.0):
    points = np.asarray(xy).reshape(-1, 2)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        ax.set_xlim(-min_span / 2, min_span / 2)
        ax.set_ylim(-min_span / 2, min_span / 2)
        ax.set_aspect("auto")
        return

    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    x_span = max(float(xmax - xmin), min_span)
    y_span = max(float(ymax - ymin), min_span)
    x_pad = x_span * pad_ratio
    y_pad = y_span * pad_ratio

    x_center = (float(xmin) + float(xmax)) / 2.0
    y_center = (float(ymin) + float(ymax)) / 2.0
    ax.set_xlim(x_center - x_span / 2.0 - x_pad, x_center + x_span / 2.0 + x_pad)
    ax.set_ylim(y_center - y_span / 2.0 - y_pad, y_center + y_span / 2.0 + y_pad)

    ratio = max(x_span, y_span) / max(min(x_span, y_span), 1e-6)
    if ratio > 8.0:
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")
def compute_series_diagnostics(trajs: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    xy = integrate_to_global(trajs)
    dx_global = np.diff(np.pad(xy[:, :, 0], ((0, 0), (1, 0))), axis=1)
    dy_global = np.diff(np.pad(xy[:, :, 1], ((0, 0), (1, 0))), axis=1)
    vx = dx_global / dt
    vy = dy_global / dt
    v = np.sqrt(vx * vx + vy * vy)
    ax = np.diff(vx, axis=1, prepend=vx[:, :1]) / dt
    ay = np.diff(vy, axis=1, prepend=vy[:, :1]) / dt
    a = np.sqrt(ax * ax + ay * ay)
    step_dist = np.sqrt(trajs[:, :, 0] ** 2 + trajs[:, :, 1] ** 2)
    kappa = trajs[:, :, 2] / (step_dist + 1e-6)
    return {"xy": xy, "vx": vx, "vy": vy, "v": v, "kappa": kappa, "ax": ax, "ay": ay, "a": a}


def _decode_layer_sweep(
    model: TrajRVQTransformer,
    base_codes: np.ndarray,
    codes: np.ndarray,
    layer: int,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
    n_each_side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    base_code = int(base_codes[layer])
    sweep_ids = select_local_neighbor_code_ids(codes, layer, base_code, n_each_side=n_each_side)
    sweep_codes = np.repeat(base_codes[None, :], len(sweep_ids), axis=0)
    sweep_codes[:, layer] = sweep_ids
    sweep_trajs = decode_from_codes_phys(model, sweep_codes, norm_params, batch_size=batch_size)
    return sweep_ids, sweep_trajs


def rank_layers_for_scenario(
    model: TrajRVQTransformer,
    base_codes: np.ndarray,
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
    n_each_side: int,
) -> List[Dict]:
    rows = []
    for layer in range(model.num_layers):
        sweep_ids, sweep_trajs = _decode_layer_sweep(model, base_codes, codes, layer, norm_params, batch_size, n_each_side)
        xy = integrate_to_global(sweep_trajs)
        base_pos = xy[sweep_ids == int(base_codes[layer])][0, -1]
        endpoint_delta = np.linalg.norm(xy[:, -1] - base_pos[None, :], axis=1)
        score = float(endpoint_delta[sweep_ids != int(base_codes[layer])].mean()) if len(sweep_ids) > 1 else 0.0
        rows.append(
            {
                "layer": int(layer),
                "base_code": int(base_codes[layer]),
                "selected_codes": [int(x) for x in sweep_ids.tolist()],
                "importance_score": score,
            }
        )
    return sorted(rows, key=lambda row: row["importance_score"], reverse=True)


def _plot_diagnostic_series(ax, t, values, sweep_ids, base_code, color_map, ylabel):
    for code_id, y in zip(sweep_ids, values):
        code_id = int(code_id)
        label = f"base={code_id}" if code_id == int(base_code) else f"{_neighbor_rank(code_id, sweep_ids, base_code)}:{code_id}"
        ax.plot(
            t,
            y,
            color=color_map[code_id],
            linewidth=2.6 if code_id == int(base_code) else 1.3,
            alpha=1.0 if code_id == int(base_code) else 0.82,
            marker="o",
            markersize=3,
            markevery=max(1, len(t) // 8),
            label=label,
        )
    ax.set_title(ylabel)
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def plot_layer_diagnostic_figure(
    scenario_name: str,
    base_idx: int,
    layer: int,
    base_code: int,
    sweep_ids: np.ndarray,
    sweep_trajs: np.ndarray,
    dt: float,
    output_dir: str,
) -> str:
    diag = compute_series_diagnostics(sweep_trajs, dt)
    color_map = _build_neighbor_color_map(sweep_ids, base_code)
    t = np.arange(sweep_trajs.shape[1]) * dt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.reshape(-1)

    for code_id, xy in zip(sweep_ids, diag["xy"]):
        code_id = int(code_id)
        label = f"base={code_id}" if code_id == int(base_code) else f"{_neighbor_rank(code_id, sweep_ids, base_code)}:{code_id}"
        axes[0].plot(
            xy[:, 0],
            xy[:, 1],
            color=color_map[code_id],
            linewidth=3.0 if code_id == int(base_code) else 1.5,
            alpha=1.0 if code_id == int(base_code) else 0.82,
            marker="o",
            markersize=3,
            markevery=max(1, len(t) // 8),
            label=label,
        )
    base_xy = diag["xy"][sweep_ids == int(base_code)][0]
    axes[0].scatter(base_xy[0, 0], base_xy[0, 1], s=45, color="green", edgecolor="black", linewidth=0.4, zorder=5)
    axes[0].scatter(base_xy[-1, 0], base_xy[-1, 1], s=45, color="red", edgecolor="black", linewidth=0.4, zorder=5)
    axes[0].set_title("XY")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].grid(True, alpha=0.25)
    _smart_set_axis_limits(axes[0], diag["xy"], min_span=1.0)
    axes[0].legend(fontsize=7, loc="best", framealpha=0.8)

    series = [("Vx", "vx"), ("Vy", "vy"), ("V", "v"), ("Curvature", "kappa"), ("Ax", "ax"), ("Ay", "ay"), ("A", "a")]
    for ax, (title, key) in zip(axes[1:], series):
        _plot_diagnostic_series(ax, t, diag[key], sweep_ids, base_code, color_map, title)

    fig.suptitle(f"[{scenario_name}] Layer {layer:02d} local sweep (base code={base_code}, base_idx={base_idx})", fontsize=14)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{scenario_name}_layer_{layer:02d}_diagnostic.png")
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scenario_selected_layers(
    model: TrajRVQTransformer,
    scenario_name: str,
    base_idx: int,
    base_codes: np.ndarray,
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    output_dir: str,
    dt: float,
    top_n_layers: int,
    n_each_side: int,
    batch_size: int,
) -> Dict:
    ranking = rank_layers_for_scenario(model, base_codes, codes, norm_params, batch_size, n_each_side)
    paths = []
    for row in ranking[:top_n_layers]:
        layer = int(row["layer"])
        sweep_ids, sweep_trajs = _decode_layer_sweep(model, base_codes, codes, layer, norm_params, batch_size, n_each_side)
        paths.append(
            plot_layer_diagnostic_figure(
                scenario_name=scenario_name,
                base_idx=base_idx,
                layer=layer,
                base_code=int(base_codes[layer]),
                sweep_ids=sweep_ids,
                sweep_trajs=sweep_trajs,
                dt=dt,
                output_dir=output_dir,
            )
        )

    ranking_path = os.path.join(output_dir, f"{scenario_name}_layer_ranking.json")
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)
    return {"ranking_json": ranking_path, "diagnostic_plots": paths}


def compute_relative_displacement_series(xy: np.ndarray) -> Dict[str, np.ndarray]:
    """
    xy: [K, T, 2], 默认第0条为base轨迹
    return:
      delta_lateral: [K, T]
      delta_longitudinal: [K, T]
    """
    base_xy = xy[0]
    direction = base_xy[-1] - base_xy[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = base_xy[min(1, len(base_xy) - 1)] - base_xy[0]
        norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
        norm = 1.0
    forward = direction / norm
    lateral = np.array([-forward[1], forward[0]], dtype=np.float32)

    rel = xy - base_xy[None, :, :]
    delta_longitudinal = np.einsum("ktc,c->kt", rel, forward)
    delta_lateral = np.einsum("ktc,c->kt", rel, lateral)
    return {"delta_lateral": delta_lateral, "delta_longitudinal": delta_longitudinal}


def _rankdata_simple(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    xr = _rankdata_simple(x)
    yr = _rankdata_simple(y)
    x_center = xr - xr.mean()
    y_center = yr - yr.mean()
    denom = np.sqrt((x_center * x_center).sum() * (y_center * y_center).sum())
    if denom < 1e-12:
        return 0.0
    return float(abs((x_center * y_center).sum() / denom))


def compute_control_scores_for_quantity(series: np.ndarray, code_ranks: np.ndarray) -> Dict[str, float]:
    """
    series: [K, T]
    code_ranks: [K]
    """
    eps = 1e-8
    std_t = np.std(series, axis=0)
    spread_score = float(std_t.mean() / (np.std(series.reshape(-1)) + eps))

    ordering_vals = []
    for t in range(series.shape[1]):
        if std_t[t] < 1e-8:
            ordering_vals.append(0.0)
        else:
            ordering_vals.append(_spearman_abs(code_ranks, series[:, t]))
    ordering_score = float(np.mean(ordering_vals))

    max_std = float(np.max(std_t))
    if max_std < 1e-8:
        persistence_score = 0.0
    else:
        threshold = 0.1 * max_std
        persistence_score = float(np.mean(std_t > threshold))

    control_score = float(spread_score * ordering_score * persistence_score)
    return {
        "spread_score": spread_score,
        "ordering_score": ordering_score,
        "persistence_score": persistence_score,
        "control_score": control_score,
    }


def analyze_layer_control_meaning(
    model: TrajRVQTransformer,
    scenario_name: str,
    layer: int,
    base_codes: np.ndarray,
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    dt: float,
    batch_size: int,
    n_each_side: int,
) -> Dict:
    sweep_ids, sweep_trajs = _decode_layer_sweep(
        model=model,
        base_codes=base_codes,
        codes=codes,
        layer=layer,
        norm_params=norm_params,
        batch_size=batch_size,
        n_each_side=n_each_side,
    )
    base_code = int(base_codes[layer])
    base_pos = int(np.where(sweep_ids == base_code)[0][0])

    # 把 base 放在第0条，方便相对位移定义
    if base_pos != 0:
        order = [base_pos] + [i for i in range(len(sweep_ids)) if i != base_pos]
        sweep_ids = sweep_ids[order]
        sweep_trajs = sweep_trajs[order]

    diag = compute_series_diagnostics(sweep_trajs, dt)
    rel = compute_relative_displacement_series(diag["xy"])

    quantities = {
        "delta_lateral": rel["delta_lateral"],
        "delta_longitudinal": rel["delta_longitudinal"],
        "vx": diag["vx"],
        "vy": diag["vy"],
        "v": diag["v"],
        "curvature": diag["kappa"],
        "ax": diag["ax"],
        "ay": diag["ay"],
        "a": diag["a"],
    }

    # 使用数值顺序构造有序rank，base为0，左负右正
    sorted_ids = np.sort(sweep_ids)
    base_sorted_pos = int(np.where(sorted_ids == base_code)[0][0])
    rank_map = {int(code_id): idx - base_sorted_pos for idx, code_id in enumerate(sorted_ids)}
    code_ranks = np.array([rank_map[int(code_id)] for code_id in sweep_ids], dtype=np.float64)

    rows = []
    for quantity_name, series in quantities.items():
        scores = compute_control_scores_for_quantity(series, code_ranks)
        rows.append(
            {
                "scenario": scenario_name,
                "layer": int(layer),
                "base_code": base_code,
                "quantity_name": quantity_name,
                **scores,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["control_score"], reverse=True)
    top = rows_sorted[:3]
    return {
        "scores": rows,
        "top": top,
        "meta": {
            "scenario": scenario_name,
            "layer": int(layer),
            "base_code": base_code,
            "selected_codes": [int(x) for x in sweep_ids.tolist()],
            "code_ranks": code_ranks.tolist(),
        },
    }


def analyze_scenario_token_controls(
    model: TrajRVQTransformer,
    scenario_name: str,
    base_idx: int,
    base_codes: np.ndarray,
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    dt: float,
    batch_size: int,
    n_each_side: int,
    layers: List[int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    score_rows = []
    top_rows = []
    meta_rows = []
    for layer in layers:
        out = analyze_layer_control_meaning(
            model=model,
            scenario_name=scenario_name,
            layer=layer,
            base_codes=base_codes,
            codes=codes,
            norm_params=norm_params,
            dt=dt,
            batch_size=batch_size,
            n_each_side=n_each_side,
        )
        score_rows.extend(out["scores"])
        top = out["top"]
        top_rows.append(
            {
                "scenario": scenario_name,
                "layer": int(layer),
                "base_code": int(base_codes[layer]),
                "top1_quantity": top[0]["quantity_name"] if len(top) > 0 else "",
                "top1_score": float(top[0]["control_score"]) if len(top) > 0 else 0.0,
                "top2_quantity": top[1]["quantity_name"] if len(top) > 1 else "",
                "top2_score": float(top[1]["control_score"]) if len(top) > 1 else 0.0,
                "top3_quantity": top[2]["quantity_name"] if len(top) > 2 else "",
                "top3_score": float(top[2]["control_score"]) if len(top) > 2 else 0.0,
                "base_idx": int(base_idx),
            }
        )
        meta_rows.append(out["meta"])
    return score_rows, top_rows, meta_rows


def aggregate_global_token_control_meaning(score_rows: List[Dict], top_rows: List[Dict]) -> List[Dict]:
    if not score_rows:
        return []
    layers = sorted(set(int(r["layer"]) for r in score_rows))
    out = []
    for layer in layers:
        layer_scores = [r for r in score_rows if int(r["layer"]) == layer]
        quantities = sorted(set(r["quantity_name"] for r in layer_scores))
        mean_scores = {
            q: float(np.mean([r["control_score"] for r in layer_scores if r["quantity_name"] == q]))
            for q in quantities
        }
        dominant = max(mean_scores, key=mean_scores.get)
        wins = {}
        for tr in top_rows:
            if int(tr["layer"]) == layer:
                q = tr["top1_quantity"]
                wins[q] = wins.get(q, 0) + 1
        out.append(
            {
                "layer": layer,
                "dominant_quantity": dominant,
                "mean_score": float(mean_scores[dominant]),
                "number_of_scenarios": int(len([tr for tr in top_rows if int(tr["layer"]) == layer])),
                "per_scenario_wins": json.dumps(wins, ensure_ascii=True),
            }
        )
    return out


def _plot_token_control_heatmap(score_rows: List[Dict], out_path: str):
    if not score_rows:
        return
    quantities = sorted(set(r["quantity_name"] for r in score_rows))
    layers = sorted(set(int(r["layer"]) for r in score_rows))
    mat = np.zeros((len(layers), len(quantities)), dtype=np.float32)
    for i, layer in enumerate(layers):
        for j, q in enumerate(quantities):
            vals = [r["control_score"] for r in score_rows if int(r["layer"]) == layer and r["quantity_name"] == q]
            mat[i, j] = float(np.mean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(max(10, len(quantities) * 0.9), 6))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(quantities)))
    ax.set_xticklabels(quantities, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([f"L{l:02d}" for l in layers])
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Layer")
    ax.set_title("Layer x Quantity Mean Control Score")
    fig.colorbar(im, ax=ax, label="control_score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_layer_top1_map(top_rows: List[Dict], out_path: str):
    if not top_rows:
        return
    scenarios = sorted(set(r["scenario"] for r in top_rows))
    layers = sorted(set(int(r["layer"]) for r in top_rows))
    quantities = sorted(set(r["top1_quantity"] for r in top_rows))
    q_to_id = {q: i for i, q in enumerate(quantities)}

    mat = np.zeros((len(scenarios), len(layers)), dtype=np.int64)
    ann = [["" for _ in layers] for _ in scenarios]
    for i, s in enumerate(scenarios):
        for j, l in enumerate(layers):
            row = next((r for r in top_rows if r["scenario"] == s and int(r["layer"]) == l), None)
            if row is None:
                mat[i, j] = -1
                ann[i][j] = "-"
            else:
                q = row["top1_quantity"]
                mat[i, j] = q_to_id[q]
                ann[i][j] = q

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.8), max(5, len(scenarios) * 0.6)))
    im = ax.imshow(mat, aspect="auto", cmap="tab20")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f"L{l:02d}" for l in layers])
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Scenario")
    ax.set_title("Scenario x Layer Top1 Controlled Quantity")
    for i in range(len(scenarios)):
        for j in range(len(layers)):
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=6, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_layer_meaning(layer_rows: List[Dict], out_path: str):
    layers = [row["layer"] for row in layer_rows]
    feature_names = [key.replace("assoc_", "") for key in layer_rows[0] if key.startswith("assoc_")]
    score_mat = np.array([[row[f"assoc_{name}"] for name in feature_names] for row in layer_rows])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(score_mat.T, aspect="auto", cmap="viridis")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers])
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("RVQ layer / token position")
    ax.set_title("Token Position vs Motion Feature Association")
    fig.colorbar(im, ax=ax, label="between-code variance ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze RVQ token usage and rough motion meaning per token position.")
    parser.add_argument("--data-path", type=str, default="/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas.npy")
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_kin_0311")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--topk-codes", type=int, default=20)
    parser.add_argument("--plot-scenario-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-plot-scenario-diagnostics", dest="plot_scenario_diagnostics", action="store_false")
    parser.add_argument("--diagnostic-top-n-layers", type=int, default=3)
    parser.add_argument("--diagnostic-neighbor-each-side", type=int, default=3)
    parser.add_argument("--run-token-control-analysis", action="store_true", default=True)
    parser.add_argument("--no-run-token-control-analysis", dest="run_token_control_analysis", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
    norm_path = os.path.join(args.save_dir, f"{args.data_type}_norm_params.pkl")
    output_dir = args.output_dir or os.path.join(args.save_dir, "token_meaning")
    os.makedirs(output_dir, exist_ok=True)

    trajs = load_trajs(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    if args.max_samples > 0:
        trajs = trajs[:args.max_samples]

    norm_params = load_norm_params(norm_path, device)
    model = build_model(model_path, input_steps=trajs.shape[1], device=device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])

    print(f"Encoding {len(trajs)} trajectories...")
    codes = encode_codes(model, trajs, norm_params, args.batch_size)
    features = compute_features(trajs, dt=args.dt)

    layer_rows, top_code_rows = summarize_token_usage(
        codes=codes,
        features=features,
        vocab_size=model.vocab_size,
        min_count=args.min_count,
        topk=args.topk_codes,
    )

    base_idx = int(np.argmin(np.abs(features["avg_speed_mps"] - np.median(features["avg_speed_mps"]))))
    intervention_rows = interventional_summary(
        model=model,
        base_codes=codes[base_idx],
        codes=codes,
        norm_params=norm_params,
        dt=args.dt,
        topk=args.topk_codes,
        batch_size=args.batch_size,
    )

    scenario_plot_paths = {}
    control_outputs = {}
    scenario_base_indices = {}
    categories = {}
    scenario_features = {}
    if args.plot_scenario_diagnostics or args.run_token_control_analysis:
        categories, scenario_features = build_scenario_masks(trajs, dt=args.dt)
        scenario_base_indices = select_scenario_base_indices(categories, scenario_features)

    if args.plot_scenario_diagnostics:
        scenario_plot_dir = os.path.join(output_dir, "scenario_layer_diagnostics")
        for scenario_name, scenario_base_idx in scenario_base_indices.items():
            scenario_plot_paths[scenario_name] = plot_scenario_selected_layers(
                model=model,
                scenario_name=scenario_name,
                base_idx=scenario_base_idx,
                base_codes=codes[scenario_base_idx],
                codes=codes,
                norm_params=norm_params,
                output_dir=scenario_plot_dir,
                dt=args.dt,
                top_n_layers=args.diagnostic_top_n_layers,
                n_each_side=args.diagnostic_neighbor_each_side,
                batch_size=args.batch_size,
            )

    if args.run_token_control_analysis:
        score_rows = []
        top_rows = []
        meta_rows = []
        layers_all = list(range(model.num_layers))
        for scenario_name, scenario_base_idx in scenario_base_indices.items():
            s_rows, t_rows, m_rows = analyze_scenario_token_controls(
                model=model,
                scenario_name=scenario_name,
                base_idx=scenario_base_idx,
                base_codes=codes[scenario_base_idx],
                codes=codes,
                norm_params=norm_params,
                dt=args.dt,
                batch_size=args.batch_size,
                n_each_side=args.diagnostic_neighbor_each_side,
                layers=layers_all,
            )
            score_rows.extend(s_rows)
            top_rows.extend(t_rows)
            meta_rows.extend(m_rows)

        global_rows = aggregate_global_token_control_meaning(score_rows, top_rows)
        scores_csv = os.path.join(output_dir, "scenario_token_control_scores.csv")
        top_csv = os.path.join(output_dir, "scenario_token_control_top1.csv")
        global_csv = os.path.join(output_dir, "token_control_global_summary.csv")
        heatmap_png = os.path.join(output_dir, "token_control_heatmap.png")
        top1_map_png = os.path.join(output_dir, "scenario_layer_top1_map.png")
        control_json = os.path.join(output_dir, "token_control_summary.json")

        write_csv(scores_csv, score_rows)
        write_csv(top_csv, top_rows)
        write_csv(global_csv, global_rows)
        _plot_token_control_heatmap(score_rows, heatmap_png)
        _plot_scenario_layer_top1_map(top_rows, top1_map_png)
        with open(control_json, "w") as f:
            json.dump(
                {
                    "num_scenarios": len(scenario_base_indices),
                    "layers": list(range(model.num_layers)),
                    "control_scores_rows": len(score_rows),
                    "top_rows": len(top_rows),
                    "meta": meta_rows,
                    "global_summary": global_rows,
                    "files": {
                        "scenario_token_control_scores_csv": scores_csv,
                        "scenario_token_control_top1_csv": top_csv,
                        "token_control_global_summary_csv": global_csv,
                        "token_control_heatmap_png": heatmap_png,
                        "scenario_layer_top1_map_png": top1_map_png,
                    },
                },
                f,
                indent=2,
            )

        control_outputs = {
            "scenario_token_control_scores_csv": scores_csv,
            "scenario_token_control_top1_csv": top_csv,
            "token_control_global_summary_csv": global_csv,
            "token_control_summary_json": control_json,
            "token_control_heatmap_png": heatmap_png,
            "scenario_layer_top1_map_png": top1_map_png,
        }

    write_csv(os.path.join(output_dir, "token_layer_summary.csv"), layer_rows)
    write_csv(os.path.join(output_dir, "top_code_feature_means.csv"), top_code_rows)
    write_csv(os.path.join(output_dir, "token_intervention_summary.csv"), intervention_rows)
    plot_layer_meaning(layer_rows, os.path.join(output_dir, "token_feature_association.png"))

    summary = {
        "num_samples": int(len(trajs)),
        "codes_shape": list(codes.shape),
        "all_15_token_positions_present": bool(codes.shape[1] == model.num_layers == 15),
        "base_idx_for_intervention": base_idx,
        "outputs": {
            "layer_summary_csv": os.path.join(output_dir, "token_layer_summary.csv"),
            "top_code_feature_means_csv": os.path.join(output_dir, "top_code_feature_means.csv"),
            "intervention_summary_csv": os.path.join(output_dir, "token_intervention_summary.csv"),
            "association_plot": os.path.join(output_dir, "token_feature_association.png"),
            "scenario_layer_diagnostics": scenario_plot_paths,
            "token_control_analysis": control_outputs,
        },
        "layer_meaning_guess": [
            {
                "layer": row["layer"],
                "meaning_guess": row["meaning_guess"],
                "association_score": row["association_score"],
                "intervention_meaning_guess": intervention_rows[row["layer"]]["intervention_meaning_guess"],
            }
            for row in layer_rows
        ],
    }
    json_path = os.path.join(output_dir, "token_meaning_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nToken meaning analysis done.")
    print(f"codes shape: {codes.shape}  (each trajectory has {codes.shape[1]} RVQ token positions)")
    print(f"all 15 token positions present: {summary['all_15_token_positions_present']}")
    for row in summary["layer_meaning_guess"]:
        print(
            f"layer {row['layer']:02d}: data={row['meaning_guess']} "
            f"(score={row['association_score']:.3f}), intervention={row['intervention_meaning_guess']}"
        )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
