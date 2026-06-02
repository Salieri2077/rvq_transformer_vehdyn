import argparse
import csv
import json
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_tfm import TrajRVQTransformer
from train_tfm_accint import AccFirstRVQTokenizer
from train_tfm_bicycle import TrajRVQBicycleTransformer
from utils import (
    build_scenario_masks,
    compute_kinematic_profiles,
    count_longitudinal_sign_changes,
    resolve_default_data_path,
)


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
    if "clip_limit" in norm_params:
        out["clip_limit"] = torch.tensor(norm_params["clip_limit"], dtype=torch.float32, device=device)
    else:
        out["clip_limit"] = None
    return out


def normalize_trajs(
    trajs: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: torch.Tensor = None,
) -> torch.Tensor:
    x = torch.tensor(trajs, dtype=torch.float32, device=mean.device)
    x_norm = (x - mean) / (std + 1e-8)
    if clip_limit is not None:
        x_norm = torch.clamp(x_norm, -clip_limit, clip_limit)
    return x_norm / scale_factor


def denormalize_trajs(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
    return x_norm * scale_factor * (std + 1e-8) + mean


def traj_signature(traj: np.ndarray, decimals: int = 6) -> bytes:
    rounded = np.round(np.asarray(traj, dtype=np.float32), decimals=decimals)
    return np.ascontiguousarray(rounded).tobytes()


def build_exclude_indices_for_worst(
    representatives: Dict[str, Dict],
    variation_representatives: Dict[str, List[Dict]],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    scenario_names = set(representatives.keys()) | set(variation_representatives.keys())
    for scenario_name in scenario_names:
        exclude_idxs: List[int] = []
        rep_idx = representatives.get(scenario_name, {}).get("idx")
        if rep_idx is not None:
            exclude_idxs.append(int(rep_idx))
        for item in variation_representatives.get(scenario_name, []):
            idx = item.get("idx")
            if idx is not None:
                exclude_idxs.append(int(idx))
        out[scenario_name] = exclude_idxs
    return out


def select_representative_indices(categories: Dict[str, np.ndarray], feat: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    net_yaw = feat["net_yaw"]
    gross_yaw = feat["gross_yaw"]
    total_dist = feat["total_dist"]
    avg_speed = feat["avg_speed"]
    sign_changes = feat["sign_changes"]
    reverse_ratio = feat.get("reverse_ratio", np.zeros_like(avg_speed))
    reverse_dist = feat.get("reverse_dist", np.zeros_like(avg_speed))
    reverse_steps = feat.get("reverse_steps", np.zeros_like(sign_changes))
    stop_ratio = feat.get("stop_ratio", np.zeros_like(avg_speed))
    long_vel_sign_changes = feat.get("long_vel_sign_changes", np.zeros_like(sign_changes))
    avg_abs_curvature = feat.get("avg_abs_curvature", np.zeros_like(avg_speed))
    max_abs_curvature = feat.get("max_abs_curvature", np.zeros_like(avg_speed))

    v_10 = 10.0 / 3.6
    v_80 = 80.0 / 3.6
    v_120 = 120.0 / 3.6
    target_uturn_speed = 6.0

    out = {}
    for name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            out[name] = {"idx": None, "info": {"count": 0}}
            continue

        if name == "Stationary":
            score = total_dist[idxs]
        elif name == "LowSpeedStraight_10kmh":
            score = np.abs(avg_speed[idxs] - v_10) + 0.5 * np.abs(net_yaw[idxs]) + 0.2 * gross_yaw[idxs]
        elif name == "HighSpeedStraight_80kmh":
            score = np.abs(avg_speed[idxs] - v_80) + 0.5 * np.abs(net_yaw[idxs]) + 0.2 * gross_yaw[idxs]
        elif name == "HighSpeedStraight_120kmh":
            score = np.abs(avg_speed[idxs] - v_120) + 0.5 * np.abs(net_yaw[idxs]) + 0.2 * gross_yaw[idxs]
        elif name == "LeftTurn":
            score = np.abs(net_yaw[idxs] - 1.0) + 0.05 * np.abs(avg_speed[idxs] - v_10)
        elif name == "RightTurn":
            score = np.abs(net_yaw[idxs] + 1.0) + 0.05 * np.abs(avg_speed[idxs] - v_10)
        elif name == "Detour":
            score = np.abs(net_yaw[idxs]) + (1.0 / (gross_yaw[idxs] + 1e-6)) + (1.0 / (sign_changes[idxs] + 1e-6))
        elif name == "Reverse":
            score = (
                -reverse_dist[idxs]
                - 2.0 * reverse_ratio[idxs]
                - 0.5 * long_vel_sign_changes[idxs]
                - 0.2 * gross_yaw[idxs]
                + 0.1 * avg_speed[idxs]
            )
        elif name == "DirectUTurn":
            score = np.abs(np.abs(net_yaw[idxs]) - np.pi) + 0.05 * np.abs(avg_speed[idxs] - target_uturn_speed)
        else:
            score = np.zeros(len(idxs))

        best_idx = int(idxs[np.argmin(score)])
        out[name] = {
            "idx": best_idx,
            "info": {
                "count": int(len(idxs)),
                "net_yaw": float(net_yaw[best_idx]),
                "gross_yaw": float(gross_yaw[best_idx]),
                "total_dist": float(total_dist[best_idx]),
                "avg_speed_mps": float(avg_speed[best_idx]),
                "avg_speed_kmh": float(avg_speed[best_idx] * 3.6),
                "sign_changes": int(sign_changes[best_idx]),
                "reverse_ratio": float(reverse_ratio[best_idx]),
                "reverse_dist": float(reverse_dist[best_idx]),
                "reverse_steps": int(reverse_steps[best_idx]),
                "stop_ratio": float(stop_ratio[best_idx]),
                "long_vel_sign_changes": int(long_vel_sign_changes[best_idx]),
                "avg_abs_curvature": float(avg_abs_curvature[best_idx]),
                "max_abs_curvature": float(max_abs_curvature[best_idx]),
            },
        }
    return out


def select_velocity_acc_variation_indices(
    trajs: np.ndarray,
    categories: Dict[str, np.ndarray],
    dt: float,
    num_samples: int,
    exclude_indices: Dict[str, Dict],
) -> Dict[str, List[Dict]]:
    profiles = compute_kinematic_profiles(trajs, dt)
    speed = profiles["speed"]
    acc = profiles["acc"]
    curvature = profiles["curvature"]
    local_vx = profiles["local_vx"]

    speed_std = np.std(speed, axis=1)
    acc_std = np.std(acc, axis=1)
    acc_abs_mean = np.mean(np.abs(acc), axis=1)
    curvature_std = np.std(curvature, axis=1)
    curvature_abs_mean = np.mean(np.abs(curvature), axis=1)

    reverse_mask = local_vx < -0.2
    reverse_ratio = reverse_mask.mean(axis=1)
    reverse_dist = np.sum(np.abs(local_vx) * dt * reverse_mask, axis=1)
    stop_ratio = (speed < 0.3).mean(axis=1)
    long_vel_sign_changes = count_longitudinal_sign_changes(local_vx, speed_th=0.2)

    out = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0 or num_samples <= 0:
            out[scenario_name] = []
            continue

        used_signatures: Set[bytes] = set()
        classic_idx = exclude_indices.get(scenario_name, {}).get("idx")
        if classic_idx is not None:
            used_signatures.add(traj_signature(trajs[int(classic_idx)]))
            idxs = idxs[idxs != classic_idx]
        if len(idxs) == 0:
            out[scenario_name] = []
            continue

        scenario_speed_std = speed_std[idxs]
        scenario_acc_std = acc_std[idxs]
        scenario_acc_abs_mean = acc_abs_mean[idxs]
        scenario_curvature_std = curvature_std[idxs]
        scenario_curvature_abs_mean = curvature_abs_mean[idxs]
        scenario_reverse_ratio = reverse_ratio[idxs]
        scenario_reverse_dist = reverse_dist[idxs]
        scenario_stop_ratio = stop_ratio[idxs]
        scenario_long_sign_changes = long_vel_sign_changes[idxs]

        if scenario_name == "Reverse":
            score = (
                scenario_reverse_dist / (scenario_reverse_dist.max() + 1e-6)
                + scenario_reverse_ratio / (scenario_reverse_ratio.max() + 1e-6)
                + scenario_long_sign_changes / (scenario_long_sign_changes.max() + 1e-6)
                + scenario_stop_ratio / (scenario_stop_ratio.max() + 1e-6)
                + scenario_curvature_abs_mean / (scenario_curvature_abs_mean.max() + 1e-6)
            )
        else:
            score = (
                scenario_speed_std / (scenario_speed_std.max() + 1e-6)
                + scenario_acc_std / (scenario_acc_std.max() + 1e-6)
                + scenario_acc_abs_mean / (scenario_acc_abs_mean.max() + 1e-6)
                + scenario_curvature_abs_mean / (scenario_curvature_abs_mean.max() + 1e-6)
            )

        order = np.argsort(-score)
        selected_items: List[Dict] = []
        for i in order:
            sample_idx = int(idxs[i])
            sign = traj_signature(trajs[sample_idx])
            if sign in used_signatures:
                continue
            used_signatures.add(sign)
            selected_items.append(
                {
                    "idx": sample_idx,
                    "info": {
                        "speed_std_mps": float(speed_std[sample_idx]),
                        "acc_std_mps2": float(acc_std[sample_idx]),
                        "acc_abs_mean_mps2": float(acc_abs_mean[sample_idx]),
                        "curvature_std_1pm": float(curvature_std[sample_idx]),
                        "curvature_abs_mean_1pm": float(curvature_abs_mean[sample_idx]),
                        "reverse_ratio": float(reverse_ratio[sample_idx]),
                        "reverse_dist_m": float(reverse_dist[sample_idx]),
                        "stop_ratio": float(stop_ratio[sample_idx]),
                        "long_vel_sign_changes": int(long_vel_sign_changes[sample_idx]),
                    },
                }
            )
            if len(selected_items) >= num_samples:
                break
        out[scenario_name] = selected_items
    return out

ModelUnion = Union[TrajRVQTransformer, AccFirstRVQTokenizer, TrajRVQBicycleTransformer]


def reconstruct_trajs(
    model: ModelUnion,
    trajs: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: torch.Tensor,
    batch_size: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    all_recons: List[np.ndarray] = []
    all_codes: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(trajs), batch_size):
            batch = trajs[start:start + batch_size]
            x_norm = normalize_trajs(batch, mean, std, scale_factor, clip_limit)
            z = model.encode(x_norm)
            _, _, codes = model.rvq(z)
            if isinstance(model, TrajRVQBicycleTransformer):
                # Bicycle decoder needs initial signed longitudinal speed.
                # During training forward this comes from GT dx[0] / dt; pass
                # the same context here so reconstruction metrics are aligned.
                v0 = torch.tensor(batch[:, 0, 0] / (dt + 1e-8), dtype=torch.float32, device=mean.device)
                x_recon_norm = model.decode_from_codes(codes, v0=v0)
            else:
                x_recon_norm = model.decode_from_codes(codes)
            x_recon = denormalize_trajs(x_recon_norm, mean, std, scale_factor)
            all_recons.append(x_recon.cpu().numpy())
            all_codes.append(codes.cpu().numpy())
    return np.concatenate(all_recons, axis=0), np.concatenate(all_codes, axis=0)


def scenario_metrics(gt_trajs: np.ndarray, pred_trajs: np.ndarray, dt: float) -> Dict[str, float]:
    gt_prof = compute_kinematic_profiles(gt_trajs, dt)
    pred_prof = compute_kinematic_profiles(pred_trajs, dt)

    step_dist_error = np.sqrt(np.sum((pred_prof["xy"] - gt_prof["xy"]) ** 2, axis=-1) + 1e-6)

    speed_err = pred_prof["speed"] - gt_prof["speed"]
    acc_err = pred_prof["acc"] - gt_prof["acc"]
    vx_err = pred_prof["vx"] - gt_prof["vx"]
    vy_err = pred_prof["vy"] - gt_prof["vy"]
    ax_err = pred_prof["ax"] - gt_prof["ax"]
    ay_err = pred_prof["ay"] - gt_prof["ay"]

    curvature_err = pred_prof["curvature"] - gt_prof["curvature"]
    valid_curvature = (gt_prof["speed"] > 0.3) & (pred_prof["speed"] > 0.3)
    if np.any(valid_curvature):
        curvature_mae = float(np.mean(np.abs(curvature_err[valid_curvature])))
        curvature_rmse = float(np.sqrt(np.mean(curvature_err[valid_curvature] ** 2)))
    else:
        curvature_mae = 0.0
        curvature_rmse = 0.0

    reverse_mask = gt_prof["local_vx"] < -0.2
    reverse_ratio = reverse_mask.mean(axis=1)
    reverse_dist = np.sum(np.abs(gt_prof["local_vx"]) * dt * reverse_mask, axis=1)

    return {
        "count": int(len(gt_trajs)),
        "ade_m": float(step_dist_error.mean()),
        "fde_m": float(step_dist_error[:, -1].mean()),
        "max_traj_error_m": float(step_dist_error.max(axis=1).mean()),
        "vrr_1m": float((step_dist_error.max(axis=1) < 1.0).mean()),
        "speed_mae_mps": float(np.abs(speed_err).mean()),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_err ** 2))),
        "gt_speed_mean_mps": float(gt_prof["speed"].mean()),
        "pred_speed_mean_mps": float(pred_prof["speed"].mean()),
        "acc_mae_mps2": float(np.abs(acc_err).mean()),
        "acc_rmse_mps2": float(np.sqrt(np.mean(acc_err ** 2))),
        "gt_acc_mean_mps2": float(gt_prof["acc"].mean()),
        "pred_acc_mean_mps2": float(pred_prof["acc"].mean()),
        "vx_mae_mps": float(np.abs(vx_err).mean()),
        "vy_mae_mps": float(np.abs(vy_err).mean()),
        "ax_mae_mps2": float(np.abs(ax_err).mean()),
        "ay_mae_mps2": float(np.abs(ay_err).mean()),
        "curvature_mae_1pm": curvature_mae,
        "curvature_rmse_1pm": curvature_rmse,
        "gt_curvature_abs_mean_1pm": float(np.mean(np.abs(gt_prof["curvature"]))),
        "pred_curvature_abs_mean_1pm": float(np.mean(np.abs(pred_prof["curvature"]))),
        "reverse_ratio_mean": float(np.mean(reverse_ratio)),
        "reverse_dist_mean_m": float(np.mean(reverse_dist)),
    }


def select_worst_reconstruction_indices(
    trajs: np.ndarray,
    recon_trajs: np.ndarray,
    categories: Dict[str, np.ndarray],
    num_samples: int,
    dt: float,
    exclude_indices: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, List[Dict]]:
    gt_prof = compute_kinematic_profiles(trajs, dt=dt)
    pred_prof = compute_kinematic_profiles(recon_trajs, dt=dt)

    step_dist_error = np.sqrt(np.sum((pred_prof["xy"] - gt_prof["xy"]) ** 2, axis=-1) + 1e-6)
    ade = step_dist_error.mean(axis=1)
    fde = step_dist_error[:, -1]
    max_error = step_dist_error.max(axis=1)
    speed_mae = np.mean(np.abs(pred_prof["speed"] - gt_prof["speed"]), axis=1)
    acc_mae = np.mean(np.abs(pred_prof["acc"] - gt_prof["acc"]), axis=1)

    curvature_err = np.abs(pred_prof["curvature"] - gt_prof["curvature"])
    curvature_valid = (gt_prof["speed"] > 0.3) & (pred_prof["speed"] > 0.3)
    curvature_valid_counts = np.maximum(curvature_valid.sum(axis=1), 1)
    curvature_mae = np.sum(curvature_err * curvature_valid, axis=1) / curvature_valid_counts

    out = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0 or num_samples <= 0:
            out[scenario_name] = []
            continue

        used_signatures: Set[bytes] = set()
        if exclude_indices is not None:
            for ex_idx in exclude_indices.get(scenario_name, []):
                used_signatures.add(traj_signature(trajs[int(ex_idx)]))

        order = np.argsort(-ade[idxs])
        selected_items: List[Dict] = []
        for i in order:
            sample_idx = int(idxs[i])
            sign = traj_signature(trajs[sample_idx])
            if sign in used_signatures:
                continue
            used_signatures.add(sign)
            selected_items.append(
                {
                    "idx": sample_idx,
                    "info": {
                        "ade_m": float(ade[sample_idx]),
                        "fde_m": float(fde[sample_idx]),
                        "max_error_m": float(max_error[sample_idx]),
                        "speed_mae_mps": float(speed_mae[sample_idx]),
                        "acc_mae_mps2": float(acc_mae[sample_idx]),
                        "curvature_mae_1pm": float(curvature_mae[sample_idx]),
                    },
                }
            )
            if len(selected_items) >= num_samples:
                break
        out[scenario_name] = selected_items
    return out


def build_unclassified_mask(categories: Dict[str, np.ndarray], total_samples: int) -> np.ndarray:
    if total_samples <= 0:
        return np.zeros((0,), dtype=bool)
    covered = np.zeros((total_samples,), dtype=bool)
    for mask in categories.values():
        covered |= np.asarray(mask, dtype=bool)
    return ~covered


def select_endpoint_sign_flip_indices(
    trajs: np.ndarray,
    recon_trajs: np.ndarray,
    categories: Dict[str, np.ndarray],
    num_samples: int,
    dt: float,
    sign_eps: float = 1e-3,
    exclude_indices: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, List[Dict]]:
    gt_prof = compute_kinematic_profiles(trajs, dt=dt)
    pred_prof = compute_kinematic_profiles(recon_trajs, dt=dt)

    gt_end_xy = gt_prof["xy"][:, -1, :]
    pred_end_xy = pred_prof["xy"][:, -1, :]
    end_err = np.sqrt(np.sum((pred_end_xy - gt_end_xy) ** 2, axis=-1) + 1e-6)
    end_x_abs_diff = np.abs(pred_end_xy[:, 0] - gt_end_xy[:, 0])
    end_y_abs_diff = np.abs(pred_end_xy[:, 1] - gt_end_xy[:, 1])
    min_flip_endpoint_gap_m = 1.0 # 最小距离设置为1m，以避免微小误差导致的符号翻转被误判为有意义的翻转

    x_flip = (
        (np.abs(gt_end_xy[:, 0]) > sign_eps)
        & (np.abs(pred_end_xy[:, 0]) > sign_eps)
        & (gt_end_xy[:, 0] * pred_end_xy[:, 0] < 0.0)
        & (end_x_abs_diff >= min_flip_endpoint_gap_m)
    )
    y_flip = (
        (np.abs(gt_end_xy[:, 1]) > sign_eps)
        & (np.abs(pred_end_xy[:, 1]) > sign_eps)
        & (gt_end_xy[:, 1] * pred_end_xy[:, 1] < 0.0)
        & (end_y_abs_diff >= min_flip_endpoint_gap_m)
    )
    sign_flip_mask = x_flip | y_flip

    out = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask & sign_flip_mask)[0]
        if len(idxs) == 0 or num_samples <= 0:
            out[scenario_name] = []
            continue

        used_signatures: Set[bytes] = set()
        if exclude_indices is not None:
            for ex_idx in exclude_indices.get(scenario_name, []):
                used_signatures.add(traj_signature(trajs[int(ex_idx)]))

        order = np.argsort(-end_err[idxs])
        selected_items: List[Dict] = []
        for i in order:
            sample_idx = int(idxs[i])
            sign = traj_signature(trajs[sample_idx])
            if sign in used_signatures:
                continue
            used_signatures.add(sign)
            selected_items.append(
                {
                    "idx": sample_idx,
                    "info": {
                        "x_flip": bool(x_flip[sample_idx]),
                        "y_flip": bool(y_flip[sample_idx]),
                        "endpoint_error_m": float(end_err[sample_idx]),
                        "gt_end_x_m": float(gt_end_xy[sample_idx, 0]),
                        "gt_end_y_m": float(gt_end_xy[sample_idx, 1]),
                        "pred_end_x_m": float(pred_end_xy[sample_idx, 0]),
                        "pred_end_y_m": float(pred_end_xy[sample_idx, 1]),
                    },
                }
            )
            if len(selected_items) >= num_samples:
                break
        out[scenario_name] = selected_items
    return out


def plot_representative_case(
    scenario_name: str,
    sample_idx: int,
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    dt: float,
    save_path: str,
    sample_tokens: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    gt_prof = compute_kinematic_profiles(gt_traj[None, ...], dt)
    pred_prof = compute_kinematic_profiles(pred_traj[None, ...], dt)

    gt_xy = gt_prof["xy"][0]
    pred_xy = pred_prof["xy"][0]
    t = np.arange(gt_traj.shape[0]) * dt

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ax_traj = axes[0, 0]
    ax_vx = axes[0, 1]
    ax_vy = axes[0, 2]
    ax_v = axes[0, 3]
    ax_curv = axes[1, 0]
    ax_ax = axes[1, 1]
    ax_ay = axes[1, 2]
    ax_a = axes[1, 3]

    ax_traj.plot(gt_xy[:, 0], gt_xy[:, 1], label="GT", linewidth=2)
    ax_traj.plot(pred_xy[:, 0], pred_xy[:, 1], label="Recon", linewidth=2, linestyle="--")
    ax_traj.scatter(gt_xy[0, 0], gt_xy[0, 1], c="green", s=40, label="Start")
    ax_traj.scatter(gt_xy[-1, 0], gt_xy[-1, 1], c="red", s=40, label="GT End")
    ax_traj.scatter(pred_xy[-1, 0], pred_xy[-1, 1], c="orange", s=40, label="Recon End")
    ax_traj.set_title(f"{scenario_name} Trajectory")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=8)

    ax_vx.plot(t, gt_prof["vx"][0], label="GT vx", linewidth=2.0)
    ax_vx.plot(t, pred_prof["vx"][0], label="Recon vx", linewidth=2.0, linestyle="--")
    ax_vx.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_vx.set_title("vx")
    ax_vx.set_xlabel("Time (s)")
    ax_vx.set_ylabel("Velocity (m/s)")
    ax_vx.grid(True, alpha=0.3)
    ax_vx.legend(fontsize=8)

    ax_vy.plot(t, gt_prof["vy"][0], label="GT vy", linewidth=2.0)
    ax_vy.plot(t, pred_prof["vy"][0], label="Recon vy", linewidth=2.0, linestyle="--")
    ax_vy.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_vy.set_title("vy")
    ax_vy.set_xlabel("Time (s)")
    ax_vy.set_ylabel("Velocity (m/s)")
    ax_vy.grid(True, alpha=0.3)
    ax_vy.legend(fontsize=8)

    ax_v.plot(t, gt_prof["speed"][0], label="GT |v|", linewidth=2.0)
    ax_v.plot(t, pred_prof["speed"][0], label="Recon |v|", linewidth=2.0, linestyle="--")
    ax_v.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_v.set_title("v")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Speed (m/s)")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(fontsize=8)

    ax_curv.plot(t, gt_prof["curvature"][0], label="GT curvature", linewidth=2.0)
    ax_curv.plot(t, pred_prof["curvature"][0], label="Recon curvature", linewidth=2.0, linestyle="--")
    ax_curv.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_curv.set_title("Signed Curvature")
    ax_curv.set_xlabel("Time (s)")
    ax_curv.set_ylabel("Curvature (1/m)")
    ax_curv.grid(True, alpha=0.3)
    ax_curv.legend(fontsize=8)

    ax_ax.plot(t, gt_prof["ax"][0], label="GT ax", linewidth=2.0)
    ax_ax.plot(t, pred_prof["ax"][0], label="Recon ax", linewidth=2.0, linestyle="--")
    ax_ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_ax.set_title("ax")
    ax_ax.set_xlabel("Time (s)")
    ax_ax.set_ylabel("Acceleration (m/s^2)")
    ax_ax.grid(True, alpha=0.3)
    ax_ax.legend(fontsize=8)

    ax_ay.plot(t, gt_prof["ay"][0], label="GT ay", linewidth=2.0)
    ax_ay.plot(t, pred_prof["ay"][0], label="Recon ay", linewidth=2.0, linestyle="--")
    ax_ay.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_ay.set_title("ay")
    ax_ay.set_xlabel("Time (s)")
    ax_ay.set_ylabel("Acceleration (m/s^2)")
    ax_ay.grid(True, alpha=0.3)
    ax_ay.legend(fontsize=8)

    ax_a.plot(t, gt_prof["acc"][0], label="GT |a|", linewidth=2.0)
    ax_a.plot(t, pred_prof["acc"][0], label="Recon |a|", linewidth=2.0, linestyle="--")
    ax_a.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_a.set_title("a")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Acceleration (m/s^2)")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=8)

    title = f"{scenario_name} | sample_idx={sample_idx}"
    if sample_tokens is not None:
        token_values = np.asarray(sample_tokens).reshape(-1)[:15]
        token_values_str = ", ".join(str(int(v)) for v in token_values)
        title = f"{title}\nTokens(15): [{token_values_str}]"
    if metrics is not None:
        title = (
            f"{title}\n"
            f"recon_mse={float(metrics.get('recon_mse', 0.0)):.6f}, "
            f"ADE={float(metrics.get('ade', 0.0)):.3f}m, "
            f"FDE={float(metrics.get('fde', 0.0)):.3f}m, "
            f"max_error={float(metrics.get('max_error', 0.0)):.3f}m"
        )

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_plot_indices_with_acc_gate(
    *,
    primary_indices: List[int],
    fallback_indices: np.ndarray,
    target_count: int,
    trajs: np.ndarray,
    recon_trajs: np.ndarray,
    dt: float,
    acc_max: float,
    acc_peak_cache: Dict[int, float],
) -> Tuple[List[int], int]:
    """先按主候选筛，再在同场景补齐；筛选条件是重建 |acc| 峰值 <= acc_max。"""
    selected: List[int] = []
    used_signatures: Set[bytes] = set()
    rejected_by_acc = 0

    def _get_peak_acc(sample_idx: int) -> float:
        if sample_idx not in acc_peak_cache:
            prof = compute_kinematic_profiles(recon_trajs[sample_idx: sample_idx + 1], dt=dt)
            acc_peak_cache[sample_idx] = float(np.max(np.abs(prof["acc"][0])))
        return float(acc_peak_cache[sample_idx])

    def _try_add(sample_idx: int) -> bool:
        nonlocal rejected_by_acc
        sign = traj_signature(trajs[sample_idx])
        if sign in used_signatures:
            return False
        if _get_peak_acc(int(sample_idx)) > float(acc_max):
            rejected_by_acc += 1
            return False
        used_signatures.add(sign)
        selected.append(int(sample_idx))
        return True

    for idx in primary_indices:
        if len(selected) >= target_count:
            break
        _try_add(int(idx))

    if len(selected) < target_count:
        for idx in fallback_indices.tolist():
            if len(selected) >= target_count:
                break
            _try_add(int(idx))

    return selected, rejected_by_acc


def infer_model_type(model_path: str, model_type: str) -> str:
    if model_type != "auto":
        return model_type
    basename = os.path.basename(model_path)
    if "bicycle" in basename:
        return "bicycle"
    if ("accint" in basename) or ("accfirst" in basename):
        return "accint"
    return "taae"


def load_bicycle_config(model_path: str, data_type: str) -> Dict[str, float]:
    """Load optional bicycle config saved by train_tfm_bicycle.py."""
    config_path = os.path.join(
        os.path.dirname(model_path),
        f"{data_type}_rvq_bicycle_config.json",
    )
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(
    model_path: str,
    input_steps: int,
    device: torch.device,
    model_type: str,
    num_transformer_layers: int = 2,
    num_layers: int = 15,
    dt: float = 0.2,
    acc_max: float = 8.0,
    yaw_rate_max: float = 1.0,
) -> ModelUnion:
    state_dict = torch.load(model_path, map_location=device)
    if int(num_layers) <= 0:
        layer_ids = []
        for key in state_dict.keys():
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "rvq" and parts[1] == "layers":
                try:
                    layer_ids.append(int(parts[2]))
                except ValueError:
                    pass
        num_layers = (max(layer_ids) + 1) if layer_ids else 15

    if model_type == "accint":
        model = AccFirstRVQTokenizer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=int(num_layers),
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=num_transformer_layers,
            dt=0.2,
        ).to(device)
    elif model_type == "bicycle":
        model = TrajRVQBicycleTransformer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=int(num_layers),
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=num_transformer_layers,
            dt=dt,
            acc_max=acc_max,
            yaw_rate_max=yaw_rate_max,
        ).to(device)
    else:
        model = TrajRVQTransformer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=int(num_layers),
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=num_transformer_layers,
        ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def save_summary_csv(metrics_by_scenario: Dict[str, Dict[str, float]], csv_path: str):
    rows = []
    for scenario_name, metrics in metrics_by_scenario.items():
        row = {"scenario": scenario_name}
        row.update(metrics)
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Scenario-wise tokenizer evaluation for kinematic RVQ transformer.")
    parser.add_argument("--data-path", type=str, default=resolve_default_data_path())
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_kin_0311")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "taae", "accint", "bicycle"])
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-var-plots", type=int, default=3)
    parser.add_argument("--num-worst-plots", type=int, default=3)
    parser.add_argument("--num-sign-flip-plots", type=int, default=0)
    parser.add_argument("--num-unclassified-plots", type=int, default=0)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--plot-acc-max", type=float, default=8.0)
    parser.add_argument("--acc-max", type=float, default=8.0)
    parser.add_argument("--yaw-rate-max", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_path is not None:
        model_path = args.model_path
    elif args.model_type == "bicycle":
        model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_bicycle_model.pth")
    elif args.model_type == "accint":
        model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_accint_model.pth")
    elif args.model_type == "taae":
        model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
    else:
        bicycle_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_bicycle_model.pth")
        accint_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_accint_model.pth")
        taae_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
        if os.path.exists(bicycle_path):
            model_path = bicycle_path
        elif os.path.exists(accint_path):
            model_path = accint_path
        else:
            model_path = taae_path

    resolved_model_type = infer_model_type(model_path, args.model_type)
    bicycle_config = load_bicycle_config(model_path, args.data_type) if resolved_model_type == "bicycle" else {}
    eval_dt = float(bicycle_config.get("dt", args.dt))
    eval_num_layers = int(bicycle_config.get("num_layers", args.num_layers))
    eval_num_transformer_layers = int(
        bicycle_config.get("num_transformer_layers", args.num_transformer_layers)
    )
    eval_acc_max = float(bicycle_config.get("acc_max", args.acc_max))
    eval_yaw_rate_max = float(bicycle_config.get("yaw_rate_max", args.yaw_rate_max))
    norm_path = os.path.join(args.save_dir, f"{args.data_type}_norm_params.pkl")
    output_dir = args.output_dir or os.path.join(args.save_dir, "scenario_eval")
    os.makedirs(output_dir, exist_ok=True)

    trajs = load_trajs(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]

    model = build_model(
        model_path,
        input_steps=trajs.shape[1],
        device=device,
        model_type=resolved_model_type,
        num_transformer_layers=eval_num_transformer_layers,
        num_layers=eval_num_layers,
        dt=eval_dt,
        acc_max=eval_acc_max,
        yaw_rate_max=eval_yaw_rate_max,
    )
    norm_params = load_norm_params(norm_path, device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])

    recon_trajs, codes = reconstruct_trajs(
        model=model,
        trajs=trajs,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=args.batch_size,
        dt=eval_dt,
    )

    categories, feat = build_scenario_masks(trajs, fps=args.fps)
    representatives = select_representative_indices(categories, feat)
    variation_representatives = select_velocity_acc_variation_indices(
        trajs=trajs,
        categories=categories,
        dt=eval_dt,
        num_samples=args.num_var_plots,
        exclude_indices=representatives,
    )
    worst_exclude_indices = build_exclude_indices_for_worst(representatives, variation_representatives)
    worst_representatives = select_worst_reconstruction_indices(
        trajs=trajs,
        recon_trajs=recon_trajs,
        categories=categories,
        num_samples=args.num_worst_plots,
        dt=eval_dt,
        exclude_indices=worst_exclude_indices,
    )
    sign_flip_representatives = select_endpoint_sign_flip_indices(
        trajs=trajs,
        recon_trajs=recon_trajs,
        categories=categories,
        num_samples=args.num_sign_flip_plots,
        dt=eval_dt,
        exclude_indices=worst_exclude_indices,
    )
    unclassified_mask = build_unclassified_mask(categories, len(trajs))
    unclassified_categories = {"Unclassified": unclassified_mask}
    unclassified_representatives = select_worst_reconstruction_indices(
        trajs=trajs,
        recon_trajs=recon_trajs,
        categories=unclassified_categories,
        num_samples=args.num_unclassified_plots,
        dt=eval_dt,
    )

    total_samples = max(int(len(trajs)), 1)
    scenario_counts = {name: int(mask.sum()) for name, mask in categories.items()}
    scenario_ratios = {name: float(count / total_samples) for name, count in scenario_counts.items()}
    unclassified_count = int(unclassified_mask.sum())
    unclassified_ratio = float(unclassified_count / total_samples)

    metrics_by_scenario = {}
    plot_saved_count = 0
    plot_rejected_by_acc_count = 0
    acc_peak_cache: Dict[int, float] = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            metrics_by_scenario[scenario_name] = {"count": 0}
            continue

        metrics = scenario_metrics(trajs[idxs], recon_trajs[idxs], dt=eval_dt)
        rep_idx = representatives[scenario_name]["idx"]
        rep_info = representatives[scenario_name]["info"]
        metrics.update(
            {
                "rep_idx": int(rep_idx) if rep_idx is not None else -1,
                "rep_avg_speed_kmh": float(rep_info.get("avg_speed_kmh", 0.0)),
                "rep_total_dist_m": float(rep_info.get("total_dist", 0.0)),
                "rep_net_yaw_rad": float(rep_info.get("net_yaw", 0.0)),
            }
        )
        metrics_by_scenario[scenario_name] = metrics

        primary_classic = [int(rep_idx)] if rep_idx is not None else []
        classic_indices, rej = select_plot_indices_with_acc_gate(
            primary_indices=primary_classic,
            fallback_indices=idxs,
            target_count=1 if rep_idx is not None else 0,
            trajs=trajs,
            recon_trajs=recon_trajs,
            dt=eval_dt,
            acc_max=args.plot_acc_max,
            acc_peak_cache=acc_peak_cache,
        )
        plot_rejected_by_acc_count += rej
        for classic_idx in classic_indices:
            plot_representative_case(
                scenario_name=scenario_name,
                sample_idx=classic_idx,
                gt_traj=trajs[classic_idx],
                pred_traj=recon_trajs[classic_idx],
                dt=eval_dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_classic.png"),
                sample_tokens=codes[classic_idx],
            )
            plot_saved_count += 1

        primary_var = [int(item["idx"]) for item in variation_representatives[scenario_name]]
        var_indices, rej = select_plot_indices_with_acc_gate(
            primary_indices=primary_var,
            fallback_indices=idxs,
            target_count=int(args.num_var_plots),
            trajs=trajs,
            recon_trajs=recon_trajs,
            dt=eval_dt,
            acc_max=args.plot_acc_max,
            acc_peak_cache=acc_peak_cache,
        )
        plot_rejected_by_acc_count += rej
        for plot_id, var_idx in enumerate(var_indices, start=1):
            plot_representative_case(
                scenario_name=f"{scenario_name} Var{plot_id}",
                sample_idx=var_idx,
                gt_traj=trajs[var_idx],
                pred_traj=recon_trajs[var_idx],
                dt=eval_dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_var{plot_id:02d}.png"),
                sample_tokens=codes[var_idx],
            )
            plot_saved_count += 1

        primary_worst = [int(item["idx"]) for item in worst_representatives[scenario_name]]
        worst_indices, rej = select_plot_indices_with_acc_gate(
            primary_indices=primary_worst,
            fallback_indices=idxs,
            target_count=int(args.num_worst_plots),
            trajs=trajs,
            recon_trajs=recon_trajs,
            dt=eval_dt,
            acc_max=args.plot_acc_max,
            acc_peak_cache=acc_peak_cache,
        )
        plot_rejected_by_acc_count += rej
        for plot_id, worst_idx in enumerate(worst_indices, start=1):
            plot_representative_case(
                scenario_name=f"{scenario_name} Worst{plot_id}",
                sample_idx=worst_idx,
                gt_traj=trajs[worst_idx],
                pred_traj=recon_trajs[worst_idx],
                dt=eval_dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_worst_{plot_id:02d}.png"),
                sample_tokens=codes[worst_idx],
            )
            plot_saved_count += 1

        primary_flip = [int(item["idx"]) for item in sign_flip_representatives[scenario_name]]
        flip_indices, rej = select_plot_indices_with_acc_gate(
            primary_indices=primary_flip,
            fallback_indices=idxs,
            target_count=int(args.num_sign_flip_plots),
            trajs=trajs,
            recon_trajs=recon_trajs,
            dt=eval_dt,
            acc_max=args.plot_acc_max,
            acc_peak_cache=acc_peak_cache,
        )
        plot_rejected_by_acc_count += rej
        for plot_id, flip_idx in enumerate(flip_indices, start=1):
            plot_representative_case(
                scenario_name=f"{scenario_name} SignFlip{plot_id}",
                sample_idx=flip_idx,
                gt_traj=trajs[flip_idx],
                pred_traj=recon_trajs[flip_idx],
                dt=eval_dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_signflip_{plot_id:02d}.png"),
                sample_tokens=codes[flip_idx],
            )
            plot_saved_count += 1

    unclassified_indices = np.where(unclassified_mask)[0]
    if len(unclassified_indices) > 0:
        unclassified_metrics = scenario_metrics(trajs[unclassified_indices], recon_trajs[unclassified_indices], dt=eval_dt)
    else:
        unclassified_metrics = {"count": 0}
    primary_unclassified = [int(item["idx"]) for item in unclassified_representatives["Unclassified"]]
    unclassified_plot_indices, rej = select_plot_indices_with_acc_gate(
        primary_indices=primary_unclassified,
        fallback_indices=unclassified_indices,
        target_count=int(args.num_unclassified_plots),
        trajs=trajs,
        recon_trajs=recon_trajs,
        dt=eval_dt,
        acc_max=args.plot_acc_max,
        acc_peak_cache=acc_peak_cache,
    )
    plot_rejected_by_acc_count += rej
    for plot_id, unclassified_idx in enumerate(unclassified_plot_indices, start=1):
        plot_representative_case(
            scenario_name=f"Unclassified Worst{plot_id}",
            sample_idx=unclassified_idx,
            gt_traj=trajs[unclassified_idx],
            pred_traj=recon_trajs[unclassified_idx],
            dt=eval_dt,
            save_path=os.path.join(output_dir, f"Unclassified_worst_{plot_id:02d}.png"),
            sample_tokens=codes[unclassified_idx],
        )
        plot_saved_count += 1

    overall_metrics = scenario_metrics(trajs, recon_trajs, dt=eval_dt)
    summary = {
        "config": {
            "data_path": args.data_path,
            "save_dir": args.save_dir,
            "model_path": model_path,
            "model_type": resolved_model_type,
            "data_type": args.data_type,
            "batch_size": args.batch_size,
            "fps": args.fps,
            "dt": eval_dt,
            "output_dir": output_dir,
            "num_var_plots": args.num_var_plots,
            "num_worst_plots": args.num_worst_plots,
            "num_sign_flip_plots": args.num_sign_flip_plots,
            "num_unclassified_plots": args.num_unclassified_plots,
            "num_layers": eval_num_layers,
            "num_transformer_layers": eval_num_transformer_layers,
            "acc_max": eval_acc_max,
            "yaw_rate_max": eval_yaw_rate_max,
            "plot_acc_max": args.plot_acc_max,
        },
        "overall": overall_metrics,
        "scenarios": metrics_by_scenario,
        "representatives": representatives,
        "variation_representatives": variation_representatives,
        "worst_representatives": worst_representatives,
        "sign_flip_representatives": sign_flip_representatives,
        "scenario_counts": scenario_counts,
        "scenario_ratios": scenario_ratios,
        "unclassified": {
            "count": unclassified_count,
            "ratio": unclassified_ratio,
            "metrics": unclassified_metrics,
            "representatives": unclassified_representatives["Unclassified"],
        },
        "token_shape": list(codes.shape),
    }

    json_path = os.path.join(output_dir, "scenario_metrics.json")
    csv_path = os.path.join(output_dir, "scenario_metrics.csv")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    save_summary_csv(metrics_by_scenario, csv_path)

    print("=" * 80)
    print("Overall Metrics")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")
    print("=" * 80)
    print("Scenario Counts")
    for scenario_name, count in scenario_counts.items():
        ratio = scenario_ratios[scenario_name]
        print(f"{scenario_name}: count={count}, ratio={ratio:.4f}")
    print(f"Unclassified: count={unclassified_count}, ratio={unclassified_ratio:.4f}")
    print(
        f"Plot filter: preselect acc_peak<= {args.plot_acc_max:.2f} m/s^2 "
        f"| saved={plot_saved_count}, rejected_candidates={plot_rejected_by_acc_count}"
    )
    print("=" * 80)
    print("Scenario Metrics")
    for scenario_name, metrics in metrics_by_scenario.items():
        print(f"[{scenario_name}]")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    print("=" * 80)
    print(f"Saved summary json to: {json_path}")
    print(f"Saved summary csv to:  {csv_path}")
    print(f"Saved scenario plots to: {output_dir}")


if __name__ == "__main__":
    main()

# python eval_tokenizer_by_scenario.py \
#   --data-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --save-dir ./work_dirs/tokenizer/rvq_tfm_kin_0311 \
#   --data-type pred \
#   --num-var-plots 3 \
#   --num-worst-plots 5 \
#   --model-type taae \
#   --num-sign-flip-plots 3 \
#   --num-unclassified-plots 0

# python eval_tokenizer_by_scenario.py \
#   --data-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --save-dir ./work_dirs/tokenizer/rvq_tfm_accint_0423 \
#   --model-path ./work_dirs/tokenizer/rvq_tfm_accint_0423/pred_rvq_accint_model.pth \
#   --model-type accint \
#   --data-type pred \
#   --dt 0.2 \
#   --fps 5.0
