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

ModelUnion = Union[TrajRVQTransformer, AccFirstRVQTokenizer]


def reconstruct_trajs(
    model: ModelUnion,
    trajs: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: torch.Tensor,
    batch_size: int,
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


def plot_representative_case(
    scenario_name: str,
    sample_idx: int,
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    dt: float,
    save_path: str,
    sample_tokens: Optional[np.ndarray] = None,
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

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def infer_model_type(model_path: str, model_type: str) -> str:
    if model_type != "auto":
        return model_type
    basename = os.path.basename(model_path)
    if ("accint" in basename) or ("accfirst" in basename):
        return "accint"
    return "taae"


def build_model(
    model_path: str,
    input_steps: int,
    device: torch.device,
    model_type: str,
    num_transformer_layers: int = 2,
) -> ModelUnion:
    if model_type == "accint":
        model = AccFirstRVQTokenizer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=15,
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=num_transformer_layers,
            dt=0.2,
        ).to(device)
    else:
        model = TrajRVQTransformer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=15,
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=num_transformer_layers,
        ).to(device)
    state_dict = torch.load(model_path, map_location=device)
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
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "taae", "accint"])
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-var-plots", type=int, default=3)
    parser.add_argument("--num-worst-plots", type=int, default=3)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_path is not None:
        model_path = args.model_path
    elif args.model_type == "accint":
        model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_accint_model.pth")
    elif args.model_type == "taae":
        model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
    else:
        accint_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_accint_model.pth")
        taae_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
        model_path = accint_path if os.path.exists(accint_path) else taae_path

    resolved_model_type = infer_model_type(model_path, args.model_type)
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
        num_transformer_layers=args.num_transformer_layers,
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
    )

    categories, feat = build_scenario_masks(trajs, fps=args.fps)
    representatives = select_representative_indices(categories, feat)
    variation_representatives = select_velocity_acc_variation_indices(
        trajs=trajs,
        categories=categories,
        dt=args.dt,
        num_samples=args.num_var_plots,
        exclude_indices=representatives,
    )
    worst_exclude_indices = build_exclude_indices_for_worst(representatives, variation_representatives)
    worst_representatives = select_worst_reconstruction_indices(
        trajs=trajs,
        recon_trajs=recon_trajs,
        categories=categories,
        num_samples=args.num_worst_plots,
        dt=args.dt,
        exclude_indices=worst_exclude_indices,
    )

    total_samples = max(int(len(trajs)), 1)
    scenario_counts = {name: int(mask.sum()) for name, mask in categories.items()}
    scenario_ratios = {name: float(count / total_samples) for name, count in scenario_counts.items()}

    metrics_by_scenario = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            metrics_by_scenario[scenario_name] = {"count": 0}
            continue

        metrics = scenario_metrics(trajs[idxs], recon_trajs[idxs], dt=args.dt)
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

        if rep_idx is not None:
            plot_representative_case(
                scenario_name=scenario_name,
                sample_idx=rep_idx,
                gt_traj=trajs[rep_idx],
                pred_traj=recon_trajs[rep_idx],
                dt=args.dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_classic.png"),
                sample_tokens=codes[rep_idx],
            )
        for plot_id, item in enumerate(variation_representatives[scenario_name], start=1):
            var_idx = item["idx"]
            plot_representative_case(
                scenario_name=f"{scenario_name} Var{plot_id}",
                sample_idx=var_idx,
                gt_traj=trajs[var_idx],
                pred_traj=recon_trajs[var_idx],
                dt=args.dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_var{plot_id:02d}.png"),
                sample_tokens=codes[var_idx],
            )
        for plot_id, item in enumerate(worst_representatives[scenario_name], start=1):
            worst_idx = item["idx"]
            plot_representative_case(
                scenario_name=f"{scenario_name} Worst{plot_id}",
                sample_idx=worst_idx,
                gt_traj=trajs[worst_idx],
                pred_traj=recon_trajs[worst_idx],
                dt=args.dt,
                save_path=os.path.join(output_dir, f"{scenario_name}_worst_{plot_id:02d}.png"),
                sample_tokens=codes[worst_idx],
            )

    overall_metrics = scenario_metrics(trajs, recon_trajs, dt=args.dt)
    summary = {
        "config": {
            "data_path": args.data_path,
            "save_dir": args.save_dir,
            "model_path": model_path,
            "model_type": resolved_model_type,
            "data_type": args.data_type,
            "batch_size": args.batch_size,
            "fps": args.fps,
            "dt": args.dt,
            "output_dir": output_dir,
            "num_var_plots": args.num_var_plots,
            "num_worst_plots": args.num_worst_plots,
            "num_transformer_layers": args.num_transformer_layers,
        },
        "overall": overall_metrics,
        "scenarios": metrics_by_scenario,
        "representatives": representatives,
        "variation_representatives": variation_representatives,
        "worst_representatives": worst_representatives,
        "scenario_counts": scenario_counts,
        "scenario_ratios": scenario_ratios,
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
#   --model-type taae

# python eval_tokenizer_by_scenario.py \
#   --data-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --save-dir ./work_dirs/tokenizer/rvq_tfm_accint_0423 \
#   --model-path ./work_dirs/tokenizer/rvq_tfm_accint_0423/pred_rvq_accint_model.pth \
#   --model-type accint \
#   --data-type pred \
#   --dt 0.2 \
#   --fps 5.0
