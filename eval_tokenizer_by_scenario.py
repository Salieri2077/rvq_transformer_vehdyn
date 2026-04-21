import argparse
import csv
import json
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def compute_motion_features(all_dxdydyaw_clips: np.ndarray, fps: float = 5.0, time_duration=None):
    clips = np.asarray(all_dxdydyaw_clips)
    n, t, _ = clips.shape
    dxs = clips[:, :, 0]
    dys = clips[:, :, 1]
    dyaws = clips[:, :, 2]

    cumulative_yaws = np.cumsum(dyaws, axis=1)
    prev_yaws = np.zeros((n, t), dtype=dxs.dtype)
    prev_yaws[:, 1:] = cumulative_yaws[:, :-1]

    net_yaw = cumulative_yaws[:, -1] if t > 0 else np.zeros(n, dtype=np.float32)
    gross_yaw = np.sum(np.abs(dyaws), axis=1)

    cos_y = np.cos(prev_yaws)
    sin_y = np.sin(prev_yaws)
    dx_g = cos_y * dxs - sin_y * dys
    dy_g = sin_y * dxs + cos_y * dys

    step_d = np.sqrt(dx_g**2 + dy_g**2)
    total_dist = np.sum(step_d, axis=1)

    duration = time_duration if time_duration is not None else (t / fps)
    duration = max(float(duration), 1e-6)
    avg_speed = total_dist / duration

    s = np.sign(dyaws)
    s[s == 0] = 1
    sign_changes = np.sum(s[:, 1:] != s[:, :-1], axis=1)

    return {
        "net_yaw": net_yaw,
        "gross_yaw": gross_yaw,
        "total_dist": total_dist,
        "avg_speed": avg_speed,
        "sign_changes": sign_changes,
    }


def build_scenario_masks(all_trajs: np.ndarray, fps: float = 5.0) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    feat = compute_motion_features(all_trajs, fps=fps)

    net_yaw = feat["net_yaw"]
    gross_yaw = feat["gross_yaw"]
    total_dist = feat["total_dist"]
    avg_speed = feat["avg_speed"]
    sign_changes = feat["sign_changes"]

    th_dist_static = 1.0
    th_straight_net = 0.10
    th_straight_gross = 0.20
    v_10 = 10.0 / 3.6
    v_80 = 80.0 / 3.6
    v_120 = 120.0 / 3.6
    th_turn = 0.35
    th_uturn = 2.35

    mask_static = total_dist < th_dist_static
    mask_straight = (np.abs(net_yaw) < th_straight_net) & (gross_yaw < th_straight_gross) & (~mask_static)
    mask_low_straight = mask_straight & (avg_speed >= (v_10 - 2 / 3.6)) & (avg_speed <= (v_10 + 2 / 3.6))
    mask_high_straight = mask_straight & (avg_speed >= (v_80 - 10 / 3.6)) & (avg_speed <= (v_80 + 10 / 3.6))
    mask_high_straight_120 = mask_straight & (avg_speed >= (v_120 - 15 / 3.6)) & (avg_speed <= (v_120 + 15 / 3.6))
    mask_left = (net_yaw >= th_turn) & (np.abs(net_yaw) < th_uturn) & (~mask_static)
    mask_right = (net_yaw <= -th_turn) & (np.abs(net_yaw) < th_uturn) & (~mask_static)
    mask_detour = (np.abs(net_yaw) < 0.20) & (gross_yaw >= 0.80) & (sign_changes >= 2) & (~mask_static)
    mask_uturn = (np.abs(net_yaw) >= th_uturn) & (~mask_static)

    categories = {
        "Stationary": mask_static,
        "LowSpeedStraight_10kmh": mask_low_straight,
        "HighSpeedStraight_80kmh": mask_high_straight,
        "HighSpeedStraight_120kmh": mask_high_straight_120,
        "LeftTurn": mask_left,
        "RightTurn": mask_right,
        "Detour": mask_detour,
        "UTurn": mask_uturn,
    }
    return categories, feat


def select_representative_indices(categories: Dict[str, np.ndarray], feat: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    net_yaw = feat["net_yaw"]
    gross_yaw = feat["gross_yaw"]
    total_dist = feat["total_dist"]
    avg_speed = feat["avg_speed"]
    sign_changes = feat["sign_changes"]

    v_10 = 10.0 / 3.6
    v_80 = 80.0 / 3.6
    v_120 = 120.0 / 3.6

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
        elif name == "UTurn":
            score = np.abs(np.abs(net_yaw[idxs]) - np.pi)
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
    speed, acc = compute_speed_and_acc(trajs, dt)
    speed_std = np.std(speed, axis=1)
    if acc.shape[1] > 0:
        acc_std = np.std(acc, axis=1)
        acc_abs_mean = np.mean(np.abs(acc), axis=1)
    else:
        acc_std = np.zeros(len(trajs), dtype=np.float32)
        acc_abs_mean = np.zeros(len(trajs), dtype=np.float32)

    out = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0 or num_samples <= 0:
            out[scenario_name] = []
            continue

        classic_idx = exclude_indices.get(scenario_name, {}).get("idx")
        if classic_idx is not None:
            idxs = idxs[idxs != classic_idx]
        if len(idxs) == 0:
            out[scenario_name] = []
            continue

        scenario_speed_std = speed_std[idxs]
        scenario_acc_std = acc_std[idxs]
        scenario_acc_abs_mean = acc_abs_mean[idxs]
        score = (
            scenario_speed_std / (scenario_speed_std.max() + 1e-6)
            + scenario_acc_std / (scenario_acc_std.max() + 1e-6)
            + scenario_acc_abs_mean / (scenario_acc_abs_mean.max() + 1e-6)
        )

        order = np.argsort(-score)[:num_samples]
        out[scenario_name] = [
            {
                "idx": int(idxs[i]),
                "info": {
                    "speed_std_mps": float(speed_std[idxs[i]]),
                    "acc_std_mps2": float(acc_std[idxs[i]]),
                    "acc_abs_mean_mps2": float(acc_abs_mean[idxs[i]]),
                },
            }
            for i in order
        ]
    return out


def integrate_to_global(trajs: np.ndarray) -> np.ndarray:
    dx = trajs[:, :, 0]
    dy = trajs[:, :, 1]
    dyaw = trajs[:, :, 2]
    yaw = np.cumsum(dyaw, axis=1)
    prev_yaw = np.zeros_like(yaw)
    prev_yaw[:, 1:] = yaw[:, :-1]

    dx_global = dx * np.cos(prev_yaw) - dy * np.sin(prev_yaw)
    dy_global = dx * np.sin(prev_yaw) + dy * np.cos(prev_yaw)

    x_global = np.cumsum(dx_global, axis=1)
    y_global = np.cumsum(dy_global, axis=1)
    return np.stack([x_global, y_global], axis=-1)


def compute_speed_and_acc(trajs: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(trajs[:, :, 0] ** 2 + trajs[:, :, 1] ** 2) / dt
    acc = np.diff(speed, axis=1) / dt
    return speed, acc


def reconstruct_trajs(
    model: TrajRVQTransformer,
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
    gt_xy = integrate_to_global(gt_trajs)
    pred_xy = integrate_to_global(pred_trajs)
    step_dist_error = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))

    gt_speed, pred_acc_gt = compute_speed_and_acc(gt_trajs, dt)
    pred_speed, pred_acc = compute_speed_and_acc(pred_trajs, dt)
    speed_err = pred_speed - gt_speed
    acc_err = pred_acc - pred_acc_gt

    return {
        "count": int(len(gt_trajs)),
        "ade_m": float(step_dist_error.mean()),
        "fde_m": float(step_dist_error[:, -1].mean()),
        "max_traj_error_m": float(step_dist_error.max(axis=1).mean()),
        "vrr_1m": float((step_dist_error.max(axis=1) < 1.0).mean()),
        "speed_mae_mps": float(np.abs(speed_err).mean()),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_err ** 2))),
        "gt_speed_mean_mps": float(gt_speed.mean()),
        "pred_speed_mean_mps": float(pred_speed.mean()),
        "acc_mae_mps2": float(np.abs(acc_err).mean()) if acc_err.size > 0 else 0.0,
        "acc_rmse_mps2": float(np.sqrt(np.mean(acc_err ** 2))) if acc_err.size > 0 else 0.0,
        "gt_acc_mean_mps2": float(pred_acc_gt.mean()) if pred_acc_gt.size > 0 else 0.0,
        "pred_acc_mean_mps2": float(pred_acc.mean()) if pred_acc.size > 0 else 0.0,
    }


def plot_representative_case(
    scenario_name: str,
    sample_idx: int,
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    dt: float,
    save_path: str,
):
    gt_xy = integrate_to_global(gt_traj[None, ...])[0]
    pred_xy = integrate_to_global(pred_traj[None, ...])[0]
    gt_speed, gt_acc = compute_speed_and_acc(gt_traj[None, ...], dt)
    pred_speed, pred_acc = compute_speed_and_acc(pred_traj[None, ...], dt)

    t = np.arange(gt_traj.shape[0]) * dt
    t_acc = np.arange(max(gt_traj.shape[0] - 1, 0)) * dt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(gt_xy[:, 0], gt_xy[:, 1], label="GT", linewidth=2)
    axes[0].plot(pred_xy[:, 0], pred_xy[:, 1], label="Recon", linewidth=2, linestyle="--")
    axes[0].scatter(gt_xy[0, 0], gt_xy[0, 1], c="green", s=40, label="Start")
    axes[0].scatter(gt_xy[-1, 0], gt_xy[-1, 1], c="red", s=40, label="GT End")
    axes[0].scatter(pred_xy[-1, 0], pred_xy[-1, 1], c="orange", s=40, label="Recon End")
    axes[0].set_title(f"{scenario_name} Trajectory")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, gt_speed[0], label="GT vel", linewidth=2)
    axes[1].plot(t, pred_speed[0], label="Recon vel", linewidth=2, linestyle="--")
    axes[1].set_title("Velocity")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_acc, gt_acc[0], label="GT acc", linewidth=2)
    axes[2].plot(t_acc, pred_acc[0], label="Recon acc", linewidth=2, linestyle="--")
    axes[2].set_title("Acceleration")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (m/s^2)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(f"{scenario_name} | sample_idx={sample_idx}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--data-path", type=str, default="/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas.npy")
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_kin_0311")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-var-plots", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
    norm_path = os.path.join(args.save_dir, f"{args.data_type}_norm_params.pkl")
    output_dir = args.output_dir or os.path.join(args.save_dir, "scenario_eval")
    os.makedirs(output_dir, exist_ok=True)

    trajs = load_trajs(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]

    model = build_model(model_path, input_steps=trajs.shape[1], device=device)
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

    metrics_by_scenario = {}
    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            metrics_by_scenario[scenario_name] = {"count": 0}
            continue

        metrics = scenario_metrics(trajs[idxs], recon_trajs[idxs], dt=args.dt)
        metrics.update(
            {
                "rep_idx": int(representatives[scenario_name]["idx"]),
                "rep_avg_speed_kmh": float(representatives[scenario_name]["info"]["avg_speed_kmh"]),
                "rep_total_dist_m": float(representatives[scenario_name]["info"]["total_dist"]),
                "rep_net_yaw_rad": float(representatives[scenario_name]["info"]["net_yaw"]),
            }
        )
        metrics_by_scenario[scenario_name] = metrics

        rep_idx = representatives[scenario_name]["idx"]
        plot_representative_case(
            scenario_name=scenario_name,
            sample_idx=rep_idx,
            gt_traj=trajs[rep_idx],
            pred_traj=recon_trajs[rep_idx],
            dt=args.dt,
            save_path=os.path.join(output_dir, f"{scenario_name}_classic.png"),
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
            )

    overall_metrics = scenario_metrics(trajs, recon_trajs, dt=args.dt)
    summary = {
        "config": {
            "data_path": args.data_path,
            "save_dir": args.save_dir,
            "data_type": args.data_type,
            "batch_size": args.batch_size,
            "fps": args.fps,
            "dt": args.dt,
            "output_dir": output_dir,
            "num_var_plots": args.num_var_plots,
        },
        "overall": overall_metrics,
        "scenarios": metrics_by_scenario,
        "representatives": representatives,
        "variation_representatives": variation_representatives,
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
#   --data-path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas.npy \
#   --save-dir ./work_dirs/tokenizer/rvq_tfm_kin_0311 \
#   --data-type pred \
#   --num-var-plots 5

