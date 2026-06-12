import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Support both "python train_tfm_bicycle_adaptive.py" from this directory and
# imports from the repository root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from train_tfm_bicycle import (  # noqa: E402
    TrajRVQBicycleTransformer,
    _build_dataloader,
    _make_scheduler,
    compute_gt_bicycle_targets_from_dxdydyaw,
)
from utils import build_scenario_masks, load_sampled_datas, preprocess_and_save_norm_params  # noqa: E402


def build_balanced_gate_indices(
    data_array: np.ndarray,
    samples_per_scenario: int,
    fps: float = 5.0,
    seed: int = 20260610,
) -> Dict[str, np.ndarray]:
    """
    Build a fixed scene-balanced subset for adaptive gate evaluation.

    Each non-empty scenario contributes up to samples_per_scenario trajectories.
    The subset is sampled once before training and reused for every gate eval so
    gate metrics are comparable across stages.
    """
    categories, _ = build_scenario_masks(data_array, fps=fps)
    rng = np.random.default_rng(seed)
    gate_indices: Dict[str, np.ndarray] = {}

    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0].astype(np.int64)
        if idxs.size == 0:
            continue
        if idxs.size > samples_per_scenario:
            idxs = rng.choice(idxs, size=samples_per_scenario, replace=False)
        gate_indices[scenario_name] = np.sort(idxs)

    return gate_indices


def gate_threshold_for_horizon(horizon: int, base: float, per_step: float) -> float:
    """ADE threshold for a stage: threshold(H) = base + per_step * H."""
    return float(base + per_step * horizon)


def evaluate_gate_by_scenario(
    model: TrajRVQBicycleTransformer,
    data_normalized: np.ndarray,
    gate_indices: Dict[str, np.ndarray],
    horizon: int,
    batch_size: int,
    dt: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate current-horizon ADE on the fixed balanced scenario subset.

    ADE is computed in global xy over [:horizon]. The model receives the same
    v0 context as training: gt_v[:, 0].
    """
    model.eval()
    scenario_ade: Dict[str, float] = {}

    with torch.no_grad():
        for scenario_name, idxs in gate_indices.items():
            total_dist = 0.0
            total_count = 0

            for start in range(0, len(idxs), batch_size):
                batch_idxs = idxs[start:start + batch_size]
                x_norm = torch.as_tensor(
                    data_normalized[batch_idxs],
                    dtype=torch.float32,
                    device=device,
                )

                gt_phys = model.to_phys(x_norm)
                gt = compute_gt_bicycle_targets_from_dxdydyaw(gt_phys, dt=dt)
                out = model(x_norm, v0=gt["gt_v"][:, 0])

                step_dist = torch.sqrt(
                    (out["x_global"][:, :horizon] - gt["gt_x"][:, :horizon]).pow(2)
                    + (out["y_global"][:, :horizon] - gt["gt_y"][:, :horizon]).pow(2)
                    + 1e-6
                )
                total_dist += step_dist.sum().item()
                total_count += int(step_dist.numel())

            scenario_ade[scenario_name] = total_dist / max(total_count, 1)

    return scenario_ade


def save_curriculum_history(save_dir: str, history: List[Dict]):
    path = os.path.join(save_dir, "adaptive_curriculum_history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def train_rvq_bicycle_adaptive(
    data_array: np.ndarray,
    save_dir: str = "./work_dirs/tokenizer/rvq_tfm_bicycle_adaptive_gate_v1",
    data_type: str = "pred",
    batch_size: int = 4096,
    num_layers: int = 15,
    num_transformer_layers: int = 2,
    max_epochs: int = 800,
    dt: float = 0.2,
    acc_max: float = 8.0,
    yaw_rate_max: float = 1.0,
    start_horizon: int = 5,
    horizon_step: int = 1,
    min_stage_epochs: int = 8,
    gate_eval_interval: int = 4,
    max_stage_epochs: int = 40,
    gate_samples_per_scenario: int = 2048,
    gate_ade_base: float = 0.04,
    gate_ade_per_step: float = 0.01,
    gate_seed: int = 20260610,
    gate_ignore_scenarios: Optional[List[str]] = None,
    gate_auto_ignore_stalled: bool = False,
    gate_stall_patience_epochs: int = 50,
    gate_stall_min_delta: float = 0.01,
    fps: float = 5.0,
):
    """
    Train bicycle RVQ tokenizer with adaptive horizon gate.

    The current horizon H is increased only after every non-empty scenario in a
    fixed balanced gate subset has ADE(H) below threshold(H). Scenarios in
    gate_ignore_scenarios are still evaluated/logged, but skipped by the gate
    pass/fail decision. Optionally, a scenario can also be auto-ignored for the
    current stage when its ADE has changed very little for a long enough window.
    If a stage does not pass for max_stage_epochs, H is advanced anyway to avoid
    a permanent stall.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_array = np.asarray(data_array, dtype=np.float32)
    num_steps = int(data_array.shape[1])
    input_dim = int(data_array.shape[2])
    current_horizon = int(min(max(start_horizon, 1), num_steps))
    horizon_step = int(max(horizon_step, 1))
    gate_ignore_set = {name.strip() for name in (gate_ignore_scenarios or []) if name.strip()}
    gate_auto_ignore_set = set()
    gate_ade_history: Dict[str, List[Tuple[int, int, float]]] = {}

    # 1) Normalize and prepare training loader.
    data_normalized = preprocess_and_save_norm_params(data_array, save_dir, data_type)
    dataloader = _build_dataloader(data_normalized, batch_size=batch_size)

    # 2) Fixed balanced gate subset.
    print("Building fixed balanced gate subset...")
    gate_indices = build_balanced_gate_indices(
        data_array=data_array,
        samples_per_scenario=gate_samples_per_scenario,
        fps=fps,
        seed=gate_seed,
    )
    gate_counts = {name: int(len(idxs)) for name, idxs in gate_indices.items()}
    print(f"Gate subset counts: {gate_counts}")
    if gate_ignore_set:
        unknown = sorted(gate_ignore_set - set(gate_indices.keys()))
        print(f"Gate ignored scenarios: {sorted(gate_ignore_set)}")
        if unknown:
            print(f"Warning: ignored scenarios not found in gate subset: {unknown}")
    if len(gate_indices) == 0:
        raise RuntimeError("No non-empty scenarios found for gate evaluation.")

    # 3) Model.
    model = TrajRVQBicycleTransformer(
        input_steps=num_steps,
        input_dim=input_dim,
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=num_transformer_layers,
        dt=dt,
        acc_max=acc_max,
        yaw_rate_max=yaw_rate_max,
    ).to(device)

    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")
    with open(norm_path, "rb") as f:
        norm_params = pickle.load(f)

    mean = torch.tensor(norm_params["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(norm_params["std"], device=device, dtype=torch.float32)
    scale_factor = torch.tensor(norm_params["scale_factor"], device=device, dtype=torch.float32)
    model.set_norm_params(mean, std, scale_factor)
    print(
        f"Norm params set: mean={mean.squeeze().cpu().numpy()}, "
        f"std={std.squeeze().cpu().numpy()}, scale={scale_factor.squeeze().cpu().numpy()}"
    )

    # 4) AMP, optimizer, scheduler.
    use_amp = torch.cuda.is_available()
    if use_amp and torch.cuda.get_device_capability()[0] >= 8:
        amp_dtype = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        print("Using BF16 mixed precision training")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print("Using FP16 mixed precision training")
    else:
        amp_dtype = torch.float32
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        print("Using FP32 training")

    initial_lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = _make_scheduler(optimizer, epochs=max_epochs, initial_lr=initial_lr)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))
    print("Start Training (Adaptive-Horizon Bicycle RVQ Transformer)...")

    # Loss weights kept consistent with train_tfm_bicycle.py.
    recon_loss_weight = 10.0
    xy_loss_weight = 1.0
    yaw_loss_weight = 2.0
    v_loss_weight = 0.5
    control_loss_weight = 0.2
    vq_loss_weight = 5.0
    yaw_rate_weight = 1.0
    max_lateral = 1.0

    stage_epoch = 0
    curriculum_history: List[Dict] = []
    stopped_early = False

    for epoch in range(max_epochs):
        model.train()
        smooth_loss_weight = 1e-3 if epoch > 30 else 0.0

        total_recon = 0.0
        total_xy = 0.0
        total_yaw = 0.0
        total_v = 0.0
        total_control = 0.0
        total_smooth = 0.0
        total_vq = 0.0
        total_loss = 0.0

        total_ade = 0.0
        total_fde = 0.0
        total_vrr_count = 0
        total_samples = 0
        total_acc_mean = 0.0
        total_acc_abs = 0.0
        total_yaw_rate_abs = 0.0
        total_v_abs = 0.0
        running_v_min = float("inf")
        running_v_max = float("-inf")

        for batch in dataloader:
            x_norm = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                gt_phys = model.to_phys(x_norm)
                gt = compute_gt_bicycle_targets_from_dxdydyaw(gt_phys, dt=dt)
                out = model(x_norm, v0=gt["gt_v"][:, 0])

                h = current_horizon
                pred_recon_h = out["x_recon"][:, :h]
                target_recon_h = x_norm[:, :h]

                mse_dxdy = F.mse_loss(pred_recon_h[..., :2], target_recon_h[..., :2])
                mse_dyaw = F.mse_loss(pred_recon_h[..., 2], target_recon_h[..., 2])
                recon_loss = mse_dxdy + 14.0 * mse_dyaw

                xy_loss = (
                    F.mse_loss(out["x_global"][:, :h], gt["gt_x"][:, :h])
                    + F.mse_loss(out["y_global"][:, :h], gt["gt_y"][:, :h])
                )
                yaw_loss = F.mse_loss(out["yaw"][:, :h], gt["gt_yaw"][:, :h])
                v_loss = F.mse_loss(out["v"][:, :h], gt["gt_v"][:, :h])
                control_loss = (
                    F.mse_loss(out["acc"][:, :h], gt["gt_acc"][:, :h])
                    + yaw_rate_weight
                    * F.mse_loss(out["yaw_rate"][:, :h], gt["gt_yaw_rate"][:, :h])
                )

                if h > 1:
                    jerk = (out["acc"][:, 1:h] - out["acc"][:, : h - 1]) / dt
                    yaw_acc = (out["yaw_rate"][:, 1:h] - out["yaw_rate"][:, : h - 1]) / dt
                    smooth_loss = jerk.pow(2).mean() + yaw_acc.pow(2).mean()
                else:
                    smooth_loss = out["acc"].sum() * 0.0

                vq_loss = out["vq_loss"]
                loss = (
                    recon_loss_weight * recon_loss
                    + xy_loss_weight * xy_loss
                    + yaw_loss_weight * yaw_loss
                    + v_loss_weight * v_loss
                    + control_loss_weight * control_loss
                    + smooth_loss_weight * smooth_loss
                    + vq_loss_weight * vq_loss
                )

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_recon += recon_loss.item()
            total_xy += xy_loss.item()
            total_yaw += yaw_loss.item()
            total_v += v_loss.item()
            total_control += control_loss.item()
            total_smooth += smooth_loss.item()
            total_vq += vq_loss.item()
            total_loss += loss.item()

            with torch.no_grad():
                step_dist = torch.sqrt(
                    (out["x_global"][:, :current_horizon] - gt["gt_x"][:, :current_horizon]).pow(2)
                    + (out["y_global"][:, :current_horizon] - gt["gt_y"][:, :current_horizon]).pow(2)
                    + 1e-6
                )
                batch_size_actual = x_norm.shape[0]
                total_ade += step_dist.mean(dim=1).sum().item()
                total_fde += step_dist[:, -1].sum().item()
                total_vrr_count += (step_dist.max(dim=1)[0] < max_lateral).sum().item()
                total_samples += batch_size_actual

                total_acc_mean += out["acc"][:, :current_horizon].mean().item()
                total_acc_abs += out["acc"][:, :current_horizon].abs().mean().item()
                total_yaw_rate_abs += out["yaw_rate"][:, :current_horizon].abs().mean().item()
                total_v_abs += out["v"][:, :current_horizon].abs().mean().item()
                running_v_min = min(running_v_min, out["v"][:, :current_horizon].min().item())
                running_v_max = max(running_v_max, out["v"][:, :current_horizon].max().item())

        scheduler.step()
        stage_epoch += 1

        num_batches = len(dataloader)
        avg_recon = total_recon / num_batches
        avg_xy = total_xy / num_batches
        avg_yaw = total_yaw / num_batches
        avg_v = total_v / num_batches
        avg_control = total_control / num_batches
        avg_smooth = total_smooth / num_batches
        avg_vq = total_vq / num_batches
        avg_loss = total_loss / num_batches
        avg_ade = total_ade / total_samples if total_samples > 0 else 0.0
        avg_fde = total_fde / total_samples if total_samples > 0 else 0.0
        vrr = total_vrr_count / total_samples if total_samples > 0 else 0.0
        avg_acc_mean = total_acc_mean / num_batches
        avg_acc_abs = total_acc_abs / num_batches
        avg_yaw_rate_abs = total_yaw_rate_abs / num_batches
        avg_v_abs = total_v_abs / num_batches

        writer.add_scalar("loss/recon", avg_recon, epoch + 1)
        writer.add_scalar("loss/xy", avg_xy, epoch + 1)
        writer.add_scalar("loss/yaw", avg_yaw, epoch + 1)
        writer.add_scalar("loss/v", avg_v, epoch + 1)
        writer.add_scalar("loss/control", avg_control, epoch + 1)
        writer.add_scalar("loss/smooth", avg_smooth, epoch + 1)
        writer.add_scalar("loss/vq", avg_vq, epoch + 1)
        writer.add_scalar("loss/total", avg_loss, epoch + 1)
        writer.add_scalar("metrics/ade", avg_ade, epoch + 1)
        writer.add_scalar("metrics/fde", avg_fde, epoch + 1)
        writer.add_scalar("metrics/vrr_1m", vrr, epoch + 1)
        writer.add_scalar("stats/acc_abs_mean", avg_acc_abs, epoch + 1)
        writer.add_scalar("stats/yaw_rate_abs_mean", avg_yaw_rate_abs, epoch + 1)
        writer.add_scalar("stats/v_abs_mean", avg_v_abs, epoch + 1)
        writer.add_scalar("stats/acc_mean", avg_acc_mean, epoch + 1)
        writer.add_scalar("stats/v_min", running_v_min, epoch + 1)
        writer.add_scalar("stats/v_max", running_v_max, epoch + 1)
        writer.add_scalar("adaptive/current_horizon", current_horizon, epoch + 1)
        writer.add_scalar("adaptive/stage_epoch", stage_epoch, epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == max_epochs:
            print(
                f"[BiRVQ-Adaptive] Epoch {epoch+1:03d} | H: {current_horizon:02d} | "
                f"StageEp: {stage_epoch:02d} | Recon: {avg_recon:.5f} | "
                f"XY: {avg_xy:.5f} | Yaw: {avg_yaw:.5f} | V: {avg_v:.5f} | "
                f"Ctrl: {avg_control:.5f} | Smooth: {avg_smooth:.5f} | "
                f"VQ: {avg_vq:.5f} | TrajErr: {avg_ade:.4f} | "
                f"EndErr: {avg_fde:.4f} | VRR: {vrr:.4f} | "
                f"acc_mean: {avg_acc_mean:.3f} | acc_abs: {avg_acc_abs:.3f} | "
                f"yaw_rate_abs: {avg_yaw_rate_abs:.3f} | "
                f"v_min: {running_v_min:.2f} | v_max: {running_v_max:.2f}"
            )

        should_eval_gate = (
            stage_epoch >= min_stage_epochs
            and (stage_epoch - min_stage_epochs) % max(gate_eval_interval, 1) == 0
        )
        force_advance = (
            current_horizon < num_steps
            and stage_epoch >= max_stage_epochs
        )

        if should_eval_gate or force_advance:
            scenario_ade = evaluate_gate_by_scenario(
                model=model,
                data_normalized=data_normalized,
                gate_indices=gate_indices,
                horizon=current_horizon,
                batch_size=batch_size,
                dt=dt,
                device=device,
            )
            threshold = gate_threshold_for_horizon(
                current_horizon,
                base=gate_ade_base,
                per_step=gate_ade_per_step,
            )
            for scenario_name, ade in scenario_ade.items():
                gate_ade_history.setdefault(scenario_name, []).append(
                    (epoch + 1, current_horizon, float(ade))
                )

            if gate_auto_ignore_stalled:
                for scenario_name, ade in scenario_ade.items():
                    if scenario_name in gate_ignore_set or scenario_name in gate_auto_ignore_set:
                        continue
                    if ade <= threshold:
                        continue
                    records = [
                        item
                        for item in gate_ade_history.get(scenario_name, [])
                        if item[1] == current_horizon
                    ]
                    if not records:
                        continue
                    recent = [
                        item
                        for item in records
                        if item[0] >= (epoch + 1 - gate_stall_patience_epochs)
                    ]
                    # Gate eval is discrete, e.g. every 4 epochs. A "50 epoch"
                    # window will usually span 48 epochs in recorded eval points
                    # (308, 312, ..., 356), so allow one eval interval of slack.
                    min_span = max(gate_stall_patience_epochs - max(gate_eval_interval, 1), 0)
                    if recent[-1][0] - recent[0][0] < min_span:
                        continue
                    recent_ades = [item[2] for item in recent]
                    ade_delta = max(recent_ades) - min(recent_ades)
                    if ade_delta < gate_stall_min_delta:
                        gate_auto_ignore_set.add(scenario_name)
                        print(
                            f"[Gate] Auto-ignore stalled scenario {scenario_name}: "
                            f"H={current_horizon:02d}, {len(recent)} evals over "
                            f"{recent[-1][0] - recent[0][0]} epochs, "
                            f"ade_delta={ade_delta:.4f} < {gate_stall_min_delta:.4f}"
                        )

            active_ignore_set = gate_ignore_set | gate_auto_ignore_set
            # Some scenario types can be kept for monitoring/plots while being
            # excluded from the horizon gate. DirectUTurn is useful this way when
            # its data distribution makes the global threshold too strict.
            gate_eval_ade = {
                name: ade
                for name, ade in scenario_ade.items()
                if name not in active_ignore_set
            }
            if len(gate_eval_ade) == 0:
                raise RuntimeError("All gate scenarios are ignored; at least one scenario must remain active.")
            gate_pass = all(ade <= threshold for ade in gate_eval_ade.values())
            advance = gate_pass or force_advance
            next_horizon = current_horizon
            if advance and current_horizon < num_steps:
                next_horizon = min(num_steps, current_horizon + horizon_step)

            writer.add_scalar("adaptive/gate_pass", 1.0 if gate_pass else 0.0, epoch + 1)
            writer.add_scalar("adaptive/gate_forced_advance", 1.0 if force_advance else 0.0, epoch + 1)
            writer.add_scalar("adaptive/gate_threshold", threshold, epoch + 1)
            writer.add_scalar("adaptive/gate_auto_ignore_count", len(gate_auto_ignore_set), epoch + 1)
            for scenario_name, ade in scenario_ade.items():
                writer.add_scalar(f"adaptive/gate_ade/{scenario_name}", ade, epoch + 1)
                writer.add_scalar(
                    f"adaptive/gate_auto_ignored/{scenario_name}",
                    1.0 if scenario_name in gate_auto_ignore_set else 0.0,
                    epoch + 1,
                )

            history_item = {
                "epoch": int(epoch + 1),
                "horizon": int(current_horizon),
                "stage_epoch": int(stage_epoch),
                "threshold": float(threshold),
                "gate_pass": bool(gate_pass),
                "forced_advance": bool(force_advance),
                "next_horizon": int(next_horizon),
                "gate_ignore_scenarios": sorted(gate_ignore_set),
                "gate_auto_ignore_scenarios": sorted(gate_auto_ignore_set),
                "gate_active_scenarios": sorted(gate_eval_ade.keys()),
                "scenario_ade": {k: float(v) for k, v in scenario_ade.items()},
            }
            curriculum_history.append(history_item)
            save_curriculum_history(save_dir, curriculum_history)

            worst_name, worst_ade = max(gate_eval_ade.items(), key=lambda kv: kv[1])
            ignored_text = ""
            if active_ignore_set:
                ignored_text = (
                    f" | ignored: manual={sorted(gate_ignore_set)}, "
                    f"auto={sorted(gate_auto_ignore_set)}"
                )
            print(
                f"[Gate] Epoch {epoch+1:03d} | H: {current_horizon:02d} | "
                f"thr: {threshold:.4f} | pass: {gate_pass} | "
                f"forced: {force_advance} | worst(active): {worst_name}={worst_ade:.4f}"
                f"{ignored_text}"
            )

            if gate_pass and current_horizon >= num_steps:
                print("[Gate] Full horizon passed. Early stopping.")
                stopped_early = True
                break

            if advance and current_horizon < num_steps:
                current_horizon = next_horizon
                stage_epoch = 0
                gate_auto_ignore_set.clear()
                print(f"[Gate] Advance to H={current_horizon}")
            elif force_advance:
                stage_epoch = 0
                gate_auto_ignore_set.clear()

    # Save model and config.
    model_path = os.path.join(save_dir, f"{data_type}_rvq_bicycle_adaptive_model.pth")
    torch.save(model.state_dict(), model_path)

    config = {
        "model_type": "TrajRVQBicycleTransformer",
        "training_type": "adaptive_horizon_gate",
        "dt": dt,
        "num_layers": num_layers,
        "vocab_size": 1024,
        "d_model": 128,
        "num_transformer_layers": num_transformer_layers,
        "acc_max": acc_max,
        "yaw_rate_max": yaw_rate_max,
        "input_steps": num_steps,
        "input_dim": input_dim,
        "max_epochs": max_epochs,
        "final_horizon": current_horizon,
        "stopped_early": stopped_early,
        "start_horizon": start_horizon,
        "horizon_step": horizon_step,
        "min_stage_epochs": min_stage_epochs,
        "gate_eval_interval": gate_eval_interval,
        "max_stage_epochs": max_stage_epochs,
        "gate_samples_per_scenario": gate_samples_per_scenario,
        "gate_ade_base": gate_ade_base,
        "gate_ade_per_step": gate_ade_per_step,
        "gate_seed": gate_seed,
        "gate_ignore_scenarios": sorted(gate_ignore_set),
        "gate_auto_ignore_stalled": gate_auto_ignore_stalled,
        "gate_stall_patience_epochs": gate_stall_patience_epochs,
        "gate_stall_min_delta": gate_stall_min_delta,
        "fps": fps,
        "gate_counts": gate_counts,
    }
    config_path = os.path.join(save_dir, f"{data_type}_rvq_bicycle_adaptive_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    save_curriculum_history(save_dir, curriculum_history)

    writer.close()
    print(f"Adaptive bicycle RVQ training done. Model saved to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adaptive-horizon bicycle RVQ tokenizer.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_bicycle_adaptive_gate_v1")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--acc-max", type=float, default=8.0)
    parser.add_argument("--yaw-rate-max", type=float, default=1.0)
    parser.add_argument("--start-horizon", type=int, default=5)
    parser.add_argument("--horizon-step", type=int, default=1)
    parser.add_argument("--min-stage-epochs", type=int, default=8)
    parser.add_argument("--gate-eval-interval", type=int, default=4)
    parser.add_argument("--max-stage-epochs", type=int, default=40)
    parser.add_argument("--gate-samples-per-scenario", type=int, default=2048)
    parser.add_argument("--gate-ade-base", type=float, default=0.04)
    parser.add_argument("--gate-ade-per-step", type=float, default=0.01)
    parser.add_argument("--gate-seed", type=int, default=20260610)
    parser.add_argument(
        "--gate-ignore-scenarios",
        type=str,
        default="DirectUTurn",
        help=(
            "Comma-separated scenarios to evaluate/log but exclude from gate_pass. "
            "Use an empty string to include every scenario."
        ),
    )
    parser.add_argument(
        "--gate-auto-ignore-stalled",
        action="store_true",
        help="Auto-ignore scenarios whose gate ADE has plateaued above threshold for the current horizon.",
    )
    parser.add_argument("--gate-stall-patience-epochs", type=int, default=50)
    parser.add_argument("--gate-stall-min-delta", type=float, default=0.01)
    args = parser.parse_args()

    sampled_trajs = load_sampled_datas(args.data_path)
    sampled_trajs = np.asarray(sampled_trajs, dtype=np.float32)
    if args.data_type == "history":
        sampled_trajs = sampled_trajs[:, :14, :]
    if args.max_samples > 0:
        sampled_trajs = sampled_trajs[: args.max_samples]
    gate_ignore_scenarios = [
        item.strip()
        for item in args.gate_ignore_scenarios.split(",")
        if item.strip()
    ]

    print(
        f"Train config | data_type={args.data_type} | num_layers={args.num_layers} | "
        f"num_transformer_layers={args.num_transformer_layers} | batch_size={args.batch_size} | "
        f"max_epochs={args.max_epochs} | dt={args.dt} | acc_max={args.acc_max} | "
        f"yaw_rate_max={args.yaw_rate_max} | start_horizon={args.start_horizon} | "
        f"horizon_step={args.horizon_step} | min_stage_epochs={args.min_stage_epochs} | "
        f"gate_eval_interval={args.gate_eval_interval} | max_stage_epochs={args.max_stage_epochs} | "
        f"gate_samples_per_scenario={args.gate_samples_per_scenario} | "
        f"gate_ade_base={args.gate_ade_base} | gate_ade_per_step={args.gate_ade_per_step} | "
        f"gate_ignore_scenarios={gate_ignore_scenarios} | "
        f"gate_auto_ignore_stalled={args.gate_auto_ignore_stalled} | "
        f"gate_stall_patience_epochs={args.gate_stall_patience_epochs} | "
        f"gate_stall_min_delta={args.gate_stall_min_delta} | "
        f"save_dir={args.save_dir}"
    )
    print(f"Dataset shape: {sampled_trajs.shape}")

    train_rvq_bicycle_adaptive(
        sampled_trajs,
        save_dir=args.save_dir,
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_transformer_layers=args.num_transformer_layers,
        max_epochs=args.max_epochs,
        dt=args.dt,
        acc_max=args.acc_max,
        yaw_rate_max=args.yaw_rate_max,
        start_horizon=args.start_horizon,
        horizon_step=args.horizon_step,
        min_stage_epochs=args.min_stage_epochs,
        gate_eval_interval=args.gate_eval_interval,
        max_stage_epochs=args.max_stage_epochs,
        gate_samples_per_scenario=args.gate_samples_per_scenario,
        gate_ade_base=args.gate_ade_base,
        gate_ade_per_step=args.gate_ade_per_step,
        gate_seed=args.gate_seed,
        gate_ignore_scenarios=gate_ignore_scenarios,
        gate_auto_ignore_stalled=args.gate_auto_ignore_stalled,
        gate_stall_patience_epochs=args.gate_stall_patience_epochs,
        gate_stall_min_delta=args.gate_stall_min_delta,
        fps=args.fps,
    )
