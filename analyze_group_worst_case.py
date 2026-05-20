import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RVQ_DIR = THIS_DIR
if RVQ_DIR not in sys.path:
    sys.path.insert(0, RVQ_DIR)

from eval_tokenizer_by_scenario import (  # noqa: E402
    build_model,
    infer_model_type,
    plot_representative_case,
    reconstruct_trajs,
)
from grouping_pipeline import try_load_grouping_from_cache  # noqa: E402
from utils import (  # noqa: E402
    compute_kinematic_profiles,
    compute_reconstruction_case_metrics,
    decode_token_prefix_reconstructions,
    denormalize_trajs_torch,
    load_norm_params_torch,
    plot_target_token_prefix_reconstructions,
    load_traj_array,
    normalize_trajs_torch,
    token_sequence_to_str,
    write_csv,
    write_json,
)


GROUP_CACHE_FILES = (
    "representative_indices.npy",
    "group_id_per_sample.npy",
    "group_sizes.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether a worst case is isolated or common within its similar-trajectory group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--norm_path", "--norm-path", dest="norm_path", required=True)
    parser.add_argument("--data_path", "--data-path", dest="data_path", required=True)
    parser.add_argument("--sample_idx", "--sample-idx", dest="sample_idx", type=int, required=True)
    parser.add_argument("--group_cache_dir", "--group-cache-dir", dest="group_cache_dir", required=True)
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--max_group_vis", "--max-group-vis", dest="max_group_vis", type=int, default=16)

    parser.add_argument("--group_cache_key", "--group-cache-key", dest="group_cache_key", type=str, default="")
    parser.add_argument("--source_indices_path", "--source-indices-path", dest="source_indices_path", type=str, default="")
    parser.add_argument(
        "--sample_idx_is_source",
        "--sample-idx-is-source",
        dest="sample_idx_is_source",
        action="store_true",
        help="Treat --sample_idx as the original/source row id and locate its row in the filtered data.",
    )
    parser.add_argument("--source_occurrence", "--source-occurrence", dest="source_occurrence", type=int, default=0)
    parser.add_argument("--model_type", "--model-type", dest="model_type", default="auto", choices=["auto", "taae", "accint"])
    parser.add_argument("--data_type", "--data-type", dest="data_type", default="pred", choices=["pred", "history"])
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4096)
    parser.add_argument("--num_layers", "--num-layers", dest="num_layers", type=int, default=0)
    parser.add_argument("--num_transformer_layers", "--num-transformer-layers", dest="num_transformer_layers", type=int, default=2)
    parser.add_argument(
        "--loss_epoch",
        "--loss-epoch",
        dest="loss_epoch",
        type=int,
        default=31,
        help="Only controls train-style kin_smooth weight: 1e-2 if loss_epoch > 30 else 0.",
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def has_group_cache_files(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, name)) for name in GROUP_CACHE_FILES)


def load_grouping_cache(
    group_cache_dir: str,
    group_cache_key: str,
    n_total: int,
) -> Tuple[str, Dict[str, object]]:
    """
    只读取已有 grouping cache，不重新跑 grouping。

    支持两种输入：
      1) --group_cache_dir 直接指向包含 group_id_per_sample.npy 的 cache 子目录；
      2) --group_cache_dir 指向 grouping_cache 根目录，自动查找唯一可用子目录。
    """
    root = os.path.abspath(group_cache_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"group_cache_dir does not exist or is not a directory: {root}")

    if group_cache_key.strip():
        candidates = [os.path.join(root, group_cache_key.strip())]
    else:
        candidates = [root]
        for name in sorted(os.listdir(root)):
            child = os.path.join(root, name)
            if os.path.isdir(child) and has_group_cache_files(child):
                candidates.append(child)

    loaded: List[Tuple[str, Dict[str, object]]] = []
    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        grouping = try_load_grouping_from_cache(candidate, n_total=n_total)
        if grouping is not None:
            loaded.append((candidate, grouping))

    if not loaded:
        expected = ", ".join(GROUP_CACHE_FILES)
        raise FileNotFoundError(
            f"No valid grouping cache found under {root}. "
            f"Expected files: {expected}; cache length must match data N={n_total}."
        )

    # 如果用户直接传了具体 cache 目录，优先使用它。
    if loaded[0][0] == root:
        return loaded[0]

    if len(loaded) > 1 and not group_cache_key.strip():
        dirs = "\n".join(f"  - {path}" for path, _ in loaded)
        raise ValueError(
            "Multiple valid grouping cache directories found. "
            "Pass the exact cache directory or set --group_cache_key.\n"
            f"{dirs}"
        )

    return loaded[0]


def find_sample_group(grouping: Dict[str, object], sample_idx: int) -> Tuple[int, np.ndarray]:
    """
    group_id_per_sample 是 sample_idx -> group_id 的直接映射；
    group_to_indices 再把 group_id 映射回该组所有 sample_idx。
    """
    group_id_per_sample = np.asarray(grouping["group_id_per_sample"], dtype=np.int32)
    if sample_idx < 0 or sample_idx >= group_id_per_sample.shape[0]:
        raise IndexError(
            f"sample_idx={sample_idx} is out of range for group_id_per_sample "
            f"with length {group_id_per_sample.shape[0]}."
        )

    group_id = int(group_id_per_sample[sample_idx])
    group_to_indices = grouping.get("group_to_indices")
    if group_to_indices is not None and 0 <= group_id < len(group_to_indices):
        members = np.asarray(group_to_indices[group_id], dtype=np.int64)
    else:
        members = np.where(group_id_per_sample == group_id)[0].astype(np.int64)

    if members.size == 0 or not np.any(members == sample_idx):
        raise RuntimeError(
            f"Grouping cache is inconsistent: sample_idx={sample_idx} maps to "
            f"group_id={group_id}, but that group does not contain the sample."
        )
    return group_id, members


def unique_source_group_indices(
    group_indices: np.ndarray,
    source_indices: Optional[np.ndarray],
    target_sample_idx: int,
) -> np.ndarray:
    """按 source_sample_idx 去重；目标样本对应的 row 始终排第一。"""
    group_indices = np.asarray(group_indices, dtype=np.int64)
    if source_indices is None:
        return group_indices

    ordered: List[int] = []
    seen_sources = set()

    def add_once(idx: int) -> None:
        src = int(source_indices[int(idx)])
        if src in seen_sources:
            return
        seen_sources.add(src)
        ordered.append(int(idx))

    add_once(int(target_sample_idx))
    for idx in group_indices.tolist():
        add_once(int(idx))
    return np.asarray(ordered, dtype=np.int64)


def rank_desc(values: np.ndarray, target_pos: int) -> int:
    """按 loss 从大到小排名；rank=1 表示该指标在 group 内最差。"""
    return int(np.sum(values > values[target_pos]) + 1)


def metric_at(metrics: Dict[str, np.ndarray], pos: int) -> Dict[str, float]:
    return {key: float(values[pos]) for key, values in metrics.items()}


def integrate_to_global_torch(trajs: torch.Tensor) -> torch.Tensor:
    dx = trajs[:, :, 0]
    dy = trajs[:, :, 1]
    dyaw = trajs[:, :, 2]
    yaw = torch.cumsum(dyaw, dim=1)
    prev_yaw = torch.zeros_like(yaw)
    if trajs.shape[1] > 1:
        prev_yaw[:, 1:] = yaw[:, :-1]

    dx_global = dx * torch.cos(prev_yaw) - dy * torch.sin(prev_yaw)
    dy_global = dx * torch.sin(prev_yaw) + dy * torch.cos(prev_yaw)
    return torch.stack([torch.cumsum(dx_global, dim=1), torch.cumsum(dy_global, dim=1)], dim=-1)


def per_case_vq_loss_from_latent(model, z: torch.Tensor) -> torch.Tensor:
    """
    训练时 VQ loss 是 batch mean；这里沿用 residual quantization 顺序，
    拆成 per-case commitment loss，便于定位单个 sample 的贡献。
    """
    residual = z
    losses = z.new_zeros((z.shape[0],))
    for layer in model.rvq.layers:
        flat_input = residual.view(-1, layer.embedding_dim)
        codebook = layer.embedding.to(flat_input.dtype)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(codebook ** 2, dim=1)
            - 2 * torch.matmul(flat_input, codebook.t())
        )
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, codebook)
        losses = losses + float(layer.commitment_cost) * torch.mean((quantized.detach() - residual) ** 2, dim=1)
        x_q = residual + (quantized - residual).detach()
        residual = residual - x_q
    return losses


def compute_train_style_loss_components(
    model,
    trajs: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
    loss_epoch: int,
) -> Dict[str, np.ndarray]:
    """
    逐条输出与 train_tfm.py TensorBoard loss/weight_* 对齐的诊断项。

    recon/vel/acc/turn/kin_smooth 都按单条轨迹计算；VQ 用同一 RVQ residual
    顺序计算 per-case commitment loss。weight_* 的系数与训练脚本一致。
    """
    kin_smooth_weight = 1e-2 if int(loss_epoch) > 30 else 0.0
    keys = [
        "train_recon_loss",
        "train_vq_loss",
        "train_vel_loss",
        "train_acc_loss",
        "train_kin_smooth_loss",
        "train_turn_global_loss",
        "train_turn_yaw_loss",
        "weight_recon",
        "weight_vq",
        "weight_vel",
        "weight_acc",
        "weight_kin_smooth",
        "weight_turn_global",
        "weight_turn_yaw",
        "weight",
    ]
    out = {key: [] for key in keys}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(trajs), int(batch_size)):
            batch = trajs[start:start + int(batch_size)]
            x_norm = normalize_trajs_torch(
                batch,
                mean=norm_params["mean"],
                std=norm_params["std"],
                scale_factor=norm_params["scale_factor"],
                clip_limit=norm_params["clip_limit"],
            )

            z = model.encode(x_norm)
            vq_loss_case = per_case_vq_loss_from_latent(model, z)
            x_recon, _, _, v, kappa = model(x_norm)

            recon_dxdy = torch.mean((x_recon[..., :2] - x_norm[..., :2]) ** 2, dim=(1, 2))
            recon_dyaw = torch.mean((x_recon[..., 2] - x_norm[..., 2]) ** 2, dim=1)
            recon_loss = recon_dxdy + 14.0 * recon_dyaw

            pred_phys = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
            gt_phys = (x_norm * model.norm_scale * model.norm_std) + model.norm_mean

            vx_pred = pred_phys[:, :, 0] / model.dt
            vy_pred = pred_phys[:, :, 1] / model.dt
            vx_gt = gt_phys[:, :, 0] / model.dt
            vy_gt = gt_phys[:, :, 1] / model.dt
            vel_loss = torch.mean((vx_pred - vx_gt) ** 2, dim=1) + 0.2 * torch.mean((vy_pred - vy_gt) ** 2, dim=1)

            if pred_phys.shape[1] > 1:
                ax_pred = (vx_pred[:, 1:] - vx_pred[:, :-1]) / model.dt
                ax_gt = (vx_gt[:, 1:] - vx_gt[:, :-1]) / model.dt
                ay_pred = (vy_pred[:, 1:] - vy_pred[:, :-1]) / model.dt
                ay_gt = (vy_gt[:, 1:] - vy_gt[:, :-1]) / model.dt
                acc_loss = torch.mean((ax_pred - ax_gt) ** 2, dim=1) + 0.2 * torch.mean((ay_pred - ay_gt) ** 2, dim=1)

                acc = (v[:, 1:] - v[:, :-1]) / model.dt
                kappa_rate = (kappa[:, 1:] - kappa[:, :-1]) / model.dt
                kin_smooth_loss = torch.mean(acc ** 2, dim=1) + torch.mean(kappa_rate ** 2, dim=1)
            else:
                acc_loss = torch.zeros_like(recon_loss)
                kin_smooth_loss = torch.zeros_like(recon_loss)

            net_yaw = torch.sum(gt_phys[:, :, 2], dim=1)
            turn_mask = torch.abs(net_yaw) > 0.35
            turn_global_loss = torch.zeros_like(recon_loss)
            turn_yaw_loss = torch.zeros_like(recon_loss)
            if torch.any(turn_mask):
                pred_xy = integrate_to_global_torch(pred_phys[turn_mask])
                gt_xy = integrate_to_global_torch(gt_phys[turn_mask])
                turn_global_loss[turn_mask] = torch.mean((pred_xy - gt_xy) ** 2, dim=(1, 2))

                pred_yaw = torch.cumsum(pred_phys[turn_mask, :, 2], dim=1)
                gt_yaw = torch.cumsum(gt_phys[turn_mask, :, 2], dim=1)
                turn_yaw_loss[turn_mask] = torch.mean((pred_yaw - gt_yaw) ** 2, dim=1)

            weighted = {
                "weight_recon": 10.0 * recon_loss,
                "weight_vq": 5.0 * vq_loss_case,
                "weight_vel": 0.5 * vel_loss,
                "weight_acc": 0.05 * acc_loss,
                "weight_kin_smooth": kin_smooth_weight * kin_smooth_loss,
                "weight_turn_global": turn_global_loss,
                "weight_turn_yaw": 2.0 * turn_yaw_loss,
            }
            weight_total = (
                weighted["weight_recon"]
                + weighted["weight_vq"]
                + weighted["weight_vel"]
                + weighted["weight_acc"]
                + weighted["weight_kin_smooth"]
                + weighted["weight_turn_global"]
                + weighted["weight_turn_yaw"]
            )

            batch_terms = {
                "train_recon_loss": recon_loss,
                "train_vq_loss": vq_loss_case,
                "train_vel_loss": vel_loss,
                "train_acc_loss": acc_loss,
                "train_kin_smooth_loss": kin_smooth_loss,
                "train_turn_global_loss": turn_global_loss,
                "train_turn_yaw_loss": turn_yaw_loss,
                **weighted,
                "weight": weight_total,
            }
            for key, value in batch_terms.items():
                out[key].append(value.detach().cpu().numpy())

    return {key: np.concatenate(chunks, axis=0).astype(np.float64) for key, chunks in out.items()}


def choose_visualization_indices(group_indices: np.ndarray, sample_idx: int, max_group_vis: int) -> np.ndarray:
    if max_group_vis <= 0:
        return np.asarray([], dtype=np.int64)
    # 可视化数量受 max_group_vis 限制，同时确保目标 worst case 一定被画出来。
    ordered = [int(sample_idx)]
    ordered.extend(int(v) for v in group_indices.tolist() if int(v) != int(sample_idx))
    return np.asarray(ordered[:max_group_vis], dtype=np.int64)


def decode_group_unique_first_tokens(model, codes: np.ndarray, norm_params: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造人工 token：group 内所有 unique 第一层 token，各自只用这一层 token decode。"""
    first_tokens = np.asarray(codes)[:, 0].astype(np.int64)
    values, counts = np.unique(first_tokens, return_counts=True)
    order = np.lexsort((values, -counts))  # count 高的排前，便于 overview 对比。
    values = values[order].astype(np.int64)
    counts = counts[order].astype(np.int64)

    token_tensor = torch.tensor(values[:, None], dtype=torch.long, device=norm_params["mean"].device)
    model.eval()
    with torch.no_grad():
        recon_norm = model.decode_from_codes(token_tensor)
        recon = denormalize_trajs_torch(
            recon_norm,
            mean=norm_params["mean"],
            std=norm_params["std"],
            scale_factor=norm_params["scale_factor"],
        )
    return recon.detach().cpu().numpy().astype(np.float32), values, counts


def plot_recon_only_case(
    sample_idx: int,
    pred_trajs: np.ndarray,
    dt: float,
    save_path: str,
    title: str,
    labels: Optional[List[str]] = None,
) -> None:
    """只画人工 token decode 出来的重建及运动学曲线，不画 GT；支持多条曲线叠加。"""
    pred_trajs = np.asarray(pred_trajs, dtype=np.float32)
    if pred_trajs.ndim == 2:
        pred_trajs = pred_trajs[None, ...]
    pred_prof = compute_kinematic_profiles(pred_trajs, dt=dt)
    pred_xy = pred_prof["xy"]
    t = np.arange(pred_trajs.shape[1]) * dt
    labels = labels or [f"Recon {i}" for i in range(pred_trajs.shape[0])]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ax_traj, ax_vx, ax_vy, ax_v = axes[0]
    ax_curv, ax_ax, ax_ay, ax_a = axes[1]
    cmap = plt.get_cmap("tab20")

    for i in range(pred_trajs.shape[0]):
        color = cmap(i % 20)
        ax_traj.plot(pred_xy[i, :, 0], pred_xy[i, :, 1], label=labels[i], linewidth=2.0, linestyle="--", color=color)
        ax_traj.scatter(pred_xy[i, 0, 0], pred_xy[i, 0, 1], c=[color], s=24)
        ax_traj.scatter(pred_xy[i, -1, 0], pred_xy[i, -1, 1], c=[color], s=24, marker="x")
    ax_traj.set_title("Trajectory")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis("equal")
    ax_traj.legend(fontsize=7)

    for ax, key, label, ylabel in [
        (ax_vx, "vx", "vx", "Velocity (m/s)"),
        (ax_vy, "vy", "vy", "Velocity (m/s)"),
        (ax_v, "speed", "v", "Speed (m/s)"),
        (ax_curv, "curvature", "Signed Curvature", "Curvature (1/m)"),
        (ax_ax, "ax", "ax", "Acceleration (m/s^2)"),
        (ax_ay, "ay", "ay", "Acceleration (m/s^2)"),
        (ax_a, "acc", "a", "Acceleration (m/s^2)"),
    ]:
        for i in range(pred_trajs.shape[0]):
            color = cmap(i % 20)
            ax.plot(t, pred_prof[key][i], label=labels[i], linewidth=1.8, linestyle="--", color=color)
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"{title} | sample_idx={sample_idx}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_group_overview(
    group_id: int,
    sample_indices: np.ndarray,
    gt_trajs: np.ndarray,
    pred_trajs: np.ndarray,
    metrics: Dict[str, np.ndarray],
    dt: float,
    save_path: str,
    total_group_size: int,
    artificial_pred_trajs: Optional[np.ndarray] = None,
    artificial_tokens: Optional[np.ndarray] = None,
    artificial_counts: Optional[np.ndarray] = None,
) -> None:
    if sample_indices.size == 0:
        return

    gt_prof = compute_kinematic_profiles(gt_trajs, dt=dt)
    pred_prof = compute_kinematic_profiles(pred_trajs, dt=dt)
    gt_xy = gt_prof["xy"]
    pred_xy = pred_prof["xy"]

    artificial_pred_trajs = None if artificial_pred_trajs is None else np.asarray(artificial_pred_trajs, dtype=np.float32)
    artificial_count_n = 0 if artificial_pred_trajs is None else int(artificial_pred_trajs.shape[0])
    artificial_tokens = np.asarray(artificial_tokens if artificial_tokens is not None else [], dtype=np.int64)
    artificial_counts = np.asarray(artificial_counts if artificial_counts is not None else [], dtype=np.int64)

    n = int(sample_indices.size)
    total_panels = n + artificial_count_n
    cols = min(4, total_panels)
    rows = int(math.ceil(total_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows), squeeze=False)
    flat_axes = axes.reshape(-1)

    for i, ax in enumerate(flat_axes):
        if i >= total_panels:
            ax.axis("off")
            continue
        if i >= n:
            j = i - n
            artificial_xy = compute_kinematic_profiles(artificial_pred_trajs[j:j + 1], dt=dt)["xy"][0]
            ax.plot(artificial_xy[:, 0], artificial_xy[:, 1], linewidth=2.4, linestyle="--", label="Artificial recon")
            ax.scatter(artificial_xy[0, 0], artificial_xy[0, 1], c="green", s=18)
            ax.scatter(artificial_xy[-1, 0], artificial_xy[-1, 1], c="orange", s=18)
            ax.set_title(
                f"first token={int(artificial_tokens[j])}\n"
                f"count={int(artificial_counts[j])}/{total_group_size}, only layer-1 used",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            ax.legend(fontsize=8)
            continue
        ax.plot(gt_xy[i, :, 0], gt_xy[i, :, 1], linewidth=2.0, label="GT")
        ax.plot(pred_xy[i, :, 0], pred_xy[i, :, 1], linewidth=2.0, linestyle="--", label="Recon")
        ax.scatter(gt_xy[i, 0, 0], gt_xy[i, 0, 1], c="green", s=18)
        ax.scatter(gt_xy[i, -1, 0], gt_xy[i, -1, 1], c="red", s=18)
        ax.scatter(pred_xy[i, -1, 0], pred_xy[i, -1, 1], c="orange", s=18)
        ax.set_title(
            f"sample_idx={int(sample_indices[i])}\n"
            f"MSE={float(metrics['recon_mse'][i]):.4g}, "
            f"ADE={float(metrics['ade'][i]):.2f}, FDE={float(metrics['fde'][i]):.2f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"group_id={group_id} overview | shown={n}/{total_group_size} | "
        f"unique first tokens={artificial_count_n}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    trajs = load_traj_array(args.data_path, dtype=np.float32)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]

    n_total = int(trajs.shape[0])

    requested_sample_idx = int(args.sample_idx)
    target_sample_idx = requested_sample_idx
    source_indices = None
    source_match_indices = np.asarray([], dtype=np.int64)
    if bool(args.sample_idx_is_source) and not str(args.source_indices_path).strip():
        raise ValueError("--sample_idx_is_source requires --source_indices_path.")
    if str(args.source_indices_path).strip():
        source_indices = np.load(args.source_indices_path).astype(np.int64).reshape(-1)
        if source_indices.shape[0] != n_total:
            raise ValueError(
                f"source_indices length mismatch: {source_indices.shape[0]} vs data N={n_total}"
            )
    if bool(args.sample_idx_is_source):
        source_match_indices = np.where(source_indices == requested_sample_idx)[0].astype(np.int64)
        if source_match_indices.size == 0:
            raise ValueError(
                f"original/source sample_idx={requested_sample_idx} was not found in filtered data. "
                "It may have been removed as an outlier."
            )
        occurrence = int(args.source_occurrence)
        if occurrence < 0 or occurrence >= source_match_indices.size:
            raise IndexError(
                f"--source_occurrence={occurrence} out of range for source sample "
                f"{requested_sample_idx}; found {source_match_indices.size} rows."
            )
        target_sample_idx = int(source_match_indices[occurrence])

    cache_dir, grouping = load_grouping_cache(
        group_cache_dir=args.group_cache_dir,
        group_cache_key=args.group_cache_key,
        n_total=n_total,
    )
    group_id, group_indices = find_sample_group(grouping, sample_idx=target_sample_idx)
    raw_group_indices = np.asarray(group_indices, dtype=np.int64)
    group_indices = unique_source_group_indices(
        group_indices=group_indices,
        source_indices=source_indices,
        target_sample_idx=target_sample_idx,
    )

    print(f"Loaded grouping cache: {cache_dir}")
    if bool(args.sample_idx_is_source):
        print(f"target source sample_idx: {requested_sample_idx}")
        print(
            f"target current sample_idx: {target_sample_idx} "
            f"(occurrence {int(args.source_occurrence) + 1}/{int(source_match_indices.size)})"
        )
        print("all current rows for this source sample_idx:")
        print(",".join(str(int(v)) for v in source_match_indices.tolist()))
    else:
        print(f"target sample_idx: {target_sample_idx}")
    print(f"group_id: {group_id}")
    if source_indices is not None and raw_group_indices.size != group_indices.size:
        print(f"raw group size: {int(raw_group_indices.size)}")
        print(f"unique-source group size: {int(group_indices.size)}")
    else:
        print(f"group size: {int(group_indices.size)}")
    print("group sample_idx list:")
    print(",".join(str(int(v)) for v in group_indices.tolist()))

    resolved_model_type = infer_model_type(args.model_path, args.model_type)
    model = build_model(
        model_path=args.model_path,
        input_steps=int(trajs.shape[1]),
        device=device,
        model_type=resolved_model_type,
        num_transformer_layers=int(args.num_transformer_layers),
        num_layers=int(args.num_layers),
    )
    norm_params = load_norm_params_torch(args.norm_path, device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])

    group_trajs = np.asarray(trajs[group_indices], dtype=np.float32)
    recon_trajs, codes = reconstruct_trajs(
        model=model,
        trajs=group_trajs,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=int(args.batch_size),
    )
    artificial_trajs, artificial_tokens, artificial_counts = decode_group_unique_first_tokens(
        model=model,
        codes=codes,
        norm_params=norm_params,
    )
    token_summary = ", ".join(
        f"{int(tok)}:{int(cnt)}" for tok, cnt in zip(artificial_tokens.tolist(), artificial_counts.tolist())
    )
    print(
        f"Artificial first-token recon: unique_count={int(artificial_tokens.size)} | "
        f"token:count = {token_summary}"
    )

    metrics = compute_reconstruction_case_metrics(group_trajs, recon_trajs, dt=float(args.dt))
    if resolved_model_type == "taae":
        train_loss_metrics = compute_train_style_loss_components(
            model=model,
            trajs=group_trajs,
            norm_params=norm_params,
            batch_size=int(args.batch_size),
            loss_epoch=int(args.loss_epoch),
        )
        metrics.update(train_loss_metrics)
    else:
        print("[warning] train_tfm.py weighted loss columns are only implemented for model_type=taae; skipping them.")
    target_positions = np.where(group_indices == int(target_sample_idx))[0]
    target_pos = int(target_positions[0])

    rows = []
    for pos, sample_idx in enumerate(group_indices.tolist()):
        row_metrics = metric_at(metrics, pos)
        rows.append(
            {
                "group_id": int(group_id),
                "sample_idx": int(sample_idx),
                "source_sample_idx": int(source_indices[int(sample_idx)]) if source_indices is not None else "",
                "is_target_case": int(int(sample_idx) == int(target_sample_idx)),
                "recon_mse": row_metrics["recon_mse"],
                "ade": row_metrics["ade"],
                "fde": row_metrics["fde"],
                "max_error": row_metrics["max_error"],
                **{
                    key: row_metrics[key]
                    for key in [
                        "train_recon_loss",
                        "train_vq_loss",
                        "train_vel_loss",
                        "train_acc_loss",
                        "train_kin_smooth_loss",
                        "train_turn_global_loss",
                        "train_turn_yaw_loss",
                        "weight_recon",
                        "weight_vq",
                        "weight_vel",
                        "weight_acc",
                        "weight_kin_smooth",
                        "weight_turn_global",
                        "weight_turn_yaw",
                        "weight",
                    ]
                    if key in row_metrics
                },
                "tokens": token_sequence_to_str(codes[pos]),
            }
        )

    csv_path = os.path.join(args.out_dir, "group_case_losses.csv")
    write_csv(rows, csv_path)

    stats = {
        name: {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in metrics.items()
    }
    ranks = {
        "recon_mse": rank_desc(metrics["recon_mse"], target_pos),
        "ade": rank_desc(metrics["ade"], target_pos),
        "fde": rank_desc(metrics["fde"], target_pos),
    }
    target_metrics = metric_at(metrics, target_pos)

    prefix_lengths = np.arange(1, int(codes.shape[1]) + 1, 2, dtype=np.int64)
    prefix_lengths, prefix_recon_trajs = decode_token_prefix_reconstructions(
        model=model,
        codes=codes[target_pos],
        norm_params=norm_params,
        prefix_lengths=prefix_lengths,
    )
    target_prefix_path = os.path.join(
        args.out_dir,
        f"target_sample_{int(target_sample_idx)}_token_prefix_recon.png",
    )
    plot_target_token_prefix_reconstructions(
        sample_idx=int(target_sample_idx),
        gt_traj=group_trajs[target_pos],
        full_recon_traj=recon_trajs[target_pos],
        prefix_lengths=prefix_lengths,
        prefix_recon_trajs=prefix_recon_trajs,
        dt=float(args.dt),
        save_path=target_prefix_path,
        tokens=codes[target_pos],
        metrics=target_metrics,
    )

    print("\nTarget case loss:")
    print(
        f"  recon_mse={target_metrics['recon_mse']:.6f}, "
        f"ADE={target_metrics['ade']:.4f}, FDE={target_metrics['fde']:.4f}, "
        f"max_error={target_metrics['max_error']:.4f}"
    )
    print("\nGroup loss stats:")
    for name in ["recon_mse", "ade", "fde", "max_error"]:
        s = stats[name]
        print(f"  {name}: mean={s['mean']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}")
    if "weight" in metrics:
        print("\nTrain-style weighted loss stats:")
        for name in [
            "weight_recon",
            "weight_vq",
            "weight_vel",
            "weight_acc",
            "weight_kin_smooth",
            "weight_turn_global",
            "weight_turn_yaw",
            "weight",
        ]:
            s = stats[name]
            print(f"  {name}: target={target_metrics[name]:.6f}, mean={s['mean']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}")
    print("\nTarget rank within group (1 = worst/highest loss):")
    print(
        f"  recon_mse: {ranks['recon_mse']}/{int(group_indices.size)} | "
        f"ADE: {ranks['ade']}/{int(group_indices.size)} | "
        f"FDE: {ranks['fde']}/{int(group_indices.size)}"
    )

    vis_indices = choose_visualization_indices(group_indices, int(target_sample_idx), int(args.max_group_vis))
    pos_by_sample = {int(sample_idx): pos for pos, sample_idx in enumerate(group_indices.tolist())}
    for sample_idx in vis_indices.tolist():
        pos = pos_by_sample[int(sample_idx)]
        save_path = os.path.join(args.out_dir, f"group_{group_id}_sample_{int(sample_idx)}.png")
        plot_representative_case(
            scenario_name=f"Group {group_id}",
            sample_idx=int(sample_idx),
            gt_traj=group_trajs[pos],
            pred_traj=recon_trajs[pos],
            dt=float(args.dt),
            save_path=save_path,
            sample_tokens=codes[pos],
            metrics=metric_at(metrics, pos),
        )

    # 人工 token 图单独保存，避免覆盖正常的 group_{gid}_sample_{idx}.png。
    artificial_sample_path = os.path.join(
        args.out_dir,
        f"group_{group_id}_sample_artificial_token.png",
    )
    artificial_labels = [
        f"first token={int(tok)} ({int(cnt)}/{int(group_indices.size)})"
        for tok, cnt in zip(artificial_tokens.tolist(), artificial_counts.tolist())
    ]
    plot_recon_only_case(
        sample_idx=int(target_sample_idx),
        pred_trajs=artificial_trajs,
        dt=float(args.dt),
        save_path=artificial_sample_path,
        title=f"Group {group_id} artificial first-token recons",
        labels=artificial_labels,
    )

    if vis_indices.size > 0:
        vis_positions = np.asarray([pos_by_sample[int(v)] for v in vis_indices.tolist()], dtype=np.int64)
        overview_metrics = {name: values[vis_positions] for name, values in metrics.items()}
        overview_path = os.path.join(args.out_dir, f"group_{group_id}_overview.png")
        plot_group_overview(
            group_id=group_id,
            sample_indices=vis_indices,
            gt_trajs=group_trajs[vis_positions],
            pred_trajs=recon_trajs[vis_positions],
            metrics=overview_metrics,
            dt=float(args.dt),
            save_path=overview_path,
            total_group_size=int(group_indices.size),
            artificial_pred_trajs=artificial_trajs,
            artificial_tokens=artificial_tokens,
            artificial_counts=artificial_counts,
        )

    summary = {
        "requested_sample_idx": int(requested_sample_idx),
        "sample_idx_is_source": bool(args.sample_idx_is_source),
        "target_sample_idx": int(target_sample_idx),
        "source_indices_path": os.path.abspath(args.source_indices_path) if source_indices is not None else "",
        "source_match_indices": [int(v) for v in source_match_indices.tolist()],
        "group_id": int(group_id),
        "group_size": int(group_indices.size),
        "group_sample_indices": [int(v) for v in group_indices.tolist()],
        "unique_source_group": bool(source_indices is not None),
        "raw_group_size": int(raw_group_indices.size),
        "raw_group_sample_indices": [int(v) for v in raw_group_indices.tolist()],
        "group_cache_dir": cache_dir,
        "model_path": os.path.abspath(args.model_path),
        "num_layers_arg": int(args.num_layers),
        "num_layers_used": int(codes.shape[1]) if codes.ndim == 2 else int(np.asarray(codes).shape[-1]),
        "model_type": resolved_model_type,
        "norm_path": os.path.abspath(args.norm_path),
        "data_path": os.path.abspath(args.data_path),
        "csv_path": os.path.abspath(csv_path),
        "target_metrics": target_metrics,
        "group_loss_stats": stats,
        "target_rank_desc": ranks,
        "visualized_sample_indices": [int(v) for v in vis_indices.tolist()],
        "artificial_first_tokens": [int(v) for v in artificial_tokens.tolist()],
        "artificial_first_token_counts": [int(v) for v in artificial_counts.tolist()],
        "artificial_first_token": int(artificial_tokens[0]) if artificial_tokens.size else None,
        "artificial_first_token_count": int(artificial_counts[0]) if artificial_counts.size else None,
        "artificial_token_plot_path": os.path.abspath(artificial_sample_path),
        "target_token_prefix_plot_path": os.path.abspath(target_prefix_path),
        "target_token_prefix_lengths": [int(v) for v in prefix_lengths.tolist()],
    }
    summary_path = os.path.join(args.out_dir, "group_case_analysis_summary.json")
    write_json(summary_path, summary)

    print(f"\nSaved per-case losses to: {csv_path}")
    print(f"Saved visualizations to: {args.out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()


# Example:
# python analyze_group_worst_case.py \
#   --model_path /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_kin_0311/pred_rvq_taae_model.pth \
#   --norm_path /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_kin_0311/pred_norm_params.pkl \
#   --data_path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120_group_loss_filtered.npy \
#   --sample_idx 971299 \
#   --group_cache_dir /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/similar_single_train/grouping_cache \
#   --out_dir /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_kin_0311/group_case_analysis \
#   --max_group_vis 16 \
#   --loss_epoch 31
