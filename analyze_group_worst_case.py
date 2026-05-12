import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RVQ_DIR = os.path.join(THIS_DIR, "rvq_transformer_vehdyn")
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
    load_norm_params_torch,
    load_traj_array,
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
    parser.add_argument("--model_type", "--model-type", dest="model_type", default="auto", choices=["auto", "taae", "accint"])
    parser.add_argument("--data_type", "--data-type", dest="data_type", default="pred", choices=["pred", "history"])
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4096)
    parser.add_argument("--num_layers", "--num-layers", dest="num_layers", type=int, default=0)
    parser.add_argument("--num_transformer_layers", "--num-transformer-layers", dest="num_transformer_layers", type=int, default=2)
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


def rank_desc(values: np.ndarray, target_pos: int) -> int:
    """按 loss 从大到小排名；rank=1 表示该指标在 group 内最差。"""
    return int(np.sum(values > values[target_pos]) + 1)


def metric_at(metrics: Dict[str, np.ndarray], pos: int) -> Dict[str, float]:
    return {
        "recon_mse": float(metrics["recon_mse"][pos]),
        "ade": float(metrics["ade"][pos]),
        "fde": float(metrics["fde"][pos]),
        "max_error": float(metrics["max_error"][pos]),
    }


def choose_visualization_indices(group_indices: np.ndarray, sample_idx: int, max_group_vis: int) -> np.ndarray:
    if max_group_vis <= 0:
        return np.asarray([], dtype=np.int64)
    # 可视化数量受 max_group_vis 限制，同时确保目标 worst case 一定被画出来。
    ordered = [int(sample_idx)]
    ordered.extend(int(v) for v in group_indices.tolist() if int(v) != int(sample_idx))
    return np.asarray(ordered[:max_group_vis], dtype=np.int64)


def plot_group_overview(
    group_id: int,
    sample_indices: np.ndarray,
    gt_trajs: np.ndarray,
    pred_trajs: np.ndarray,
    metrics: Dict[str, np.ndarray],
    dt: float,
    save_path: str,
    total_group_size: int,
) -> None:
    if sample_indices.size == 0:
        return

    gt_prof = compute_kinematic_profiles(gt_trajs, dt=dt)
    pred_prof = compute_kinematic_profiles(pred_trajs, dt=dt)
    gt_xy = gt_prof["xy"]
    pred_xy = pred_prof["xy"]

    n = int(sample_indices.size)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows), squeeze=False)
    flat_axes = axes.reshape(-1)

    for i, ax in enumerate(flat_axes):
        if i >= n:
            ax.axis("off")
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

    fig.suptitle(f"group_id={group_id} overview | shown={n}/{total_group_size}")
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
    cache_dir, grouping = load_grouping_cache(
        group_cache_dir=args.group_cache_dir,
        group_cache_key=args.group_cache_key,
        n_total=n_total,
    )
    group_id, group_indices = find_sample_group(grouping, sample_idx=int(args.sample_idx))

    print(f"Loaded grouping cache: {cache_dir}")
    print(f"target sample_idx: {int(args.sample_idx)}")
    print(f"group_id: {group_id}")
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

    metrics = compute_reconstruction_case_metrics(group_trajs, recon_trajs, dt=float(args.dt))
    target_positions = np.where(group_indices == int(args.sample_idx))[0]
    target_pos = int(target_positions[0])

    rows = []
    for pos, sample_idx in enumerate(group_indices.tolist()):
        row_metrics = metric_at(metrics, pos)
        rows.append(
            {
                "group_id": int(group_id),
                "sample_idx": int(sample_idx),
                "is_target_case": int(int(sample_idx) == int(args.sample_idx)),
                "recon_mse": row_metrics["recon_mse"],
                "ade": row_metrics["ade"],
                "fde": row_metrics["fde"],
                "max_error": row_metrics["max_error"],
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
    print("\nTarget rank within group (1 = worst/highest loss):")
    print(
        f"  recon_mse: {ranks['recon_mse']}/{int(group_indices.size)} | "
        f"ADE: {ranks['ade']}/{int(group_indices.size)} | "
        f"FDE: {ranks['fde']}/{int(group_indices.size)}"
    )

    vis_indices = choose_visualization_indices(group_indices, int(args.sample_idx), int(args.max_group_vis))
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
        )

    summary = {
        "sample_idx": int(args.sample_idx),
        "group_id": int(group_id),
        "group_size": int(group_indices.size),
        "group_sample_indices": [int(v) for v in group_indices.tolist()],
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
#   --model_path /path/to/rvq_tfm_kin_0311/pred_rvq_taae_model.pth \
#   --norm_path /path/to/rvq_tfm_kin_0311/pred_norm_params.pkl \
#   --data_path /path/to/all_datas.npy \
#   --sample_idx 12345 \
#   --group_cache_dir /path/to/similar_single_train/grouping_cache \
#   --out_dir /path/to/rvq_tfm_kin_0311/group_case_analysis \
#   --max_group_vis 16
