import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from analyze_group_worst_case import compute_train_style_loss_components
    from eval_tokenizer_by_scenario import plot_representative_case, reconstruct_trajs
    from experiment_similar_traj_single_train import build_model_and_norm, _expand_unique_source_grouping
    from filter_group_loss_outliers import find_group_outliers, normalize_csv_rows
    from grouping_pipeline import compute_motion_stats, find_representative_groups, save_grouping_cache
    from train_tfm import train_rvq_taae
    from utils import (
        compute_reconstruction_case_metrics_batched,
        load_sampled_datas,
        load_source_indices,
        percentiles,
        resolve_default_data_path,
        resolve_source_indices_path,
        select_global_hard_samples,
        select_top_metric_indices,
        source_indices_sidecar_path,
        summarize_reconstruction_case_metrics,
        write_csv,
        write_json,
    )
except ImportError:
    from rvq_transformer_vehdyn.analyze_group_worst_case import compute_train_style_loss_components
    from rvq_transformer_vehdyn.eval_tokenizer_by_scenario import plot_representative_case, reconstruct_trajs
    from rvq_transformer_vehdyn.experiment_similar_traj_single_train import build_model_and_norm, _expand_unique_source_grouping
    from rvq_transformer_vehdyn.filter_group_loss_outliers import find_group_outliers, normalize_csv_rows
    from rvq_transformer_vehdyn.grouping_pipeline import compute_motion_stats, find_representative_groups, save_grouping_cache
    from rvq_transformer_vehdyn.train_tfm import train_rvq_taae
    from rvq_transformer_vehdyn.utils import (
        compute_reconstruction_case_metrics_batched,
        load_sampled_datas,
        load_source_indices,
        percentiles,
        resolve_default_data_path,
        resolve_source_indices_path,
        select_global_hard_samples,
        select_top_metric_indices,
        source_indices_sidecar_path,
        summarize_reconstruction_case_metrics,
        write_csv,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively train, find hard-but-valid worst cases, duplicate them, and retrain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--source-indices-path", type=str, default="")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--output-root", type=str, default="./work_dirs/tokenizer/iterative_worst_case_train")
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default="",
        help="Resume an existing iterative run directory instead of creating a new timestamped run.",
    )
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-per-sample-arrays", action="store_true")
    parser.add_argument(
        "--num-max-error-plots",
        type=int,
        default=20,
        help="Plot this many highest per-sample max_error reconstructions for each iteration; set 0 to disable.",
    )
    parser.add_argument(
        "--allow-duplicate-source-max-error-plots",
        action="store_true",
        help="Do not dedupe max-error plots by source_sample_idx.",
    )
    parser.add_argument(
        "--no-track-first-iter-max-error-sources",
        dest="track_first_iter_max_error_sources",
        action="store_false",
        help="Disable plotting the first iteration's max-error source ids again in later iterations.",
    )
    parser.set_defaults(track_first_iter_max_error_sources=True)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--loss-epoch", type=int, default=31)
    parser.add_argument("--dt", type=float, default=0.2)

    parser.add_argument("--num-representative-groups", type=int, default=80000)
    parser.add_argument("--grouping-method", type=str, default="minibatch_kmeans", choices=["minibatch_kmeans", "kinematic_bins"])
    parser.add_argument("--grouping-stage", type=str, default="scenario_first", choices=["scenario_first", "global"])
    parser.add_argument("--group-feature", type=str, default="kinematic_plus_shape", choices=["kinematic", "shape", "kinematic_plus_shape"])
    parser.add_argument("--shape-downsample-steps", type=int, default=10)
    parser.add_argument("--feature-xy-weight", type=float, default=1.0)
    parser.add_argument("--feature-yaw-weight", type=float, default=3.0)
    parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=1)
    parser.add_argument("--kmeans-random-state", type=int, default=42)
    parser.add_argument("--no-group-unique-source", dest="group_unique_source", action="store_false")
    parser.set_defaults(group_unique_source=True)

    parser.add_argument("--large-group-min-size", type=int, default=4)
    parser.add_argument("--ratio-threshold", type=float, default=10.0)
    parser.add_argument("--iqr-mult", type=float, default=3.0)
    parser.add_argument("--hard-iqr-mult", type=float, default=1.0)
    parser.add_argument("--small-group-hard-ratio", type=float, default=5.0)
    parser.add_argument("--small-group-remove-ratio", type=float, default=12.0)
    parser.add_argument("--global-hard-percentile", type=float, default=80.0)
    parser.add_argument("--global-remove-percentile", type=float, default=99.5)
    parser.add_argument("--duplicate-hard-count", type=int, default=5)
    parser.add_argument(
        "--global-fallback-max-per-group",
        type=int,
        default=1,
        help="Extra absolute/global hard samples to duplicate per bad group, even if the whole group is bad together.",
    )
    parser.add_argument("--global-fallback-max-total", type=int, default=0)
    parser.add_argument("--max-duplicate-unique-per-iter", type=int, default=0)

    parser.add_argument("--vrr-threshold-m", type=float, default=1.0)
    parser.add_argument("--target-vrr", type=float, default=0.95)
    parser.add_argument("--target-ade", type=float, default=-1.0)
    parser.add_argument("--target-fde", type=float, default=-1.0)
    parser.add_argument("--target-max-error", type=float, default=-1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def model_view(trajs: np.ndarray, data_type: str) -> np.ndarray:
    if data_type == "history":
        return np.asarray(trajs[:, :14, :], dtype=np.float32)
    return np.asarray(trajs, dtype=np.float32)


def save_dataset_with_source(trajs: np.ndarray, source_indices: np.ndarray, out_path: str) -> Tuple[str, str]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, np.asarray(trajs, dtype=np.float32))
    source_path = source_indices_sidecar_path(out_path)
    np.save(source_path, np.asarray(source_indices, dtype=np.int64))
    return out_path, source_path


def load_json_if_exists(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def iteration_dir(output_root: str, iteration: int) -> str:
    return os.path.join(output_root, f"iter_{int(iteration):02d}")


def model_artifacts_exist(iter_dir: str, data_type: str) -> bool:
    model_dir = os.path.join(iter_dir, "model")
    model_path = os.path.join(model_dir, f"{data_type}_rvq_taae_model.pth")
    norm_path = os.path.join(model_dir, f"{data_type}_norm_params.pkl")
    return os.path.exists(model_path) and os.path.exists(norm_path)


def iteration_complete(iter_dir: str) -> bool:
    next_path = os.path.join(iter_dir, "next_dataset.npy")
    next_source_path = source_indices_sidecar_path(next_path)
    summary_path = os.path.join(iter_dir, "iteration_summary.json")
    return os.path.exists(summary_path) and os.path.exists(next_path) and os.path.exists(next_source_path)


def existing_iteration_numbers(output_root: str) -> List[int]:
    if not os.path.isdir(output_root):
        return []
    numbers = []
    for name in os.listdir(output_root):
        if not name.startswith("iter_"):
            continue
        suffix = name[len("iter_"):]
        if suffix.isdigit() and os.path.isdir(os.path.join(output_root, name)):
            numbers.append(int(suffix))
    return sorted(numbers)


def load_dataset_state(dataset_path: str, source_path: str = "") -> Tuple[np.ndarray, np.ndarray, str, str]:
    dataset_path = os.path.abspath(str(dataset_path))
    trajs = np.asarray(load_sampled_datas(dataset_path), dtype=np.float32)
    n_total = int(trajs.shape[0])

    source_path = str(source_path or "").strip()
    if not source_path:
        candidate = source_indices_sidecar_path(dataset_path)
        source_path = candidate if os.path.exists(candidate) else ""

    if source_path:
        source_path = os.path.abspath(source_path)
        source_indices = load_source_indices(source_path, n_total=n_total)
    else:
        source_indices = np.arange(n_total, dtype=np.int64)
    return trajs, source_indices.astype(np.int64), dataset_path, source_path


def load_tracked_max_error_sources_from_run(output_root: str) -> Optional[np.ndarray]:
    first_summary = load_json_if_exists(
        os.path.join(iteration_dir(output_root, 1), "max_error_visualization_summary.json")
    )
    if first_summary is not None:
        values = first_summary.get("source_sample_indices", [])
        if values:
            return np.asarray(values, dtype=np.int64).reshape(-1)

    for iteration in existing_iteration_numbers(output_root):
        summary = load_json_if_exists(
            os.path.join(
                iteration_dir(output_root, iteration),
                "tracked_max_error_source_visualization_summary.json",
            )
        )
        if summary is None:
            continue
        values = summary.get("tracked_source_indices", [])
        if values:
            return np.asarray(values, dtype=np.int64).reshape(-1)
    return None


def discover_resume_state(
    output_root: str,
    args: argparse.Namespace,
    data_path: str,
    source_path: str,
) -> Tuple[np.ndarray, np.ndarray, int, List[Dict[str, object]], str, str, Optional[np.ndarray], Dict[str, object]]:
    numbers = existing_iteration_numbers(output_root)
    max_seen = max(numbers) if numbers else 0

    history: List[Dict[str, object]] = []
    last_complete = 0
    for iteration in range(1, max_seen + 1):
        iter_dir = iteration_dir(output_root, iteration)
        if not iteration_complete(iter_dir):
            break
        iter_summary = load_json_if_exists(os.path.join(iter_dir, "iteration_summary.json"))
        if iter_summary is None:
            break
        history.append(iter_summary)
        last_complete = iteration

    start_iteration = int(last_complete + 1)
    final_dataset_path = ""
    final_source_path = ""
    if last_complete > 0:
        prev_iter_dir = iteration_dir(output_root, last_complete)
        prev_summary = history[-1]
        final_dataset_path = str(prev_summary.get("next_dataset_path") or os.path.join(prev_iter_dir, "next_dataset.npy"))
        final_source_path = str(
            prev_summary.get("next_source_indices_path") or source_indices_sidecar_path(final_dataset_path)
        )

    current_dataset_path = ""
    current_source_path = ""
    resume_from = "initial_data"
    reuse_input_iteration = 0
    reuse_model_iteration = 0

    incomplete_iter_dir = iteration_dir(output_root, start_iteration)
    incomplete_input_path = os.path.join(incomplete_iter_dir, "input_dataset.npy")
    incomplete_input_source_path = source_indices_sidecar_path(incomplete_input_path)
    if os.path.exists(incomplete_input_path):
        current_dataset_path = incomplete_input_path
        current_source_path = incomplete_input_source_path if os.path.exists(incomplete_input_source_path) else ""
        resume_from = "incomplete_iteration_input"
        reuse_input_iteration = int(start_iteration)
        if model_artifacts_exist(incomplete_iter_dir, args.data_type):
            reuse_model_iteration = int(start_iteration)
    elif last_complete > 0:
        current_dataset_path = final_dataset_path
        current_source_path = final_source_path
        resume_from = "last_complete_next_dataset"
    else:
        current_dataset_path = data_path
        current_source_path = source_path
        start_iteration = 1

    current_trajs, current_source_indices, loaded_dataset_path, loaded_source_path = load_dataset_state(
        current_dataset_path, current_source_path
    )
    tracked_sources = load_tracked_max_error_sources_from_run(output_root)

    resume_info: Dict[str, object] = {
        "enabled": True,
        "resume_from": resume_from,
        "existing_iterations": [int(v) for v in numbers],
        "last_complete_iteration": int(last_complete),
        "start_iteration": int(start_iteration),
        "loaded_dataset_path": loaded_dataset_path,
        "loaded_source_indices_path": loaded_source_path,
        "reuse_input_iteration": int(reuse_input_iteration),
        "reuse_model_iteration": int(reuse_model_iteration),
        "tracked_source_count": int(0 if tracked_sources is None else tracked_sources.size),
    }
    return (
        current_trajs,
        current_source_indices,
        int(start_iteration),
        history,
        final_dataset_path,
        final_source_path,
        tracked_sources,
        resume_info,
    )


def build_iteration_grouping(
    model_trajs: np.ndarray,
    source_indices: np.ndarray,
    args: argparse.Namespace,
    iter_dir: str,
    iteration: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    n_total = int(model_trajs.shape[0])
    grouping_input = model_trajs
    grouping_source_mode = False
    unique_first_rows = None
    unique_inverse = None

    if bool(args.group_unique_source) and source_indices is not None:
        _, unique_first_rows, unique_inverse = np.unique(
            np.asarray(source_indices, dtype=np.int64),
            return_index=True,
            return_inverse=True,
        )
        if unique_first_rows.shape[0] < n_total:
            grouping_input = model_trajs[unique_first_rows]
            grouping_source_mode = True

    requested_groups = int(max(1, min(int(args.num_representative_groups), int(grouping_input.shape[0]))))
    motion_stats = compute_motion_stats(grouping_input, dt=float(args.dt))
    grouping = find_representative_groups(
        trajs=grouping_input,
        dt=float(args.dt),
        num_groups=requested_groups,
        grouping_method=args.grouping_method,
        grouping_stage=args.grouping_stage,
        group_feature=args.group_feature,
        shape_downsample_steps=int(args.shape_downsample_steps),
        feature_xy_weight=float(args.feature_xy_weight),
        feature_yaw_weight=float(args.feature_yaw_weight),
        kmeans_batch_size=int(args.kmeans_batch_size),
        kmeans_max_iter=int(args.kmeans_max_iter),
        kmeans_n_init=int(args.kmeans_n_init),
        kmeans_random_state=int(args.kmeans_random_state),
        seed=int(args.seed) + int(iteration) * 1009,
        motion_stats=motion_stats,
    )
    if grouping_source_mode:
        grouping = _expand_unique_source_grouping(
            grouping=grouping,
            unique_first_rows=np.asarray(unique_first_rows, dtype=np.int64),
            unique_inverse=np.asarray(unique_inverse, dtype=np.int64),
            n_total=n_total,
        )

    group_sizes = np.asarray(grouping["group_sizes"], dtype=np.int64)
    meta = {
        "iteration": int(iteration),
        "grouping_method_used": str(grouping["grouping_method_used"]),
        "grouping_stage_used": str(grouping.get("grouping_stage_used", args.grouping_stage)),
        "grouping_method_requested": args.grouping_method,
        "grouping_stage_requested": args.grouping_stage,
        "group_feature": args.group_feature,
        "raw_count": int(n_total),
        "grouping_input_count": int(grouping_input.shape[0]),
        "group_unique_source": bool(grouping_source_mode),
        "requested_num_groups": int(requested_groups),
        "actual_num_groups": int(grouping["actual_num_groups"]),
        "group_size_percentiles": percentiles(group_sizes.astype(np.float32)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_dir = os.path.join(iter_dir, "grouping_cache")
    save_grouping_cache(cache_dir, grouping, meta=meta)
    meta["grouping_cache_dir"] = os.path.abspath(cache_dir)
    return grouping, meta


def targets_met(eval_summary: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    checks = []
    failures = []
    vrr_key = f"vrr_{float(args.vrr_threshold_m):g}m"
    if float(args.target_vrr) >= 0.0:
        ok = float(eval_summary.get(vrr_key, 0.0)) >= float(args.target_vrr)
        checks.append(f"{vrr_key}>={float(args.target_vrr):.4g}")
        if not ok:
            failures.append(f"{vrr_key}={float(eval_summary.get(vrr_key, 0.0)):.6g}")
    if float(args.target_ade) >= 0.0:
        ok = float(eval_summary.get("ade_mean", 0.0)) <= float(args.target_ade)
        checks.append(f"ade_mean<={float(args.target_ade):.4g}")
        if not ok:
            failures.append(f"ade_mean={float(eval_summary.get('ade_mean', 0.0)):.6g}")
    if float(args.target_fde) >= 0.0:
        ok = float(eval_summary.get("fde_mean", 0.0)) <= float(args.target_fde)
        checks.append(f"fde_mean<={float(args.target_fde):.4g}")
        if not ok:
            failures.append(f"fde_mean={float(eval_summary.get('fde_mean', 0.0)):.6g}")
    if float(args.target_max_error) >= 0.0:
        ok = float(eval_summary.get("max_error_mean", 0.0)) <= float(args.target_max_error)
        checks.append(f"max_error_mean<={float(args.target_max_error):.4g}")
        if not ok:
            failures.append(f"max_error_mean={float(eval_summary.get('max_error_mean', 0.0)):.6g}")
    if not checks:
        return False, ["no target metric enabled"]
    return len(failures) == 0, failures


def cap_duplicate_indices(indices: np.ndarray, scores: np.ndarray, max_count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if int(max_count) <= 0 or indices.size <= int(max_count):
        return indices
    order = np.argsort(scores[indices])[::-1][: int(max_count)]
    return indices[order].astype(np.int64)


def save_max_error_visualizations(
    *,
    model,
    trajs: np.ndarray,
    source_indices: np.ndarray,
    recon_metrics: dict,
    norm_params: dict,
    args: argparse.Namespace,
    iter_dir: str,
    iteration: int,
) -> dict:
    """Save GT-vs-reconstruction plots for the highest max_error samples in this iteration."""
    top_indices = select_top_metric_indices(
        metrics=recon_metrics,
        metric_key="max_error",
        top_k=int(args.num_max_error_plots),
        source_indices=source_indices,
        unique_source=not bool(args.allow_duplicate_source_max_error_plots),
    )
    if top_indices.size == 0:
        return {"enabled": False, "count": 0, "vis_dir": "", "csv_path": ""}

    vis_dir = os.path.join(iter_dir, "max_error_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    subset = np.asarray(trajs[top_indices], dtype=np.float32)
    recon_subset, codes = reconstruct_trajs(
        model=model,
        trajs=subset,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=max(1, min(int(args.eval_batch_size), int(subset.shape[0]))),
    )

    rows = []
    source = np.asarray(source_indices, dtype=np.int64).reshape(-1) if source_indices is not None else None
    for rank, sample_idx in enumerate(top_indices.tolist(), start=1):
        sample_idx = int(sample_idx)
        source_idx = int(source[sample_idx]) if source is not None else sample_idx
        metrics = {
            "recon_mse": float(np.asarray(recon_metrics["recon_mse"])[sample_idx]),
            "ade": float(np.asarray(recon_metrics["ade"])[sample_idx]),
            "fde": float(np.asarray(recon_metrics["fde"])[sample_idx]),
            "max_error": float(np.asarray(recon_metrics["max_error"])[sample_idx]),
        }
        filename = (
            f"rank_{rank:02d}_sample_{sample_idx}_source_{source_idx}_"
            f"maxerr_{metrics['max_error']:.3f}m.png"
        )
        save_path = os.path.join(vis_dir, filename)
        plot_representative_case(
            scenario_name=f"Iter {int(iteration):02d} MaxError #{rank:02d}",
            sample_idx=sample_idx,
            gt_traj=subset[rank - 1],
            pred_traj=recon_subset[rank - 1],
            dt=float(args.dt),
            save_path=save_path,
            sample_tokens=codes[rank - 1] if codes is not None else None,
            metrics=metrics,
        )
        rows.append(
            {
                "rank": int(rank),
                "sample_idx": sample_idx,
                "source_sample_idx": source_idx,
                "recon_mse": metrics["recon_mse"],
                "ade": metrics["ade"],
                "fde": metrics["fde"],
                "max_error": metrics["max_error"],
                "plot_path": os.path.abspath(save_path),
            }
        )

    csv_path = os.path.join(vis_dir, "max_error_visualizations.csv")
    write_csv(rows, csv_path)
    return {
        "enabled": True,
        "count": int(len(rows)),
        "vis_dir": os.path.abspath(vis_dir),
        "csv_path": os.path.abspath(csv_path),
        "indices": [int(v) for v in top_indices.tolist()],
        "source_sample_indices": [int(row["source_sample_idx"]) for row in rows],
    }


def save_tracked_source_visualizations(
    *,
    model,
    trajs: np.ndarray,
    source_indices: np.ndarray,
    tracked_sources: np.ndarray,
    recon_metrics: dict,
    norm_params: dict,
    args: argparse.Namespace,
    iter_dir: str,
    iteration: int,
) -> dict:
    """Plot the same source ids across iterations to inspect whether they improve."""
    tracked_sources = np.asarray(tracked_sources, dtype=np.int64).reshape(-1)
    if tracked_sources.size == 0:
        return {"enabled": False, "count": 0, "vis_dir": "", "csv_path": ""}

    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    max_error = np.asarray(recon_metrics["max_error"], dtype=np.float64).reshape(-1)
    current_indices = []
    rows_base = []
    for rank, source_idx in enumerate(tracked_sources.tolist(), start=1):
        source_idx = int(source_idx)
        matches = np.flatnonzero(source == source_idx).astype(np.int64)
        if matches.size == 0:
            rows_base.append(
                {
                    "tracked_rank": int(rank),
                    "source_sample_idx": source_idx,
                    "current_sample_idx": "",
                    "num_current_rows_for_source": 0,
                    "missing": True,
                }
            )
            continue
        # If the source has been duplicated, plot the current row with the largest max_error.
        chosen = int(matches[np.argmax(max_error[matches])])
        current_indices.append(chosen)
        rows_base.append(
            {
                "tracked_rank": int(rank),
                "source_sample_idx": source_idx,
                "current_sample_idx": chosen,
                "num_current_rows_for_source": int(matches.size),
                "missing": False,
            }
        )

    if not current_indices:
        return {"enabled": True, "count": 0, "vis_dir": "", "csv_path": "", "rows": rows_base}

    current_indices_np = np.asarray(current_indices, dtype=np.int64)
    vis_dir = os.path.join(iter_dir, "tracked_max_error_sources")
    os.makedirs(vis_dir, exist_ok=True)
    subset = np.asarray(trajs[current_indices_np], dtype=np.float32)
    recon_subset, codes = reconstruct_trajs(
        model=model,
        trajs=subset,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=max(1, min(int(args.eval_batch_size), int(subset.shape[0]))),
    )

    rows = []
    local_pos = 0
    for row in rows_base:
        if bool(row.get("missing", False)):
            rows.append(row)
            continue
        sample_idx = int(row["current_sample_idx"])
        metrics = {
            "recon_mse": float(np.asarray(recon_metrics["recon_mse"])[sample_idx]),
            "ade": float(np.asarray(recon_metrics["ade"])[sample_idx]),
            "fde": float(np.asarray(recon_metrics["fde"])[sample_idx]),
            "max_error": float(np.asarray(recon_metrics["max_error"])[sample_idx]),
        }
        filename = (
            f"tracked_{int(row['tracked_rank']):02d}_source_{int(row['source_sample_idx'])}_"
            f"sample_{sample_idx}_maxerr_{metrics['max_error']:.3f}m.png"
        )
        save_path = os.path.join(vis_dir, filename)
        plot_representative_case(
            scenario_name=f"Iter {int(iteration):02d} TrackedSource #{int(row['tracked_rank']):02d}",
            sample_idx=sample_idx,
            gt_traj=subset[local_pos],
            pred_traj=recon_subset[local_pos],
            dt=float(args.dt),
            save_path=save_path,
            sample_tokens=codes[local_pos] if codes is not None else None,
            metrics=metrics,
        )
        out_row = dict(row)
        out_row.update(
            {
                "recon_mse": metrics["recon_mse"],
                "ade": metrics["ade"],
                "fde": metrics["fde"],
                "max_error": metrics["max_error"],
                "plot_path": os.path.abspath(save_path),
            }
        )
        rows.append(out_row)
        local_pos += 1

    csv_path = os.path.join(vis_dir, "tracked_max_error_sources.csv")
    write_csv(rows, csv_path)
    return {
        "enabled": True,
        "count": int(local_pos),
        "vis_dir": os.path.abspath(vis_dir),
        "csv_path": os.path.abspath(csv_path),
        "tracked_source_indices": [int(v) for v in tracked_sources.tolist()],
    }


def main() -> None:
    args = parse_args()
    if int(args.epochs) <= 5:
        raise ValueError("train_rvq_taae uses a 5-epoch warmup scheduler; set --epochs > 5.")
    if int(args.max_iters) <= 0:
        raise ValueError("--max-iters must be positive.")

    set_seed(int(args.seed))
    run_name = time.strftime("%Y%m%d_%H%M%S")
    resume_mode = bool(str(args.resume_run_dir or "").strip())
    if resume_mode:
        output_root = os.path.abspath(str(args.resume_run_dir))
        if not os.path.isdir(output_root):
            raise FileNotFoundError(f"--resume-run-dir does not exist: {output_root}")
    else:
        output_root = os.path.abspath(os.path.join(args.output_root, f"{args.data_type}_iterative_{run_name}"))
        os.makedirs(output_root, exist_ok=True)

    previous_config = (
        load_json_if_exists(os.path.join(output_root, "iterative_worst_case_config.json"))
        if resume_mode
        else None
    ) or {}
    previous_args = previous_config.get("args", {}) if isinstance(previous_config.get("args", {}), dict) else {}

    data_path = str(args.data_path or previous_config.get("data_path") or resolve_default_data_path())
    source_arg = str(args.source_indices_path or previous_config.get("source_indices_path") or "")
    source_path = resolve_source_indices_path(data_path, source_arg)

    if resume_mode:
        (
            current_trajs,
            current_source_indices,
            start_iteration,
            history,
            final_dataset_path,
            final_source_path,
            tracked_max_error_sources,
            resume_info,
        ) = discover_resume_state(output_root, args, data_path, source_path)
        print("=" * 80)
        print("Resume Iterative Worst-Case Experiment")
        print(f"run_dir: {output_root}")
        print(f"resume_from: {resume_info['resume_from']}")
        print(f"last_complete_iteration: {resume_info['last_complete_iteration']}")
        print(f"start_iteration: {resume_info['start_iteration']}")
        print(f"loaded_dataset_path: {resume_info['loaded_dataset_path']}")
        print("=" * 80)
    else:
        current_trajs, current_source_indices, data_path, source_path = load_dataset_state(data_path, source_path)
        start_iteration = 1
        history = []
        final_dataset_path = ""
        final_source_path = ""
        tracked_max_error_sources = None
        resume_info: Dict[str, object] = {
            "enabled": False,
            "start_iteration": int(start_iteration),
            "last_complete_iteration": 0,
            "reuse_input_iteration": 0,
            "reuse_model_iteration": 0,
        }

    stop_reason = "max_iters_reached"
    if int(start_iteration) > int(args.max_iters):
        stop_reason = "resume_no_remaining_iterations"

    if resume_mode and previous_args:
        previous_data_type = previous_args.get("data_type")
        if previous_data_type and str(previous_data_type) != str(args.data_type):
            print(
                f"[resume-warning] previous data_type={previous_data_type}, "
                f"current data_type={args.data_type}"
            )

    config = {
        "data_path": os.path.abspath(data_path),
        "source_indices_path": os.path.abspath(source_path) if source_path else "",
        "output_root": output_root,
        "resume": resume_info,
        "args": vars(args),
    }
    config_name = (
        f"iterative_worst_case_resume_config_{run_name}.json"
        if resume_mode
        else "iterative_worst_case_config.json"
    )
    config_path = os.path.join(output_root, config_name)
    write_json(config_path, config)

    for iteration in range(int(start_iteration), int(args.max_iters) + 1):
        iter_dir = os.path.join(output_root, f"iter_{iteration:02d}")
        os.makedirs(iter_dir, exist_ok=True)
        model_dir = os.path.join(iter_dir, "model")

        expected_input_path = os.path.join(iter_dir, "input_dataset.npy")
        expected_input_source_path = source_indices_sidecar_path(expected_input_path)
        if (
            int(resume_info.get("reuse_input_iteration", 0)) == int(iteration)
            and os.path.exists(expected_input_path)
            and os.path.exists(expected_input_source_path)
        ):
            iter_input_path = expected_input_path
            iter_input_source_path = expected_input_source_path
            print(f"[resume] using existing iteration input: {iter_input_path}")
        else:
            iter_input_path, iter_input_source_path = save_dataset_with_source(
                current_trajs,
                current_source_indices,
                expected_input_path,
            )
        train_trajs = model_view(current_trajs, args.data_type)

        print("=" * 80)
        print(f"Iteration {iteration}/{int(args.max_iters)}")
        print(f"input_count: {int(current_trajs.shape[0])}")
        print(f"train_shape: {train_trajs.shape}")
        print(f"model_dir: {model_dir}")
        print("=" * 80)

        reuse_existing_model = (
            int(resume_info.get("reuse_model_iteration", 0)) == int(iteration)
            and model_artifacts_exist(iter_dir, args.data_type)
        )
        if reuse_existing_model:
            print(f"[resume] reusing existing model artifacts in {model_dir}")
        else:
            train_rvq_taae(
                train_trajs,
                save_dir=model_dir,
                data_type=args.data_type,
                batch_size=int(args.batch_size),
                num_layers=int(args.num_layers),
                num_transformer_layers=int(args.num_transformer_layers),
                epochs=int(args.epochs),
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, norm_params, model_path = build_model_and_norm(
            save_dir=model_dir,
            data_type=args.data_type,
            input_steps=int(train_trajs.shape[1]),
            num_layers=int(args.num_layers),
            num_transformer_layers=int(args.num_transformer_layers),
            device=device,
        )

        recon_metrics = compute_reconstruction_case_metrics_batched(
            model=model,
            trajs=train_trajs,
            norm_params=norm_params,
            batch_size=max(1, int(args.eval_batch_size)),
            dt=float(args.dt),
        )
        eval_summary = summarize_reconstruction_case_metrics(
            recon_metrics,
            vrr_threshold=float(args.vrr_threshold_m),
        )
        metric_path = os.path.join(iter_dir, "eval_summary.json")
        write_json(metric_path, eval_summary)
        max_error_visualization = save_max_error_visualizations(
            model=model,
            trajs=train_trajs,
            source_indices=current_source_indices,
            recon_metrics=recon_metrics,
            norm_params=norm_params,
            args=args,
            iter_dir=iter_dir,
            iteration=iteration,
        )
        write_json(os.path.join(iter_dir, "max_error_visualization_summary.json"), max_error_visualization)
        if (
            bool(args.track_first_iter_max_error_sources)
            and tracked_max_error_sources is None
            and bool(max_error_visualization.get("enabled", False))
        ):
            tracked_max_error_sources = np.asarray(
                max_error_visualization.get("source_sample_indices", []),
                dtype=np.int64,
            )
        tracked_max_error_visualization = save_tracked_source_visualizations(
            model=model,
            trajs=train_trajs,
            source_indices=current_source_indices,
            tracked_sources=np.zeros((0,), dtype=np.int64) if tracked_max_error_sources is None else tracked_max_error_sources,
            recon_metrics=recon_metrics,
            norm_params=norm_params,
            args=args,
            iter_dir=iter_dir,
            iteration=iteration,
        )
        write_json(
            os.path.join(iter_dir, "tracked_max_error_source_visualization_summary.json"),
            tracked_max_error_visualization,
        )

        ok, failures = targets_met(eval_summary, args)
        if ok:
            stop_reason = "target_metrics_met"
            final_dataset_path = iter_input_path
            final_source_path = iter_input_source_path
            history.append(
                {
                    "iteration": int(iteration),
                    "input_count": int(current_trajs.shape[0]),
                    "model_path": os.path.abspath(model_path),
                    "eval_summary": eval_summary,
                    "max_error_visualization": max_error_visualization,
                    "tracked_max_error_visualization": tracked_max_error_visualization,
                    "stop_after_eval": True,
                }
            )
            print(f"[stop] target metrics met at iteration {iteration}.")
            break

        print(f"[eval] targets not met: {', '.join(failures)}")
        grouping, grouping_meta = build_iteration_grouping(
            model_trajs=train_trajs,
            source_indices=current_source_indices,
            args=args,
            iter_dir=iter_dir,
            iteration=iteration,
        )

        print("Computing train-style loss components for hard-case selection...")
        losses = compute_train_style_loss_components(
            model=model,
            trajs=train_trajs,
            norm_params=norm_params,
            batch_size=max(1, int(args.eval_batch_size)),
            loss_epoch=int(args.loss_epoch),
        )
        weights = np.asarray(losses["weight"], dtype=np.float64)
        ades = np.asarray(recon_metrics["ade"], dtype=np.float64)
        fdes = np.asarray(recon_metrics["fde"], dtype=np.float64)

        global_thresholds = {
            "hard_weight": float(np.percentile(weights, float(args.global_hard_percentile))),
            "hard_ade": float(np.percentile(ades, float(args.global_hard_percentile))),
            "hard_fde": float(np.percentile(fdes, float(args.global_hard_percentile))),
            "remove_weight": float(np.percentile(weights, float(args.global_remove_percentile))),
        }

        remove_mask, removed_rows, group_duplicate_indices, group_duplicate_rows = find_group_outliers(
            weights=weights,
            ades=ades,
            fdes=fdes,
            group_to_indices=grouping["group_to_indices"],
            large_group_min_size=int(args.large_group_min_size),
            ratio_threshold=float(args.ratio_threshold),
            iqr_mult=float(args.iqr_mult),
            hard_iqr_mult=float(args.hard_iqr_mult),
            small_group_hard_ratio=float(args.small_group_hard_ratio),
            small_group_remove_ratio=float(args.small_group_remove_ratio),
            global_thresholds=global_thresholds,
        )

        global_duplicate_indices, global_duplicate_rows = select_global_hard_samples(
            weights=weights,
            ades=ades,
            fdes=fdes,
            group_to_indices=grouping["group_to_indices"],
            global_thresholds=global_thresholds,
            remove_mask=remove_mask,
            existing_duplicate_indices=group_duplicate_indices,
            max_per_group=int(args.global_fallback_max_per_group),
            max_total=int(args.global_fallback_max_total),
        )

        duplicate_indices = np.unique(
            np.concatenate([group_duplicate_indices, global_duplicate_indices], axis=0)
            if group_duplicate_indices.size or global_duplicate_indices.size
            else np.zeros((0,), dtype=np.int64)
        ).astype(np.int64)
        hard_weight = max(float(global_thresholds["hard_weight"]), 1e-8)
        hard_ade = max(float(global_thresholds["hard_ade"]), 1e-8)
        hard_fde = max(float(global_thresholds["hard_fde"]), 1e-8)
        hard_score = np.maximum.reduce([weights / hard_weight, ades / hard_ade, fdes / hard_fde])
        duplicate_indices = cap_duplicate_indices(
            duplicate_indices,
            scores=hard_score,
            max_count=int(args.max_duplicate_unique_per_iter),
        )

        kept_indices = np.where(~remove_mask)[0].astype(np.int64)
        removed_indices = np.where(remove_mask)[0].astype(np.int64)
        append_indices = np.repeat(duplicate_indices, max(0, int(args.duplicate_hard_count))).astype(np.int64)
        output_indices = np.concatenate([kept_indices, append_indices], axis=0).astype(np.int64)

        removed_csv = os.path.join(iter_dir, "removed_group_loss_outliers.csv")
        group_dup_csv = os.path.join(iter_dir, "duplicated_group_relative_hard_samples.csv")
        global_dup_csv = os.path.join(iter_dir, "duplicated_global_hard_fallback_samples.csv")
        write_csv(normalize_csv_rows(removed_rows), removed_csv)
        write_csv(normalize_csv_rows(group_duplicate_rows), group_dup_csv)
        write_csv(normalize_csv_rows(global_duplicate_rows), global_dup_csv)

        np.save(os.path.join(iter_dir, "kept_indices.npy"), kept_indices)
        np.save(os.path.join(iter_dir, "removed_indices.npy"), removed_indices)
        np.save(os.path.join(iter_dir, "duplicated_hard_indices.npy"), duplicate_indices)
        np.save(os.path.join(iter_dir, "output_indices.npy"), output_indices)
        if args.save_per_sample_arrays:
            np.save(os.path.join(iter_dir, "train_style_weight.npy"), weights.astype(np.float32))
            for key, value in recon_metrics.items():
                np.save(os.path.join(iter_dir, f"recon_{key}.npy"), np.asarray(value, dtype=np.float32))

        next_trajs = current_trajs[output_indices]
        next_source_indices = current_source_indices[output_indices]
        next_path, next_source_path = save_dataset_with_source(
            next_trajs,
            next_source_indices,
            os.path.join(iter_dir, "next_dataset.npy"),
        )
        final_dataset_path = next_path
        final_source_path = next_source_path

        iter_summary = {
            "iteration": int(iteration),
            "input_count": int(current_trajs.shape[0]),
            "output_count": int(next_trajs.shape[0]),
            "model_path": os.path.abspath(model_path),
            "input_dataset_path": os.path.abspath(iter_input_path),
            "input_source_indices_path": os.path.abspath(iter_input_source_path),
            "next_dataset_path": os.path.abspath(next_path),
            "next_source_indices_path": os.path.abspath(next_source_path),
            "eval_summary": eval_summary,
            "max_error_visualization": max_error_visualization,
            "tracked_max_error_visualization": tracked_max_error_visualization,
            "target_failures": failures,
            "grouping": grouping_meta,
            "global_thresholds": global_thresholds,
            "num_removed": int(removed_indices.size),
            "num_group_relative_duplicate_unique": int(group_duplicate_indices.size),
            "num_global_fallback_duplicate_unique": int(global_duplicate_indices.size),
            "num_duplicate_unique_after_cap": int(duplicate_indices.size),
            "duplicate_hard_count": int(args.duplicate_hard_count),
            "num_appended_rows": int(append_indices.size),
            "removed_csv": os.path.abspath(removed_csv),
            "group_duplicate_csv": os.path.abspath(group_dup_csv),
            "global_duplicate_csv": os.path.abspath(global_dup_csv),
        }
        write_json(os.path.join(iter_dir, "iteration_summary.json"), iter_summary)
        history.append(iter_summary)

        print(
            f"[augment] removed={removed_indices.size}, "
            f"dup_unique={duplicate_indices.size}, appended={append_indices.size}, "
            f"next_count={next_trajs.shape[0]}"
        )

        if output_indices.shape[0] == current_trajs.shape[0] and np.all(output_indices == np.arange(current_trajs.shape[0])):
            stop_reason = "no_dataset_change"
            print("[stop] no samples removed or duplicated; stopping to avoid a no-op loop.")
            break

        current_trajs = next_trajs
        current_source_indices = next_source_indices.astype(np.int64)

    summary = {
        "stop_reason": stop_reason,
        "output_root": output_root,
        "config_path": os.path.abspath(config_path),
        "resume": resume_info,
        "start_iteration": int(start_iteration),
        "max_iters": int(args.max_iters),
        "final_dataset_path": os.path.abspath(final_dataset_path) if final_dataset_path else "",
        "final_source_indices_path": os.path.abspath(final_source_path) if final_source_path else "",
        "history": history,
    }
    summary_path = os.path.join(output_root, "iterative_worst_case_summary.json")
    write_json(summary_path, summary)

    print("=" * 80)
    print("Iterative Worst-Case Experiment Done")
    print(f"stop_reason: {stop_reason}")
    print(f"summary_path: {summary_path}")
    if final_dataset_path:
        print(f"final_dataset_path: {final_dataset_path}")
        print(f"final_source_indices_path: {final_source_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
