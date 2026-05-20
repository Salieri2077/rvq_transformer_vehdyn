import argparse
import os
import sys

import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# 复用 analyze_group_worst_case.py 已经写好的 cache 读取、device 和训练式 loss 计算。
from analyze_group_worst_case import (  # noqa: E402
    compute_train_style_loss_components,
    integrate_to_global_torch,
    load_grouping_cache,
    resolve_device,
)
from eval_tokenizer_by_scenario import build_model, infer_model_type  # noqa: E402
from utils import load_norm_params_torch, normalize_trajs_torch, write_csv, write_json  # noqa: E402


DEFAULT_DATA_PATH = "/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove samples whose train-style weight loss is abnormal within their group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", "--model-path", required=True)
    parser.add_argument("--norm_path", "--norm-path", required=True)
    parser.add_argument("--data_path", "--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--group_cache_dir", "--group-cache-dir", required=True)
    parser.add_argument("--group_cache_key", "--group-cache-key", default="")
    parser.add_argument("--out_path", "--out-path", default="")
    parser.add_argument("--out_dir", "--out-dir", default="")

    parser.add_argument("--model_type", "--model-type", default="auto", choices=["auto", "taae"])
    parser.add_argument("--data_type", "--data-type", default="pred", choices=["pred", "history"])
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4096)
    parser.add_argument("--num_layers", "--num-layers", type=int, default=0)
    parser.add_argument("--num_transformer_layers", "--num-transformer-layers", type=int, default=2)
    parser.add_argument("--loss_epoch", "--loss-epoch", type=int, default=31)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # 大组沿用 median + IQR；小组用 max/min ratio + 全局分位阈值，避免小组被直接跳过。
    parser.add_argument("--min_group_size", "--min-group-size", type=int, default=3)
    parser.add_argument("--large_group_min_size", "--large-group-min-size", type=int, default=4)
    parser.add_argument("--ratio_threshold", "--ratio-threshold", type=float, default=10.0)
    parser.add_argument("--iqr_mult", "--iqr-mult", type=float, default=3.0)
    parser.add_argument("--hard_iqr_mult", "--hard-iqr-mult", type=float, default=1.0)
    parser.add_argument("--small_group_hard_ratio", "--small-group-hard-ratio", type=float, default=5.0)
    parser.add_argument("--small_group_remove_ratio", "--small-group-remove-ratio", type=float, default=12.0)
    parser.add_argument("--global_hard_percentile", "--global-hard-percentile", type=float, default=80.0)
    parser.add_argument("--global_remove_percentile", "--global-remove-percentile", type=float, default=99.5)
    parser.add_argument("--duplicate_hard_count", "--duplicate-hard-count", type=int, default=5)
    parser.add_argument("--inspect_sample_idx", "--inspect-sample-idx", type=int, default=-1)
    return parser.parse_args()


def load_original_data(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    trajs = data["trajs"] if isinstance(data, dict) and "trajs" in data else data
    return data, np.asarray(trajs, dtype=np.float32)


def default_out_path(data_path):
    root, ext = os.path.splitext(data_path)
    return f"{root}_group_loss_filtered{ext or '.npy'}"


def source_indices_sidecar_path(out_path):
    root, _ = os.path.splitext(out_path)
    return f"{root}_source_indices.npy"


def save_by_indices(data, indices, out_path, n_total):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if isinstance(data, dict) and "trajs" in data:
        filtered = dict(data)
        # 如果 dict 里还有 scenario/meta 等同长度数组，一起过滤，避免错位。
        for key, value in data.items():
            arr = np.asarray(value)
            if arr.shape[:1] == (int(n_total),):
                filtered[key] = arr[indices]
        np.save(out_path, filtered)
    else:
        np.save(out_path, np.asarray(data)[indices])


def normalize_csv_rows(rows):
    """small/large group 行字段不同；写 CSV 前补齐字段并集，保持输出稳定。"""
    if not rows:
        return rows
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def compute_ade_fde(model, trajs, norm_params, batch_size):
    """按 batch 计算 ADE/FDE，避免一次性保存全量 reconstruction。"""
    ades = []
    fdes = []
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
            x_recon, _, _, _, _ = model(x_norm)
            pred_phys = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
            gt_phys = (x_norm * model.norm_scale * model.norm_std) + model.norm_mean

            pred_xy = integrate_to_global_torch(pred_phys)
            gt_xy = integrate_to_global_torch(gt_phys)
            step_error = torch.sqrt(torch.sum((pred_xy - gt_xy) ** 2, dim=-1) + 1e-6)
            ades.append(step_error.mean(dim=1).detach().cpu().numpy())
            fdes.append(step_error[:, -1].detach().cpu().numpy())
    return np.concatenate(ades).astype(np.float64), np.concatenate(fdes).astype(np.float64)


def _iqr_hard_threshold(values, members, hard_iqr_mult):
    group_v = values[members]
    q25, q75 = np.percentile(group_v, [25, 75])
    iqr = float(q75 - q25)
    return float(q75) + float(hard_iqr_mult) * max(iqr, 1e-8)


def handle_small_group(
    group_id,
    members,
    weights,
    ades,
    fdes,
    remove_mask,
    remove_rows,
    duplicate_indices,
    duplicate_rows,
    duplicate_set,
    small_group_hard_ratio,
    small_group_remove_ratio,
    global_thresholds,
):
    """
    小组不能稳定估计 IQR，因此只做相对组内 min 的 ratio 判断，
    再叠加全局分位阈值，避免复制绝对误差很小的样本。
    """
    members = np.asarray(members, dtype=np.int64)
    if members.size <= 1:
        return

    min_weight = float(np.min(weights[members]))
    min_ade = float(np.min(ades[members]))
    min_fde = float(np.min(fdes[members]))

    for sample_idx in members.tolist():
        sample_idx = int(sample_idx)
        weight_ratio = float(weights[sample_idx] / max(min_weight, 1e-8))
        ade_ratio = float(ades[sample_idx] / max(min_ade, 1e-8))
        fde_ratio = float(fdes[sample_idx] / max(min_fde, 1e-8))

        base_row = {
            "group_id": int(group_id),
            "sample_idx": sample_idx,
            "weight": float(weights[sample_idx]),
            "ade": float(ades[sample_idx]),
            "fde": float(fdes[sample_idx]),
            "group_size": int(members.size),
            "min_group_weight": min_weight,
            "min_group_ade": min_ade,
            "min_group_fde": min_fde,
            "weight_ratio_to_min": weight_ratio,
            "ade_ratio_to_min": ade_ratio,
            "fde_ratio_to_min": fde_ratio,
            "global_hard_weight": float(global_thresholds["hard_weight"]),
            "global_hard_ade": float(global_thresholds["hard_ade"]),
            "global_hard_fde": float(global_thresholds["hard_fde"]),
            "global_remove_weight": float(global_thresholds["remove_weight"]),
        }

        # 小组删除非常保守：只看 weight，避免 ADE/FDE 单项偏高导致误删。
        should_remove = (
            weight_ratio >= float(small_group_remove_ratio)
            and weights[sample_idx] >= float(global_thresholds["remove_weight"])
        )
        if should_remove:
            remove_mask[sample_idx] = True
            row = dict(base_row)
            row["remove_reason"] = "small_group_weight_extreme"
            remove_rows.append(row)
            continue

        if remove_mask[sample_idx] or sample_idx in duplicate_set:
            continue

        hard_reasons = []
        if (
            weight_ratio >= float(small_group_hard_ratio)
            and weights[sample_idx] >= float(global_thresholds["hard_weight"])
        ):
            hard_reasons.append("small_group_weight")
        if (
            ade_ratio >= float(small_group_hard_ratio)
            and ades[sample_idx] >= float(global_thresholds["hard_ade"])
        ):
            hard_reasons.append("small_group_ade")
        if (
            fde_ratio >= float(small_group_hard_ratio)
            and fdes[sample_idx] >= float(global_thresholds["hard_fde"])
        ):
            hard_reasons.append("small_group_fde")

        if hard_reasons:
            duplicate_set.add(sample_idx)
            duplicate_indices.append(sample_idx)
            row = dict(base_row)
            row["hard_reason"] = ",".join(hard_reasons)
            duplicate_rows.append(row)


def find_group_outliers(
    weights,
    ades,
    fdes,
    group_to_indices,
    large_group_min_size,
    ratio_threshold,
    iqr_mult,
    hard_iqr_mult,
    small_group_hard_ratio,
    small_group_remove_ratio,
    global_thresholds,
):
    remove_mask = np.zeros(weights.shape[0], dtype=bool)
    remove_rows = []
    duplicate_indices = []
    duplicate_rows = []
    duplicate_set = set()

    for group_id, members in enumerate(group_to_indices):
        members = np.asarray(members, dtype=np.int64)
        if members.size <= 1:
            continue

        if members.size < int(large_group_min_size):
            handle_small_group(
                group_id=group_id,
                members=members,
                weights=weights,
                ades=ades,
                fdes=fdes,
                remove_mask=remove_mask,
                remove_rows=remove_rows,
                duplicate_indices=duplicate_indices,
                duplicate_rows=duplicate_rows,
                duplicate_set=duplicate_set,
                small_group_hard_ratio=small_group_hard_ratio,
                small_group_remove_ratio=small_group_remove_ratio,
                global_thresholds=global_thresholds,
            )
            continue

        group_w = weights[members]
        median = float(np.median(group_w))
        q25, q75 = np.percentile(group_w, [25, 75])
        iqr = float(q75 - q25)

        # 大组保留原来的策略：同时看“相对 median 的倍数”和 IQR outlier。
        ratio_threshold_value = max(median, 1e-8) * float(ratio_threshold)
        iqr_threshold_value = float(q75) + float(iqr_mult) * max(iqr, 1e-8)
        remove_threshold = max(ratio_threshold_value, iqr_threshold_value)
        bad_local = np.where(group_w > remove_threshold)[0]
        for local_pos in bad_local.tolist():
            sample_idx = int(members[local_pos])
            remove_mask[sample_idx] = True
            remove_rows.append(
                {
                    "group_id": int(group_id),
                    "sample_idx": sample_idx,
                    "weight": float(weights[sample_idx]),
                    "ade": float(ades[sample_idx]),
                    "fde": float(fdes[sample_idx]),
                    "group_size": int(members.size),
                    "group_median_weight": median,
                    "group_q75_weight": float(q75),
                    "group_iqr_weight": iqr,
                    "remove_threshold": float(remove_threshold),
                    "ratio_to_median": float(weights[sample_idx] / max(median, 1e-8)),
                    "remove_reason": "large_group_weight_extreme",
                }
            )

        # hard 样本：不是极端坏点，但 weight/ADE/FDE 任一指标在组内偏高。
        weight_hard_threshold = _iqr_hard_threshold(weights, members, hard_iqr_mult)
        ade_hard_threshold = _iqr_hard_threshold(ades, members, hard_iqr_mult)
        fde_hard_threshold = _iqr_hard_threshold(fdes, members, hard_iqr_mult)
        hard_mask = (
            (~remove_mask[members])
            & (
                (weights[members] > weight_hard_threshold)
                | (ades[members] > ade_hard_threshold)
                | (fdes[members] > fde_hard_threshold)
            )
        )
        hard_local = np.where(hard_mask)[0]
        for local_pos in hard_local.tolist():
            sample_idx = int(members[local_pos])
            if remove_mask[sample_idx] or sample_idx in duplicate_set:
                continue
            duplicate_set.add(sample_idx)
            duplicate_indices.append(sample_idx)
            duplicate_rows.append(
                {
                    "group_id": int(group_id),
                    "sample_idx": sample_idx,
                    "weight": float(weights[sample_idx]),
                    "ade": float(ades[sample_idx]),
                    "fde": float(fdes[sample_idx]),
                    "group_size": int(members.size),
                    "group_median_weight": median,
                    "group_q75_weight": float(q75),
                    "group_iqr_weight": iqr,
                    "weight_hard_threshold": float(weight_hard_threshold),
                    "ade_hard_threshold": float(ade_hard_threshold),
                    "fde_hard_threshold": float(fde_hard_threshold),
                    "remove_threshold": float(remove_threshold),
                    "ratio_to_median": float(weights[sample_idx] / max(median, 1e-8)),
                    "hard_reason": ",".join(
                        [
                            name
                            for name, ok in [
                                ("weight", weights[sample_idx] > weight_hard_threshold),
                                ("ade", ades[sample_idx] > ade_hard_threshold),
                                ("fde", fdes[sample_idx] > fde_hard_threshold),
                            ]
                            if ok
                        ]
                    ),
                }
            )

    return remove_mask, remove_rows, np.asarray(duplicate_indices, dtype=np.int64), duplicate_rows

def main():
    args = parse_args()
    out_path = args.out_path or default_out_path(args.data_path)
    out_dir = args.out_dir or os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    data, trajs = load_original_data(args.data_path)
    model_trajs = trajs[:, :14, :] if args.data_type == "history" else trajs
    n_total = int(model_trajs.shape[0])

    cache_dir, grouping = load_grouping_cache(
        group_cache_dir=args.group_cache_dir,
        group_cache_key=args.group_cache_key,
        n_total=n_total,
    )

    device = resolve_device(args.device)
    model_type = infer_model_type(args.model_path, args.model_type)
    if model_type != "taae":
        raise ValueError("This filter script currently supports only TAAE weight loss.")

    model = build_model(
        model_path=args.model_path,
        input_steps=int(model_trajs.shape[1]),
        device=device,
        model_type=model_type,
        num_transformer_layers=int(args.num_transformer_layers),
        num_layers=int(args.num_layers),
    )
    norm_params = load_norm_params_torch(args.norm_path, device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])

    print(f"Loaded data: {args.data_path} | shape={trajs.shape}")
    print(f"Loaded grouping cache: {cache_dir}")
    print("Computing train-style per-sample weight loss. This may take a while for the full dataset...")
    losses = compute_train_style_loss_components(
        model=model,
        trajs=model_trajs,
        norm_params=norm_params,
        batch_size=int(args.batch_size),
        loss_epoch=int(args.loss_epoch),
    )
    weights = np.asarray(losses["weight"], dtype=np.float64)

    print("Computing per-sample ADE/FDE for hard-sample duplication...")
    ades, fdes = compute_ade_fde(
        model=model,
        trajs=model_trajs,
        norm_params=norm_params,
        batch_size=int(args.batch_size),
    )

    # 小组 ratio 判断需要绝对误差门槛，避免复制“只是相对差、但绝对误差很小”的样本。
    global_thresholds = {
        "hard_weight": float(np.percentile(weights, float(args.global_hard_percentile))),
        "hard_ade": float(np.percentile(ades, float(args.global_hard_percentile))),
        "hard_fde": float(np.percentile(fdes, float(args.global_hard_percentile))),
        "remove_weight": float(np.percentile(weights, float(args.global_remove_percentile))),
    }
    large_group_min_size = max(int(args.large_group_min_size), int(args.min_group_size))

    remove_mask, removed_rows, duplicate_indices, duplicate_rows = find_group_outliers(
        weights=weights,
        ades=ades,
        fdes=fdes,
        group_to_indices=grouping["group_to_indices"],
        large_group_min_size=large_group_min_size,
        ratio_threshold=float(args.ratio_threshold),
        iqr_mult=float(args.iqr_mult),
        hard_iqr_mult=float(args.hard_iqr_mult),
        small_group_hard_ratio=float(args.small_group_hard_ratio),
        small_group_remove_ratio=float(args.small_group_remove_ratio),
        global_thresholds=global_thresholds,
    )
    keep_mask = ~remove_mask

    kept_indices = np.where(keep_mask)[0].astype(np.int64)
    removed_indices = np.where(remove_mask)[0].astype(np.int64)
    duplicate_count = max(0, int(args.duplicate_hard_count))
    append_indices = np.repeat(duplicate_indices, duplicate_count).astype(np.int64)
    output_indices = np.concatenate([kept_indices, append_indices], axis=0)

    save_by_indices(data, output_indices, out_path, n_total=n_total)
    source_indices_path = source_indices_sidecar_path(out_path)
    # source_indices_path 是稳定映射：filtered_data[new_row] 来自原始数据的哪一行。
    np.save(source_indices_path, output_indices.astype(np.int64))

    np.save(os.path.join(out_dir, "kept_indices.npy"), kept_indices)
    np.save(os.path.join(out_dir, "removed_indices.npy"), removed_indices)
    np.save(os.path.join(out_dir, "duplicated_hard_indices.npy"), duplicate_indices)
    np.save(os.path.join(out_dir, "output_indices.npy"), output_indices)
    np.save(os.path.join(out_dir, "train_style_weight.npy"), weights.astype(np.float32))
    np.save(os.path.join(out_dir, "recon_ade.npy"), ades.astype(np.float32))
    np.save(os.path.join(out_dir, "recon_fde.npy"), fdes.astype(np.float32))

    removed_csv = os.path.join(out_dir, "removed_group_loss_outliers.csv")
    duplicated_csv = os.path.join(out_dir, "duplicated_hard_group_loss_samples.csv")
    write_csv(normalize_csv_rows(removed_rows), removed_csv)
    write_csv(normalize_csv_rows(duplicate_rows), duplicated_csv)

    inspected = None
    if int(args.inspect_sample_idx) >= 0:
        idx = int(args.inspect_sample_idx)
        inspected = {
            "sample_idx": idx,
            "in_range": bool(0 <= idx < n_total),
            "removed": bool(remove_mask[idx]) if 0 <= idx < n_total else False,
            "duplicated": bool(np.any(duplicate_indices == idx)) if 0 <= idx < n_total else False,
            "weight": float(weights[idx]) if 0 <= idx < n_total else None,
            "ade": float(ades[idx]) if 0 <= idx < n_total else None,
            "fde": float(fdes[idx]) if 0 <= idx < n_total else None,
            "group_id": int(grouping["group_id_per_sample"][idx]) if 0 <= idx < n_total else None,
        }
        print(f"Inspect sample: {inspected}")

    summary = {
        "data_path": os.path.abspath(args.data_path),
        "out_path": os.path.abspath(out_path),
        "group_cache_dir": cache_dir,
        "model_path": os.path.abspath(args.model_path),
        "norm_path": os.path.abspath(args.norm_path),
        "num_total": n_total,
        "num_removed": int(removed_indices.size),
        "num_kept": int(kept_indices.size),
        "num_hard_samples_duplicated": int(duplicate_indices.size),
        "duplicate_hard_count": duplicate_count,
        "num_output": int(output_indices.size),
        "source_indices_path": os.path.abspath(source_indices_path),
        "removed_ratio": float(removed_indices.size / max(n_total, 1)),
        "min_group_size": int(args.min_group_size),
        "large_group_min_size": int(large_group_min_size),
        "small_group_hard_ratio": float(args.small_group_hard_ratio),
        "small_group_remove_ratio": float(args.small_group_remove_ratio),
        "global_hard_percentile": float(args.global_hard_percentile),
        "global_remove_percentile": float(args.global_remove_percentile),
        "global_thresholds": global_thresholds,
        "ratio_threshold": float(args.ratio_threshold),
        "iqr_mult": float(args.iqr_mult),
        "hard_iqr_mult": float(args.hard_iqr_mult),
        "loss_epoch": int(args.loss_epoch),
        "removed_csv": os.path.abspath(removed_csv),
        "duplicated_csv": os.path.abspath(duplicated_csv),
        "inspect_sample": inspected,
    }
    summary_path = os.path.join(out_dir, "filter_group_loss_outliers_summary.json")
    write_json(summary_path, summary)

    print("=" * 80)
    print(f"Saved filtered data to: {out_path}")
    print(f"Removed samples: {removed_indices.size} / {n_total} ({summary['removed_ratio']:.6f})")
    print(f"Duplicated hard samples: {duplicate_indices.size} unique x {duplicate_count} extra copies")
    print(f"Output samples: {output_indices.size}")
    print(f"Saved source-index sidecar to: {source_indices_path}")
    print(f"Saved removed detail csv to: {removed_csv}")
    print(f"Saved duplicated detail csv to: {duplicated_csv}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()


# Example:
# python filter_group_loss_outliers.py \
#   --model_path /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_kin_0311/pred_rvq_taae_model.pth \
#   --norm_path /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/rvq_tfm_kin_0311/pred_norm_params.pkl \
#   --data_path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy \
#   --group_cache_dir /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/similar_single_train/grouping_cache \
#   --out_path /home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120_group_loss_filtered.npy \
#   --inspect_sample_idx 1078274
