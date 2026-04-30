import argparse
import json
import os
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

try:
    from train_tfm import TrajRVQTransformer
except ImportError:
    from rvq_transformer_vehdyn.train_tfm import TrajRVQTransformer

try:
    from analyze_token_meaning import (
        analyze_scenario_token_controls,
        build_scenario_masks,
        encode_codes,
        load_norm_params,
        load_trajs,
        select_scenario_base_indices,
        write_csv,
    )
except ImportError:
    from rvq_transformer_vehdyn.analyze_token_meaning import (
        analyze_scenario_token_controls,
        build_scenario_masks,
        encode_codes,
        load_norm_params,
        load_trajs,
        select_scenario_base_indices,
        write_csv,
    )

DEFAULT_AUGMENTED_DATA_PATH = (
    "/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy"
)


def parse_token_counts(spec: str) -> List[int]:
    values = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v <= 0:
            continue
        values.append(v)
    out = sorted(set(values), reverse=True)
    if not out:
        raise ValueError("No valid token count found in --token-counts")
    return out


def parse_model_path_map(spec: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --model-path-map item: {part}. Expect K:/path/to/model.pth")
        k_str, path = part.split(":", 1)
        k = int(k_str.strip())
        path = path.strip()
        if not path:
            raise ValueError(f"Empty model path for K={k}")
        out[k] = path
    return out


def build_model_for_k(model_path: str, input_steps: int, num_layers: int, device: torch.device) -> TrajRVQTransformer:
    model = TrajRVQTransformer(
        input_steps=input_steps,
        input_dim=3,
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model


def compute_mode_codes_per_layer(codes: np.ndarray, vocab_size: int) -> np.ndarray:
    out = np.zeros(codes.shape[1], dtype=np.int64)
    for layer in range(codes.shape[1]):
        counts = np.bincount(codes[:, layer], minlength=vocab_size)
        out[layer] = int(np.argmax(counts))
    return out


def reduce_codes(codes: np.ndarray, keep_layers: int, fill_codes: np.ndarray) -> np.ndarray:
    out = np.asarray(codes, dtype=np.int64).copy()
    if keep_layers < out.shape[1]:
        out[:, keep_layers:] = fill_codes[None, keep_layers:]
    return out


def compute_active_utilization_pct(codes: np.ndarray, keep_layers: int, vocab_size: int) -> float:
    vals = []
    for layer in range(keep_layers):
        used = int(np.unique(codes[:, layer]).size)
        vals.append(used / max(vocab_size, 1) * 100.0)
    return float(np.mean(vals)) if vals else 0.0


def global_codebook_coverage(codes: np.ndarray, keep_layers: int, vocab_size: int) -> Dict[str, float]:
    active_codes = codes[:, :keep_layers].reshape(-1)
    used = np.unique(active_codes)
    used_count = int(len(used))
    return {
        "active_unique_codes": used_count,
        "global_coverage_pct": float(used_count / max(vocab_size, 1) * 100.0),
    }


def scenario_resolution_summary(
    codes: np.ndarray,
    categories: Dict[str, np.ndarray],
    keep_layers: int,
) -> Dict[str, float]:
    prototypes: Dict[str, np.ndarray] = {}
    intra_scores: List[float] = []

    for scenario_name, mask in categories.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        c = codes[idxs, :keep_layers]
        proto = np.zeros(keep_layers, dtype=np.int64)
        for layer in range(keep_layers):
            vals, cnts = np.unique(c[:, layer], return_counts=True)
            proto[layer] = int(vals[np.argmax(cnts)])
        sample_match = (c == proto[None, :]).mean(axis=1)
        intra = float(sample_match.mean())
        intra_scores.append(intra)
        prototypes[scenario_name] = proto

    pair_sims = []
    names = sorted(prototypes.keys())
    for a, b in combinations(names, 2):
        pair_sims.append(float((prototypes[a] == prototypes[b]).mean()))

    intra_mean = float(np.mean(intra_scores)) if intra_scores else 0.0
    inter_separability = float(1.0 - np.mean(pair_sims)) if pair_sims else 0.0
    resolution_score = float(intra_mean * inter_separability)

    return {
        "num_scenarios_used": int(len(intra_scores)),
        "scenario_intra_consistency_mean": intra_mean,
        "scenario_inter_separability": inter_separability,
        "scenario_resolution_score": resolution_score,
    }


def run_control_semantics(
    model: TrajRVQTransformer,
    scenario_base_indices: Dict[str, int],
    codes: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    dt: float,
    batch_size: int,
    n_each_side: int,
    keep_layers: int,
) -> Tuple[List[Dict], List[Dict]]:
    score_rows: List[Dict] = []
    top_rows: List[Dict] = []
    layers = list(range(keep_layers))

    for scenario_name, scenario_base_idx in scenario_base_indices.items():
        s_rows, t_rows, _ = analyze_scenario_token_controls(
            model=model,
            scenario_name=scenario_name,
            base_idx=scenario_base_idx,
            base_codes=codes[scenario_base_idx],
            codes=codes,
            norm_params=norm_params,
            dt=dt,
            batch_size=batch_size,
            n_each_side=n_each_side,
            layers=layers,
        )
        score_rows.extend(s_rows)
        top_rows.extend(t_rows)
    return score_rows, top_rows


def _build_layer_quantity_mean_rows(score_rows: List[Dict], keep_layers: int) -> Tuple[List[str], List[Dict]]:
    quantities = sorted(set(r["quantity_name"] for r in score_rows))
    out_rows: List[Dict] = []
    for layer in range(keep_layers):
        for q in quantities:
            vals = [r["control_score"] for r in score_rows if int(r["layer"]) == layer and r["quantity_name"] == q]
            out_rows.append(
                {
                    "layer": int(layer),
                    "quantity_name": q,
                    "mean_control_score": float(np.mean(vals)) if vals else 0.0,
                }
            )
    return quantities, out_rows


def plot_control_heatmap_compare(
    score_rows_by_k: Dict[int, List[Dict]],
    token_counts: List[int],
    out_path: str,
):
    if not score_rows_by_k:
        return

    quantities = sorted(set(r["quantity_name"] for rows in score_rows_by_k.values() for r in rows))
    max_layers = max(token_counts)

    mats: Dict[int, np.ndarray] = {}
    vmax = 0.0
    for k in token_counts:
        mat = np.full((max_layers, len(quantities)), np.nan, dtype=np.float32)
        rows = score_rows_by_k.get(k, [])
        for layer in range(k):
            for j, q in enumerate(quantities):
                vals = [r["control_score"] for r in rows if int(r["layer"]) == layer and r["quantity_name"] == q]
                if vals:
                    mat[layer, j] = float(np.mean(vals))
        mats[k] = mat
        if np.any(np.isfinite(mat)):
            vmax = max(vmax, float(np.nanmax(mat)))

    vmax = max(vmax, 1e-6)
    n = len(token_counts)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(5.4 * cols, 4.2 * rows_n),
        squeeze=False,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#e6e6e6")

    for i, k in enumerate(token_counts):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        im = ax.imshow(mats[k], aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(f"K={k}")
        ax.set_xticks(np.arange(len(quantities)))
        ax.set_xticklabels(quantities, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(max_layers))
        ax.set_yticklabels([f"L{l:02d}" for l in range(max_layers)], fontsize=8)
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Layer")

    for j in range(n, rows_n * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Mean control_score")
    fig.suptitle("Layer x Quantity Mean Control Score (Compare K)", fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top1_map_compare(
    top_rows_by_k: Dict[int, List[Dict]],
    token_counts: List[int],
    out_path: str,
):
    if not top_rows_by_k:
        return

    quantities = sorted(
        set(r["top1_quantity"] for rows in top_rows_by_k.values() for r in rows if str(r["top1_quantity"]).strip())
    )
    scenarios = sorted(set(r["scenario"] for rows in top_rows_by_k.values() for r in rows))
    q_to_id = {q: i for i, q in enumerate(quantities)}
    max_layers = max(token_counts)

    mats: Dict[int, np.ndarray] = {}
    for k in token_counts:
        mat = np.full((len(scenarios), max_layers), -1, dtype=np.int64)
        rows = top_rows_by_k.get(k, [])
        for row in rows:
            s = row["scenario"]
            l = int(row["layer"])
            q = row.get("top1_quantity", "")
            if not q or q not in q_to_id:
                continue
            si = scenarios.index(s)
            if l < max_layers:
                mat[si, l] = q_to_id[q]
        mats[k] = mat

    n = len(token_counts)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.8 * cols, max(4.0, 0.45 * len(scenarios)) * rows_n), squeeze=False)

    base = plt.get_cmap("tab20", max(len(quantities), 1))
    colors = ["#cfcfcf"] + [mcolors.to_hex(base(i)) for i in range(max(len(quantities), 1))]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(quantities) + 1.5, 1), cmap.N)

    for i, k in enumerate(token_counts):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        mat = mats[k] + 1
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"K={k}")
        ax.set_xticks(np.arange(max_layers))
        ax.set_xticklabels([f"L{l:02d}" for l in range(max_layers)], fontsize=8)
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Scenario")

    for j in range(n, rows_n * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    legend_handles = [Patch(facecolor=colors[0], edgecolor="black", label="-1 unused")]
    for i, q in enumerate(quantities):
        legend_handles.append(Patch(facecolor=colors[i + 1], edgecolor="black", label=f"{i} {q}"))

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        title="Legend",
        fontsize=9,
        title_fontsize=10,
    )
    fig.suptitle("Scenario x Layer Top1 Controlled Quantity (Compare K)", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 0.92])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def resolve_model_path_for_k(
    k: int,
    args: argparse.Namespace,
    model_path_map: Dict[int, str],
    fallback_k: int,
) -> str:
    if k in model_path_map:
        return model_path_map[k]

    if args.model_path_template:
        p = args.model_path_template.format(k=k, data_type=args.data_type)
        if os.path.exists(p):
            return p

    # fallback: only for designated fallback_k (typically max K)
    if k == fallback_k:
        fallback = os.path.join(args.save_dir, f"{args.data_type}_rvq_taae_model.pth")
        if os.path.exists(fallback):
            return fallback

    raise FileNotFoundError(
        f"No model path found for K={k}. Use --model-path-map or --model-path-template."
    )


def main():
    parser = argparse.ArgumentParser(description="Token reduction experiment (focus on utilization/coverage/resolution/control_score semantics).")
    parser.add_argument("--data-path", type=str, default=DEFAULT_AUGMENTED_DATA_PATH)
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_kin_0311")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--token-counts", type=str, default="15,12,9,6,3")
    parser.add_argument("--diagnostic-neighbor-each-side", type=int, default=3)

    parser.add_argument("--reduction-mode", type=str, default="model", choices=["model", "truncate"])
    parser.add_argument(
        "--model-path-map",
        type=str,
        default="",
        help="Format: '15:/path/a.pth,12:/path/b.pth,...'",
    )
    parser.add_argument(
        "--model-path-template",
        type=str,
        default="",
        help="Template with {k} and {data_type}, e.g. ./work_dirs/tokenizer/rvq_tfm_kin_k{k}/{data_type}_rvq_taae_model.pth",
    )
    parser.add_argument("--fill-strategy", type=str, default="mode", choices=["mode", "zero"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(args.save_dir, "token_reduction_exp")
    os.makedirs(output_dir, exist_ok=True)

    trajs = load_trajs(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    if args.max_samples > 0:
        trajs = trajs[: args.max_samples]

    token_counts = parse_token_counts(args.token_counts)
    model_path_map = parse_model_path_map(args.model_path_map)

    # 默认使用 save_dir 的归一化参数（通常不同 K 可共用）
    norm_path = os.path.join(args.save_dir, f"{args.data_type}_norm_params.pkl")
    norm_params = load_norm_params(norm_path, device)

    categories, scenario_features = build_scenario_masks(trajs, dt=args.dt)
    scenario_base_indices = select_scenario_base_indices(categories, scenario_features)

    overview_rows: List[Dict] = []
    heatmap_rows_all: List[Dict] = []
    score_rows_by_k: Dict[int, List[Dict]] = {}
    top_rows_by_k: Dict[int, List[Dict]] = {}

    max_k = max(token_counts)

    if args.reduction_mode == "model":
        for k in token_counts:
            model_path_k = resolve_model_path_for_k(k, args, model_path_map, fallback_k=max_k)
            model_k = build_model_for_k(model_path_k, input_steps=trajs.shape[1], num_layers=k, device=device)
            model_k.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])

            print(f"Encoding {len(trajs)} trajectories with K={k} ...")
            codes_k = encode_codes(model_k, trajs, norm_params, args.batch_size)

            util_active = compute_active_utilization_pct(codes_k, keep_layers=k, vocab_size=model_k.vocab_size)
            coverage = global_codebook_coverage(codes_k, keep_layers=k, vocab_size=model_k.vocab_size)
            resolution = scenario_resolution_summary(codes_k, categories, keep_layers=k)

            score_rows, top_rows = run_control_semantics(
                model=model_k,
                scenario_base_indices=scenario_base_indices,
                codes=codes_k,
                norm_params=norm_params,
                dt=args.dt,
                batch_size=args.batch_size,
                n_each_side=args.diagnostic_neighbor_each_side,
                keep_layers=k,
            )
            score_rows_by_k[k] = score_rows
            top_rows_by_k[k] = top_rows

            _, lq_rows = _build_layer_quantity_mean_rows(score_rows, keep_layers=k)
            for r in lq_rows:
                heatmap_rows_all.append({"keep_layers": int(k), **r})

            control_mean = float(np.mean([r["control_score"] for r in score_rows])) if score_rows else 0.0

            overview_rows.append(
                {
                    "keep_layers": int(k),
                    "reduction_mode": args.reduction_mode,
                    "num_samples": int(len(trajs)),
                    "utilization_active_pct": util_active,
                    "coverage_pct": coverage["global_coverage_pct"],
                    "active_unique_codes": coverage["active_unique_codes"],
                    "resolution_score": resolution["scenario_resolution_score"],
                    "resolution_intra_mean": resolution["scenario_intra_consistency_mean"],
                    "resolution_inter_sep": resolution["scenario_inter_separability"],
                    "control_score_mean": control_mean,
                }
            )
            print(
                f"[K={k:02d}] util={util_active:.2f}% coverage={coverage['global_coverage_pct']:.2f}% "
                f"resolution={resolution['scenario_resolution_score']:.4f} control_mean={control_mean:.4f}"
            )

    else:
        # truncate mode: 只作为可选对照，不等价于真正减少RVQ层
        k_full = max(token_counts)
        model_path_full = resolve_model_path_for_k(k_full, args, model_path_map, fallback_k=k_full)
        model_full = build_model_for_k(model_path_full, input_steps=trajs.shape[1], num_layers=k_full, device=device)
        model_full.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])

        print(f"Encoding {len(trajs)} trajectories with full K={k_full} for truncation ...")
        full_codes = encode_codes(model_full, trajs, norm_params, args.batch_size)
        if args.fill_strategy == "mode":
            fill_codes = compute_mode_codes_per_layer(full_codes, model_full.vocab_size)
        else:
            fill_codes = np.zeros(model_full.num_layers, dtype=np.int64)

        for k in token_counts:
            codes_k = reduce_codes(full_codes, keep_layers=k, fill_codes=fill_codes)

            util_active = compute_active_utilization_pct(codes_k, keep_layers=k, vocab_size=model_full.vocab_size)
            coverage = global_codebook_coverage(codes_k, keep_layers=k, vocab_size=model_full.vocab_size)
            resolution = scenario_resolution_summary(codes_k, categories, keep_layers=k)

            score_rows, top_rows = run_control_semantics(
                model=model_full,
                scenario_base_indices=scenario_base_indices,
                codes=codes_k,
                norm_params=norm_params,
                dt=args.dt,
                batch_size=args.batch_size,
                n_each_side=args.diagnostic_neighbor_each_side,
                keep_layers=k,
            )
            score_rows_by_k[k] = score_rows
            top_rows_by_k[k] = top_rows

            _, lq_rows = _build_layer_quantity_mean_rows(score_rows, keep_layers=k)
            for r in lq_rows:
                heatmap_rows_all.append({"keep_layers": int(k), **r})

            control_mean = float(np.mean([r["control_score"] for r in score_rows])) if score_rows else 0.0

            overview_rows.append(
                {
                    "keep_layers": int(k),
                    "reduction_mode": args.reduction_mode,
                    "num_samples": int(len(trajs)),
                    "utilization_active_pct": util_active,
                    "coverage_pct": coverage["global_coverage_pct"],
                    "active_unique_codes": coverage["active_unique_codes"],
                    "resolution_score": resolution["scenario_resolution_score"],
                    "resolution_intra_mean": resolution["scenario_intra_consistency_mean"],
                    "resolution_inter_sep": resolution["scenario_inter_separability"],
                    "control_score_mean": control_mean,
                }
            )
            print(
                f"[K={k:02d}] util={util_active:.2f}% coverage={coverage['global_coverage_pct']:.2f}% "
                f"resolution={resolution['scenario_resolution_score']:.4f} control_mean={control_mean:.4f}"
            )

    overview_csv = os.path.join(output_dir, "token_reduction_overview.csv")
    heatmap_mean_csv = os.path.join(output_dir, "token_reduction_control_layer_quantity_mean.csv")
    heatmap_compare_png = os.path.join(output_dir, "token_reduction_control_heatmap_compare.png")
    top1_compare_png = os.path.join(output_dir, "token_reduction_control_top1_map_compare.png")
    summary_json = os.path.join(output_dir, "token_reduction_summary.json")

    write_csv(overview_csv, overview_rows)
    write_csv(heatmap_mean_csv, heatmap_rows_all)
    plot_control_heatmap_compare(score_rows_by_k, token_counts=token_counts, out_path=heatmap_compare_png)
    plot_top1_map_compare(top_rows_by_k, token_counts=token_counts, out_path=top1_compare_png)

    with open(summary_json, "w") as f:
        json.dump(
            {
                "config": {
                    "data_path": args.data_path,
                    "save_dir": args.save_dir,
                    "data_type": args.data_type,
                    "batch_size": args.batch_size,
                    "max_samples": args.max_samples,
                    "dt": args.dt,
                    "token_counts": token_counts,
                    "reduction_mode": args.reduction_mode,
                    "model_path_map": model_path_map,
                    "model_path_template": args.model_path_template,
                },
                "num_samples": int(len(trajs)),
                "outputs": {
                    "overview_csv": overview_csv,
                    "control_layer_quantity_mean_csv": heatmap_mean_csv,
                    "control_heatmap_compare_png": heatmap_compare_png,
                    "control_top1_map_compare_png": top1_compare_png,
                },
                "overview": overview_rows,
            },
            f,
            indent=2,
        )

    print("\nToken reduction experiment done.")
    print(f"Saved outputs to: {output_dir}")
    print(f"Overview CSV: {overview_csv}")
    print(f"Control heatmap compare: {heatmap_compare_png}")
    print(f"Control top1 map compare: {top1_compare_png}")


if __name__ == "__main__":
    main()
