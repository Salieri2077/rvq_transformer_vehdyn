import argparse
import os
import time
from typing import List

import numpy as np
import torch

try:
    from train_tfm import train_rvq_taae
    from eval_tokenizer_by_scenario import reconstruct_trajs, scenario_metrics
    from experiment_similar_traj_single_train import build_model_and_norm
    from utils import (
        build_repeated_single_sample_dataset,
        compute_reconstruction_case_metrics,
        decode_token_prefix_reconstructions,
        load_sampled_datas,
        load_source_indices,
        plot_target_token_prefix_reconstructions,
        resolve_default_data_path,
        resolve_sample_idx,
        resolve_source_indices_path,
        token_sequence_to_str,
        write_json,
    )
except ImportError:
    from rvq_transformer_vehdyn.train_tfm import train_rvq_taae
    from rvq_transformer_vehdyn.eval_tokenizer_by_scenario import reconstruct_trajs, scenario_metrics
    from rvq_transformer_vehdyn.experiment_similar_traj_single_train import build_model_and_norm
    from rvq_transformer_vehdyn.utils import (
        build_repeated_single_sample_dataset,
        compute_reconstruction_case_metrics,
        decode_token_prefix_reconstructions,
        load_sampled_datas,
        load_source_indices,
        plot_target_token_prefix_reconstructions,
        resolve_default_data_path,
        resolve_sample_idx,
        resolve_source_indices_path,
        token_sequence_to_str,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit train_tfm.py on repeated copies of one trajectory sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--sample-idx", type=int, default=None, help="Direct row index in the loaded data.")
    parser.add_argument(
        "--source-sample-idx",
        type=int,
        default=964872,
        help="Original/source row id. Uses source_indices sidecar when available; otherwise falls back to direct row index.",
    )
    parser.add_argument(
        "--source-indices-path",
        type=str,
        default="",
        help="Optional sidecar where source_indices[current_row] = original row id. Auto-detected from data-path if omitted.",
    )
    parser.add_argument(
        "--source-occurrence",
        type=int,
        default=0,
        help="When source_sample_idx appears multiple times in augmented data, choose this occurrence.",
    )
    parser.add_argument("--repeat-count", type=int, default=512)
    parser.add_argument("--save-root", type=str, default="./work_dirs/tokenizer/single_sample_repeat_train")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-train-data", action="store_true")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--prefix-lengths",
        type=str,
        default="1,3,5,10,15",
        help="Comma-separated token-prefix lengths to visualize after training.",
    )
    parser.add_argument("--skip-prefix-plot", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_prefix_lengths(raw: str, max_layers: int) -> List[int]:
    out: List[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if 1 <= value <= int(max_layers) and value not in out:
            out.append(value)
    return out


def default_output_dir(args: argparse.Namespace, resolved_sample_idx: int) -> str:
    if args.output_dir:
        return args.output_dir
    source_label = args.sample_idx if args.sample_idx is not None else args.source_sample_idx
    source_label = resolved_sample_idx if source_label is None else source_label
    run_name = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        args.save_root,
        f"{args.data_type}_src{int(source_label)}_row{int(resolved_sample_idx)}_repeat{int(args.repeat_count)}_{run_name}",
    )


def main() -> None:
    args = parse_args()
    if int(args.epochs) <= 5:
        raise ValueError("train_rvq_taae uses a 5-epoch warmup scheduler; set --epochs > 5.")
    if int(args.repeat_count) <= 0:
        raise ValueError("--repeat-count must be positive.")

    set_seed(int(args.seed))

    data_path = args.data_path or resolve_default_data_path()
    trajs = load_sampled_datas(data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    trajs = np.asarray(trajs, dtype=np.float32)
    n_total = int(trajs.shape[0])

    source_indices_path = resolve_source_indices_path(data_path, args.source_indices_path)
    source_indices = None
    source_match_count = 0
    if source_indices_path:
        source_indices = load_source_indices(source_indices_path, n_total=n_total)
        if args.source_sample_idx is not None:
            source_match_count = int(np.sum(source_indices == int(args.source_sample_idx)))

    resolved_sample_idx = resolve_sample_idx(
        n_total=n_total,
        sample_idx=args.sample_idx,
        source_sample_idx=args.source_sample_idx,
        source_indices=source_indices,
        source_occurrence=int(args.source_occurrence),
    )
    train_trajs = build_repeated_single_sample_dataset(
        trajs=trajs,
        sample_idx=resolved_sample_idx,
        repeat_count=int(args.repeat_count),
    )

    output_dir = default_output_dir(args, resolved_sample_idx=resolved_sample_idx)
    os.makedirs(output_dir, exist_ok=True)

    train_indices = np.repeat(np.asarray([resolved_sample_idx], dtype=np.int64), int(args.repeat_count))
    train_indices_path = os.path.join(output_dir, "train_indices.npy")
    np.save(train_indices_path, train_indices)
    np.save(os.path.join(output_dir, "target_sample.npy"), trajs[resolved_sample_idx : resolved_sample_idx + 1])
    if args.save_train_data:
        np.save(os.path.join(output_dir, "repeated_train_trajs.npy"), train_trajs)

    print("=" * 80)
    print("Single-Sample Repeat Train")
    print(f"data_path: {data_path}")
    print(f"source_indices_path: {source_indices_path or '<none>'}")
    print(f"source_sample_idx: {args.source_sample_idx}")
    print(f"resolved_sample_idx: {resolved_sample_idx}")
    print(f"source_match_count: {source_match_count}")
    print(f"train_shape: {train_trajs.shape}")
    print(f"output_dir: {output_dir}")
    print("=" * 80)

    train_rvq_taae(
        train_trajs,
        save_dir=output_dir,
        data_type=args.data_type,
        batch_size=int(args.batch_size),
        num_layers=int(args.num_layers),
        num_transformer_layers=int(args.num_transformer_layers),
        epochs=int(args.epochs),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_params, model_path = build_model_and_norm(
        save_dir=output_dir,
        data_type=args.data_type,
        input_steps=int(trajs.shape[1]),
        num_layers=int(args.num_layers),
        num_transformer_layers=int(args.num_transformer_layers),
        device=device,
    )

    target_traj = trajs[resolved_sample_idx : resolved_sample_idx + 1]
    recon_trajs, codes = reconstruct_trajs(
        model=model,
        trajs=target_traj,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=max(1, int(args.eval_batch_size)),
    )

    scenario_eval = scenario_metrics(target_traj, recon_trajs, dt=float(args.dt))
    case_arrays = compute_reconstruction_case_metrics(target_traj, recon_trajs, dt=float(args.dt))
    case_metrics = {key: float(np.asarray(value).reshape(-1)[0]) for key, value in case_arrays.items()}

    recon_npz_path = os.path.join(output_dir, "single_sample_reconstruction.npz")
    np.savez(
        recon_npz_path,
        target=target_traj.astype(np.float32),
        reconstruction=recon_trajs.astype(np.float32),
        codes=codes.astype(np.int64),
        train_indices=train_indices,
    )

    prefix_plot_path = ""
    prefix_plot_error = ""
    prefix_lengths_used: List[int] = []
    if not args.skip_prefix_plot:
        try:
            prefix_lengths = parse_prefix_lengths(args.prefix_lengths, max_layers=int(args.num_layers))
            used, prefix_recons = decode_token_prefix_reconstructions(
                model=model,
                codes=codes[0],
                norm_params=norm_params,
                prefix_lengths=prefix_lengths,
            )
            if used.size > 0:
                prefix_lengths_used = [int(v) for v in used.tolist()]
                prefix_plot_path = os.path.join(output_dir, f"target_sample_{int(resolved_sample_idx)}_token_prefix_recon.png")
                plot_target_token_prefix_reconstructions(
                    sample_idx=int(resolved_sample_idx),
                    gt_traj=target_traj[0],
                    full_recon_traj=recon_trajs[0],
                    prefix_lengths=used,
                    prefix_recon_trajs=prefix_recons,
                    dt=float(args.dt),
                    save_path=prefix_plot_path,
                    tokens=codes[0],
                    metrics=case_metrics,
                )
        except Exception as exc:  # Plotting should not hide the core train/eval result.
            prefix_plot_error = repr(exc)
            print(f"[warning] prefix plot skipped: {prefix_plot_error}")

    summary = {
        "selection": {
            "data_path": os.path.abspath(data_path),
            "data_type": args.data_type,
            "source_indices_path": os.path.abspath(source_indices_path) if source_indices_path else "",
            "source_sample_idx": int(args.source_sample_idx) if args.source_sample_idx is not None else None,
            "sample_idx_arg": int(args.sample_idx) if args.sample_idx is not None else None,
            "resolved_sample_idx": int(resolved_sample_idx),
            "source_occurrence": int(args.source_occurrence),
            "source_match_count": int(source_match_count),
        },
        "training": {
            "repeat_count": int(args.repeat_count),
            "train_shape": list(train_trajs.shape),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_layers": int(args.num_layers),
            "num_transformer_layers": int(args.num_transformer_layers),
            "seed": int(args.seed),
            "train_indices_path": os.path.abspath(train_indices_path),
        },
        "metrics": {
            "scenario_metrics": scenario_eval,
            "case_metrics": case_metrics,
            "tokens": token_sequence_to_str(codes[0]),
        },
        "files": {
            "model_path": os.path.abspath(model_path),
            "reconstruction_npz": os.path.abspath(recon_npz_path),
            "prefix_plot_path": os.path.abspath(prefix_plot_path) if prefix_plot_path else "",
            "prefix_plot_error": prefix_plot_error,
            "prefix_lengths_used": prefix_lengths_used,
        },
    }
    summary_path = os.path.join(output_dir, "single_sample_repeat_train_summary.json")
    write_json(summary_path, summary)

    print("=" * 80)
    print("Experiment Done")
    print(f"model_path: {model_path}")
    print(f"summary_path: {summary_path}")
    print(
        "target metrics | "
        f"recon_mse={case_metrics['recon_mse']:.6g} | "
        f"ade={case_metrics['ade']:.6g} | "
        f"fde={case_metrics['fde']:.6g} | "
        f"max_error={case_metrics['max_error']:.6g}"
    )
    print(f"tokens: {token_sequence_to_str(codes[0])}")
    if prefix_plot_path:
        print(f"prefix_plot_path: {prefix_plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
