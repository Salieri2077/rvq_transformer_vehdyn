import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from train_tfm import TrajRVQTransformer, train_rvq_taae
    from eval_tokenizer_by_scenario import (
        load_norm_params,
        normalize_trajs,
        reconstruct_trajs,
        scenario_metrics,
        traj_signature,
    )
    from eval_tokenizer_health import evaluate_tokenizer_health
    from utils import compute_kinematic_profiles, load_sampled_datas
except ImportError:
    from rvq_transformer_vehdyn.train_tfm import TrajRVQTransformer, train_rvq_taae
    from rvq_transformer_vehdyn.eval_tokenizer_by_scenario import (
        load_norm_params,
        normalize_trajs,
        reconstruct_trajs,
        scenario_metrics,
        traj_signature,
    )
    from rvq_transformer_vehdyn.eval_tokenizer_health import evaluate_tokenizer_health
    from rvq_transformer_vehdyn.utils import compute_kinematic_profiles, load_sampled_datas


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_similarity_features(
    trajs: np.ndarray,
    xy_weight: float = 1.0,
    yaw_weight: float = 3.0,
) -> np.ndarray:
    """Convert [N, T, 3] trajectories into normalized feature vectors for nearest-neighbor search."""
    x = np.asarray(trajs, dtype=np.float32).copy()
    # 通过权重调节“位置变化(dx,dy)”与“航向变化(dyaw)”在相似度中的相对重要性。
    x[..., :2] *= float(xy_weight)
    x[..., 2] *= float(yaw_weight)
    feats = x.reshape(x.shape[0], -1)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    return (feats - mean) / (std + 1e-6)


def select_similar_group(
    all_trajs: np.ndarray,
    group_size: int,
    search_pool_size: int,
    anchor_candidates: int,
    seed: int,
    xy_weight: float,
    yaw_weight: float,
) -> Dict[str, object]:
    """
    Select a compact trajectory group:
      - trajectories should be very close in feature space
      - signatures must be unique (avoid selecting duplicate same trajectory)
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    n_total = int(all_trajs.shape[0])
    if n_total < group_size:
        raise ValueError(f"Dataset too small: N={n_total}, group_size={group_size}")

    rng = np.random.default_rng(seed)
    pool_size = min(max(1, int(search_pool_size)), n_total)
    # 数据量大时先抽子集，降低 O(N^2) 级别搜索的实际开销。
    if pool_size < n_total:
        pool_global_idx = np.sort(rng.choice(n_total, size=pool_size, replace=False))
    else:
        pool_global_idx = np.arange(n_total, dtype=np.int64)

    pool_trajs = np.asarray(all_trajs[pool_global_idx], dtype=np.float32)
    feats = build_similarity_features(pool_trajs, xy_weight=xy_weight, yaw_weight=yaw_weight)
    signatures = [traj_signature(pool_trajs[i]) for i in range(pool_size)]

    first_by_signature: Dict[bytes, int] = {}
    # 用轨迹签名去重，避免“同一条轨迹的重复样本”进入相似组。
    for i, sign in enumerate(signatures):
        if sign not in first_by_signature:
            first_by_signature[sign] = i
    unique_anchor_idx = np.asarray(list(first_by_signature.values()), dtype=np.int64)

    if unique_anchor_idx.shape[0] < group_size:
        raise ValueError(
            "Not enough unique trajectories after de-duplication. "
            f"need={group_size}, unique={unique_anchor_idx.shape[0]}"
        )

    num_candidates = min(max(1, int(anchor_candidates)), unique_anchor_idx.shape[0])
    if num_candidates < unique_anchor_idx.shape[0]:
        candidate_anchor_idx = rng.choice(unique_anchor_idx, size=num_candidates, replace=False)
    else:
        candidate_anchor_idx = unique_anchor_idx

    best: Optional[Dict[str, object]] = None

    for anchor_local_idx in candidate_anchor_idx.tolist():
        # 固定一个 anchor，按特征空间距离从近到远选其余轨迹。
        diff = feats - feats[anchor_local_idx][None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        order = np.argsort(dist2)

        selected_local_idx = [int(anchor_local_idx)]
        selected_dist2 = [0.0]
        used_signatures = {signatures[anchor_local_idx]}

        for cand_local_idx in order.tolist():
            if cand_local_idx == anchor_local_idx:
                continue
            sign = signatures[cand_local_idx]
            if sign in used_signatures:
                continue
            used_signatures.add(sign)
            selected_local_idx.append(int(cand_local_idx))
            selected_dist2.append(float(dist2[cand_local_idx]))
            if len(selected_local_idx) >= group_size:
                break

        if len(selected_local_idx) < group_size:
            continue

        # 组紧凑度：anchor 到其余样本的平均距离，越小越相似。
        score = float(np.mean(selected_dist2[1:]))
        if (best is None) or (score < best["score"]):
            best = {
                "score": score,
                "anchor_local_idx": int(anchor_local_idx),
                "selected_local_idx": selected_local_idx,
                "selected_dist2": selected_dist2,
                "pool_size": int(pool_size),
                "candidate_anchors": int(num_candidates),
            }

    if best is None:
        raise RuntimeError("Failed to find a valid similar trajectory group. Try increasing search_pool_size.")

    selected_local_idx = np.asarray(best["selected_local_idx"], dtype=np.int64)
    selected_global_idx = pool_global_idx[selected_local_idx]
    selected_l2 = np.sqrt(np.asarray(best["selected_dist2"], dtype=np.float64))

    return {
        "group_global_idx": selected_global_idx,
        "group_l2": selected_l2,
        "anchor_global_idx": int(pool_global_idx[int(best["anchor_local_idx"])]),
        "score_l2": float(np.sqrt(best["score"])),
        "search_pool_size": int(best["pool_size"]),
        "anchor_candidates": int(best["candidate_anchors"]),
    }


def build_model_and_norm(
    save_dir: str,
    data_type: str,
    input_steps: int,
    num_layers: int,
    num_transformer_layers: int,
    device: torch.device,
) -> Tuple[TrajRVQTransformer, Dict[str, torch.Tensor], str]:
    model_path = os.path.join(save_dir, f"{data_type}_rvq_taae_model.pth")
    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")

    # 模型结构参数需要与训练保存时一致，确保权重可严格加载。
    model = TrajRVQTransformer(
        input_steps=input_steps,
        input_dim=3,
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=num_transformer_layers,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    # 评估时必须使用训练阶段保存的归一化参数，否则指标不可比。
    norm_params = load_norm_params(norm_path, device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])
    model.eval()
    return model, norm_params, model_path


def compute_per_sample_errors(
    gt_trajs: np.ndarray,
    pred_trajs: np.ndarray,
    dt: float,
    sample_global_idx: List[int],
    distance_to_train: List[float],
) -> List[Dict[str, float]]:
    gt_prof = compute_kinematic_profiles(gt_trajs, dt=dt)
    pred_prof = compute_kinematic_profiles(pred_trajs, dt=dt)
    # 在全局坐标系下计算逐时刻位移误差，随后得到 ADE/FDE/MaxErr。
    step_dist = np.sqrt(np.sum((pred_prof["xy"] - gt_prof["xy"]) ** 2, axis=-1) + 1e-6)

    ade = step_dist.mean(axis=1)
    fde = step_dist[:, -1]
    max_err = step_dist.max(axis=1)

    out = []
    for i in range(len(sample_global_idx)):
        out.append(
            {
                "global_idx": int(sample_global_idx[i]),
                "distance_to_train_l2": float(distance_to_train[i]),
                "ade_m": float(ade[i]),
                "fde_m": float(fde[i]),
                "max_traj_error_m": float(max_err[i]),
            }
        )
    return out


def write_per_sample_csv(rows: List[Dict[str, float]], csv_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-trajectory training on highly similar trajectory group.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--search-pool-size", type=int, default=40000)
    parser.add_argument("--anchor-candidates", type=int, default=2000)
    parser.add_argument("--feature-xy-weight", type=float, default=1.0)
    parser.add_argument("--feature-yaw-weight", type=float, default=3.0)

    parser.add_argument("--save-root", type=str, default="./work_dirs/tokenizer/similar_single_train")
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--train-repeat", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)

    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-health-eval", action="store_true")
    parser.add_argument("--noise-std-xy", type=float, default=0.05)
    parser.add_argument("--noise-std-yaw", type=float, default=0.01)
    args = parser.parse_args()

    _set_seed(args.seed)

    trajs = load_sampled_datas(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    if args.max_samples > 0:
        trajs = trajs[: args.max_samples]
    trajs = np.asarray(trajs, dtype=np.float32)

    # 1) 先自动挑一组“彼此非常接近且不重复”的轨迹。
    select_info = select_similar_group(
        all_trajs=trajs,
        group_size=args.group_size,
        search_pool_size=args.search_pool_size,
        anchor_candidates=args.anchor_candidates,
        seed=args.seed,
        xy_weight=args.feature_xy_weight,
        yaw_weight=args.feature_yaw_weight,
    )

    group_global_idx = np.asarray(select_info["group_global_idx"], dtype=np.int64)
    group_l2 = np.asarray(select_info["group_l2"], dtype=np.float64)

    # 2) 组内第 1 条用于训练，其余用于“近邻泛化”评估。
    train_idx = int(group_global_idx[0])
    eval_idx = group_global_idx[1:]

    run_name = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(args.save_root, f"{args.data_type}_single_train_{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    selected_trajs = trajs[group_global_idx]
    np.save(os.path.join(output_dir, "selected_similar_group.npy"), selected_trajs)

    train_traj = trajs[train_idx : train_idx + 1]
    # 只用单条轨迹训练时，repeat 主要用于形成足够 batch、稳定优化过程。
    train_data = np.repeat(train_traj, repeats=max(1, int(args.train_repeat)), axis=0)

    print("=" * 80)
    print("Selected similar trajectories")
    print(f"output_dir: {output_dir}")
    print(f"group_global_idx: {group_global_idx.tolist()}")
    print(f"group_l2_to_anchor: {[float(x) for x in group_l2.tolist()]}")
    print(f"train_idx: {train_idx}")
    print(f"eval_idx: {eval_idx.tolist()}")
    print("=" * 80)
    
    # 3) 复用现有训练入口，保持与你主训练流程一致。
    train_rvq_taae(
        data_array=train_data,
        save_dir=output_dir,
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_transformer_layers=args.num_transformer_layers,
        epochs=args.epochs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_params, model_path = build_model_and_norm(
        save_dir=output_dir,
        data_type=args.data_type,
        input_steps=trajs.shape[1],
        num_layers=args.num_layers,
        num_transformer_layers=args.num_transformer_layers,
        device=device,
    )

    # 4) 先做训练样本自重建指标（用于看是否至少学会该单条轨迹）。
    train_recon, train_codes = reconstruct_trajs(
        model=model,
        trajs=train_traj,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=1,
    )
    train_metrics = scenario_metrics(train_traj, train_recon, dt=args.dt)

    eval_trajs = trajs[eval_idx]
    # 5) 再在其余“相似轨迹”上评估泛化。
    eval_recon, eval_codes = reconstruct_trajs(
        model=model,
        trajs=eval_trajs,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=max(1, args.eval_batch_size),
    )
    eval_metrics = scenario_metrics(eval_trajs, eval_recon, dt=args.dt)

    # 6) 逐条轨迹误差，便于分析“距离训练样本越远是否越难重建”。
    per_sample_rows = compute_per_sample_errors(
        gt_trajs=eval_trajs,
        pred_trajs=eval_recon,
        dt=args.dt,
        sample_global_idx=eval_idx.tolist(),
        distance_to_train=group_l2[1:].tolist(),
    )

    health_summary = None
    if not args.skip_health_eval:
        # 7) 可选健康度评估：统计 token 利用率和加噪前后 token 重合度(OR)。
        eval_norm = normalize_trajs(
            eval_trajs,
            mean=norm_params["mean"],
            std=norm_params["std"],
            scale_factor=norm_params["scale_factor"],
            clip_limit=norm_params["clip_limit"],
        )
        eval_loader = DataLoader(
            TensorDataset(eval_norm),
            batch_size=min(max(1, args.eval_batch_size), max(1, len(eval_trajs))),
            shuffle=False,
        )
        util, overlap = evaluate_tokenizer_health(
            model=model,
            dataloader=eval_loader,
            device=device,
            noise_std_xy=args.noise_std_xy,
            noise_std_yaw=args.noise_std_yaw,
            clip_limit=norm_params["clip_limit"],
        )
        health_summary = {
            "avg_codebook_utilization_pct": float(util),
            "avg_overlap_rate_pct": float(overlap),
            "noise_std_xy": float(args.noise_std_xy),
            "noise_std_yaw": float(args.noise_std_yaw),
        }

    layer_unique_codes = []
    if eval_codes.size > 0:
        for i in range(eval_codes.shape[1]):
            layer_unique_codes.append(int(np.unique(eval_codes[:, i]).size))

    # 8) 统一落盘，便于后续多次实验对比。
    summary = {
        "config": {
            "data_path": args.data_path,
            "data_type": args.data_type,
            "max_samples": int(args.max_samples),
            "group_size": int(args.group_size),
            "search_pool_size": int(args.search_pool_size),
            "anchor_candidates": int(args.anchor_candidates),
            "feature_xy_weight": float(args.feature_xy_weight),
            "feature_yaw_weight": float(args.feature_yaw_weight),
            "save_root": args.save_root,
            "output_dir": output_dir,
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "train_repeat": int(args.train_repeat),
            "num_layers": int(args.num_layers),
            "num_transformer_layers": int(args.num_transformer_layers),
            "epochs": int(args.epochs),
            "dt": float(args.dt),
            "seed": int(args.seed),
        },
        "selection": {
            "group_global_idx": [int(x) for x in group_global_idx.tolist()],
            "group_l2_to_anchor": [float(x) for x in group_l2.tolist()],
            "anchor_global_idx": int(select_info["anchor_global_idx"]),
            "train_idx": int(train_idx),
            "eval_idx": [int(x) for x in eval_idx.tolist()],
            "compactness_score_l2": float(select_info["score_l2"]),
            "effective_search_pool_size": int(select_info["search_pool_size"]),
            "effective_anchor_candidates": int(select_info["anchor_candidates"]),
        },
        "model": {
            "model_path": model_path,
            "norm_path": os.path.join(output_dir, f"{args.data_type}_norm_params.pkl"),
            "eval_codes_shape": list(eval_codes.shape),
            "eval_unique_codes_per_layer": layer_unique_codes,
            "train_codes": train_codes.astype(int).tolist(),
        },
        "metrics": {
            "train_self_recon": train_metrics,
            "eval_similar_group": eval_metrics,
            "eval_per_sample": per_sample_rows,
            "tokenizer_health": health_summary,
        },
    }

    json_path = os.path.join(output_dir, "similar_single_train_summary.json")
    csv_path = os.path.join(output_dir, "eval_per_sample_metrics.csv")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_per_sample_csv(per_sample_rows, csv_path)

    print("=" * 80)
    print("Experiment Done")
    print(f"Model:   {model_path}")
    print(f"Summary: {json_path}")
    print(f"Per-sample CSV: {csv_path}")
    print("Train self metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")
    print("Eval similar metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")
    if health_summary is not None:
        print("Tokenizer health:")
        for k, v in health_summary.items():
            print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
