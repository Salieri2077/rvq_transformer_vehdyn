import argparse
import os
import pickle
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from train_tfm import (
        TrajRVQTransformer,
        train_rvq_taae,
        signed_velocity_loss_from_dxdydyaw,
        signed_acceleration_loss_from_dxdydyaw,
        turn_global_yaw_loss,
    )
    from eval_tokenizer_by_scenario import (
        load_norm_params,
        normalize_trajs,
        reconstruct_trajs,
        scenario_metrics,
    )
    from eval_tokenizer_health import evaluate_tokenizer_health
    from utils import (
        load_sampled_datas,
        preprocess_and_save_norm_params,
        integrate_to_global,
        percentiles as _percentiles,
        write_json as _write_json,
        write_csv as _write_csv,
    )
    from grouping_pipeline import (
        build_grouping_cache_key,
        try_load_grouping_from_cache,
        save_grouping_cache,
        compute_motion_stats,
        find_representative_groups,
    )
except ImportError:
    from rvq_transformer_vehdyn.train_tfm import (
        TrajRVQTransformer,
        train_rvq_taae,
        signed_velocity_loss_from_dxdydyaw,
        signed_acceleration_loss_from_dxdydyaw,
        turn_global_yaw_loss,
    )
    from rvq_transformer_vehdyn.eval_tokenizer_by_scenario import (
        load_norm_params,
        normalize_trajs,
        reconstruct_trajs,
        scenario_metrics,
    )
    from rvq_transformer_vehdyn.eval_tokenizer_health import evaluate_tokenizer_health
    from rvq_transformer_vehdyn.utils import (
        load_sampled_datas,
        preprocess_and_save_norm_params,
        integrate_to_global,
        percentiles as _percentiles,
        write_json as _write_json,
        write_csv as _write_csv,
    )
    from rvq_transformer_vehdyn.grouping_pipeline import (
        build_grouping_cache_key,
        try_load_grouping_from_cache,
        save_grouping_cache,
        compute_motion_stats,
        find_representative_groups,
    )


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def get_primary_codebook_weight(model: TrajRVQTransformer) -> Optional[torch.Tensor]:
    """尽力提取 RVQ 第一层 codebook（兼容不同实现字段名）。"""
    rvq = getattr(model, "rvq", None)
    if rvq is None:
        return None

    def _extract_weight(obj) -> Optional[torch.Tensor]:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, "weight") and isinstance(getattr(obj, "weight"), torch.Tensor):
            return getattr(obj, "weight")
        return None

    def _to_2d(w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if w is None:
            return None
        if w.ndim == 2:
            return w
        if w.ndim == 3:
            return w[0]
        return None

    if hasattr(rvq, "layers") and len(rvq.layers) > 0:
        layer0 = rvq.layers[0]
        for name in ["embedding", "embeddings", "codebook", "codebooks"]:
            if hasattr(layer0, name):
                w = _to_2d(_extract_weight(getattr(layer0, name)))
                if w is not None:
                    return w
        w = _to_2d(_extract_weight(layer0))
        if w is not None:
            return w

    for name in ["codebooks", "codebook", "embedding", "embeddings"]:
        if hasattr(rvq, name):
            obj = getattr(rvq, name)
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                w = _to_2d(_extract_weight(obj[0]))
            else:
                w = _to_2d(_extract_weight(obj))
            if w is not None:
                return w
    return None


def flatten_or_pool_latent(z: torch.Tensor) -> torch.Tensor:
    """将 encoder 输出统一为 [B, D]，便于 consistency 计算。"""
    if z.ndim == 2:
        return z
    if z.ndim == 3:
        return z.mean(dim=1)
    return z.view(z.shape[0], -1)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """监督式对比损失：同 label 为正样本，不同 label 为负样本。"""
    if features.shape[0] <= 1:
        return features.new_zeros(())

    z = F.normalize(features, dim=1)
    sim = torch.matmul(z, z.t()) / max(float(temperature), 1e-6)
    n = int(sim.shape[0])

    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    pos_mask = (labels[:, None] == labels[None, :]) & (~eye)
    den_mask = ~eye

    has_pos = pos_mask.any(dim=1)
    if not torch.any(has_pos):
        return sim.new_zeros(())

    losses = []
    for i in torch.where(has_pos)[0].tolist():
        sim_i = sim[i]
        pos_lse = torch.logsumexp(sim_i[pos_mask[i]], dim=0)
        den_lse = torch.logsumexp(sim_i[den_mask[i]], dim=0)
        losses.append(-(pos_lse - den_lse))
    return torch.stack(losses).mean() if losses else sim.new_zeros(())


def sample_consistency_batch(
    trajs: np.ndarray,
    group_id_per_sample: np.ndarray,
    group_to_indices: List[np.ndarray],
    positive_groups_per_step: int,
    positives_per_group: int,
    negative_groups_per_step: int,
    seed_or_rng,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态采样 consistency batch，避免在 108 万条上构造全量标签矩阵。
    返回:
      consistency_trajs: [B, T, 3]
      labels: [B]
    """
    if isinstance(seed_or_rng, np.random.Generator):
        rng = seed_or_rng
    else:
        rng = np.random.default_rng(int(seed_or_rng))

    valid_positive_groups = [gid for gid, idxs in enumerate(group_to_indices) if idxs.size >= 2]
    if len(valid_positive_groups) == 0:
        return np.zeros((0,) + trajs.shape[1:], dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n_pos_groups = min(max(1, int(positive_groups_per_step)), len(valid_positive_groups))
    pos_groups = rng.choice(valid_positive_groups, size=n_pos_groups, replace=False)

    traj_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    next_label = 0
    selected_pos_group_set = set(int(g) for g in pos_groups.tolist())
    for gid in pos_groups.tolist():
        idxs = group_to_indices[int(gid)]
        take = min(max(1, int(positives_per_group)), int(idxs.size))
        picked = rng.choice(idxs, size=take, replace=False)
        traj_parts.append(trajs[picked].astype(np.float32))
        label_parts.append(np.full((take,), next_label, dtype=np.int64))
        next_label += 1

    # negatives: 其它 group 各取 1 条，作为分母项。
    candidate_neg_groups = [gid for gid, idxs in enumerate(group_to_indices) if idxs.size > 0 and gid not in selected_pos_group_set]
    if len(candidate_neg_groups) > 0 and int(negative_groups_per_step) > 0:
        n_neg = min(int(negative_groups_per_step), len(candidate_neg_groups))
        neg_groups = rng.choice(candidate_neg_groups, size=n_neg, replace=False)
        for gid in neg_groups.tolist():
            idxs = group_to_indices[int(gid)]
            picked = int(rng.choice(idxs, size=1, replace=False)[0])
            traj_parts.append(trajs[picked : picked + 1].astype(np.float32))
            label_parts.append(np.full((1,), next_label, dtype=np.int64))
            next_label += 1

    if not traj_parts:
        return np.zeros((0,) + trajs.shape[1:], dtype=np.float32), np.zeros((0,), dtype=np.int64)

    x = np.concatenate(traj_parts, axis=0).astype(np.float32)
    y = np.concatenate(label_parts, axis=0).astype(np.int64)
    return x, y


def _normalize_by_saved_params(
    trajs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: Optional[torch.Tensor],
) -> torch.Tensor:
    x = (trajs - mean) / (std + 1e-8)
    if clip_limit is not None:
        x = torch.clamp(x, -clip_limit, clip_limit)
    return x / scale_factor


def _base_taae_loss(
    model: TrajRVQTransformer,
    x_norm: torch.Tensor,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """保持 train_rvq_taae 的 base loss 风格。"""
    x_recon, vq_loss, _, v, kappa = model(x_norm)

    mse_dxdy = F.mse_loss(x_recon[..., :2], x_norm[..., :2])
    mse_dyaw = F.mse_loss(x_recon[..., 2], x_norm[..., 2])
    recon_loss = mse_dxdy + 14.0 * mse_dyaw

    pred_phys = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
    gt_phys = (x_norm * model.norm_scale * model.norm_std) + model.norm_mean

    vel_loss = signed_velocity_loss_from_dxdydyaw(pred_phys, gt_phys, dt=model.dt)
    acc_loss = signed_acceleration_loss_from_dxdydyaw(pred_phys, gt_phys, dt=model.dt)
    turn_global_loss, turn_yaw_loss, _ = turn_global_yaw_loss(
        pred_phys,
        gt_phys,
        turn_threshold=0.35,
    )

    acc = (v[:, 1:] - v[:, :-1]) / model.dt
    kappa_rate = (kappa[:, 1:] - kappa[:, :-1]) / model.dt
    kin_smooth_loss = acc.pow(2).mean() + kappa_rate.pow(2).mean()

    recon_loss_weight = 10.0
    vq_loss_weight = 5.0
    vel_loss_weight = 0.5
    acc_loss_weight = 0.05
    kin_smooth_weight = 1e-2 if epoch > 30 else 0.0
    turn_global_weight = 1.0
    turn_yaw_weight = 2.0

    base_loss = (
        recon_loss_weight * recon_loss
        + vq_loss_weight * vq_loss
        + vel_loss_weight * vel_loss
        + acc_loss_weight * acc_loss
        + kin_smooth_weight * kin_smooth_loss
        + turn_global_weight * turn_global_loss
        + turn_yaw_weight * turn_yaw_loss
    )

    terms = {
        "recon_loss": recon_loss,
        "vq_loss": vq_loss,
        "vel_loss": vel_loss,
        "acc_loss": acc_loss,
        "turn_global_loss": turn_global_loss,
        "turn_yaw_loss": turn_yaw_loss,
        "kin_smooth_loss": kin_smooth_loss,
    }
    return base_loss, terms, x_recon


def _latent_consistency_loss(z_pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for lab in torch.unique(labels).tolist():
        mask = labels == int(lab)
        if int(mask.sum().item()) < 2:
            continue
        z_group = F.normalize(z_pooled[mask], dim=1)
        proto = F.normalize(z_group.detach().mean(dim=0, keepdim=True), dim=1)
        losses.append((1.0 - torch.sum(z_group * proto, dim=1)).mean())
    if not losses:
        return z_pooled.new_zeros(())
    return torch.stack(losses).mean()


def _soft_primary_code_consistency_loss(
    z_pooled: torch.Tensor,
    labels: torch.Tensor,
    codebook_weight: Optional[torch.Tensor],
    temperature: float,
    warn_state: Dict[str, bool],
) -> torch.Tensor:
    if codebook_weight is None:
        return z_pooled.new_zeros(())

    cb = codebook_weight
    if cb.ndim != 2:
        if not warn_state.get("warned_codebook_shape", False):
            print("[warning] primary codebook 维度异常，soft_primary_code consistency 已跳过。")
            warn_state["warned_codebook_shape"] = True
        return z_pooled.new_zeros(())

    if cb.shape[1] != z_pooled.shape[1] and cb.shape[0] == z_pooled.shape[1]:
        cb = cb.t()

    if cb.shape[1] != z_pooled.shape[1]:
        if not warn_state.get("warned_codebook_dim", False):
            print("[warning] primary codebook 与 latent 维度不匹配，soft_primary_code consistency 已跳过。")
            warn_state["warned_codebook_dim"] = True
        return z_pooled.new_zeros(())

    z_sq = torch.sum(z_pooled * z_pooled, dim=1, keepdim=True)
    cb_sq = torch.sum(cb * cb, dim=1, keepdim=True).t()
    dist = z_sq + cb_sq - 2.0 * torch.matmul(z_pooled, cb.t())
    prob = torch.softmax(-dist / max(float(temperature), 1e-6), dim=1)

    losses = []
    eps = 1e-8
    for lab in torch.unique(labels).tolist():
        mask = labels == int(lab)
        if int(mask.sum().item()) < 2:
            continue
        p_group = prob[mask]
        p_proto = p_group.detach().mean(dim=0, keepdim=True)

        p_group_safe = torch.clamp(p_group, min=eps)
        p_proto_safe = torch.clamp(p_proto, min=eps)
        p_group_safe = p_group_safe / p_group_safe.sum(dim=1, keepdim=True)
        p_proto_safe = p_proto_safe / p_proto_safe.sum(dim=1, keepdim=True)

        kl_proto_to_group = (p_proto_safe * (torch.log(p_proto_safe) - torch.log(p_group_safe))).sum(dim=1)
        kl_group_to_proto = (p_group_safe * (torch.log(p_group_safe) - torch.log(p_proto_safe))).sum(dim=1)
        losses.append(0.5 * (kl_proto_to_group + kl_group_to_proto).mean())

    if not losses:
        return z_pooled.new_zeros(())
    return torch.stack(losses).mean()


def _hard_code_consistency_loss(
    model: TrajRVQTransformer,
    z_or_z_pooled: torch.Tensor,
    labels: torch.Tensor,
    num_layers: int,
    temperature: float,
) -> torch.Tensor:
    """让同 group 样本的前若干层 RVQ hard token 向 anchor token 对齐。"""
    if int(num_layers) <= 0 or z_or_z_pooled.shape[0] <= 1:
        return z_or_z_pooled.new_zeros(())
    if not hasattr(model.rvq, "forward_with_distances"):
        return z_or_z_pooled.new_zeros(())

    _, _, all_codes, all_dists = model.rvq.forward_with_distances(z_or_z_pooled)
    if all_codes is None or len(all_dists) == 0:
        return z_or_z_pooled.new_zeros(())
    if all_codes.ndim != 2 or any(dist.ndim != 2 for dist in all_dists):
        return z_or_z_pooled.new_zeros(())

    L = min(int(num_layers), int(all_codes.shape[1]), len(all_dists))
    if L <= 0:
        return z_or_z_pooled.new_zeros(())

    losses = []
    temp = max(float(temperature), 1e-6)
    for lab in torch.unique(labels).tolist():
        idx = torch.where(labels == int(lab))[0]
        if int(idx.numel()) < 2:
            continue
        anchor_idx = idx[0]
        member_idx = idx[1:]
        for layer_idx in range(L):
            target_code = all_codes[anchor_idx, layer_idx].detach().expand(member_idx.numel())
            logits = -all_dists[layer_idx][member_idx] / temp
            losses.append(F.cross_entropy(logits.float(), target_code.long()))

    if not losses:
        return z_or_z_pooled.new_zeros(())
    return torch.stack(losses).mean()


def train_rvq_taae_with_group_consistency(
    base_train_trajs: np.ndarray,
    all_trajs: np.ndarray,
    group_id_per_sample: np.ndarray,
    group_to_indices: List[np.ndarray],
    save_dir: str,
    data_type: str,
    batch_size: int,
    num_layers: int,
    num_transformer_layers: int,
    epochs: int,
    positive_groups_per_step: int,
    positives_per_group: int,
    negative_groups_per_step: int,
    lambda_latent_consistency: float,
    lambda_soft_code_consistency: float,
    lambda_supcon: float,
    lambda_hard_code_consistency: float,
    contrastive_temperature: float,
    soft_code_temperature: float,
    hard_code_temperature: float,
    consistency_warmup_epochs: int,
    consistency_target: str,
    hard_code_consistency_layers: int,
    seed: int,
) -> List[Dict[str, float]]:
    """similar_consistency 实验训练：base loss + 动态采样 consistency loss。"""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_normalized = preprocess_and_save_norm_params(base_train_trajs, save_dir, data_type)
    dataset = TensorDataset(torch.tensor(data_normalized, dtype=torch.float32))
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = TrajRVQTransformer(
        input_steps=base_train_trajs.shape[1],
        input_dim=base_train_trajs.shape[2],
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=num_transformer_layers,
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        use_amp = True
        amp_dtype = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        print("Using BF16 mixed precision training")
    elif torch.cuda.is_available():
        use_amp = True
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print("Using FP16 mixed precision training")
    else:
        use_amp = False
        amp_dtype = torch.float32
        scaler = None
        print("Using FP32 training")

    initial_lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-5 / initial_lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs) - warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")
    with open(norm_path, "rb") as f:
        norm_params = pickle.load(f)

    mean = torch.tensor(norm_params["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm_params["std"], dtype=torch.float32, device=device)
    scale_factor = torch.tensor(norm_params["scale_factor"], dtype=torch.float32, device=device)
    clip_limit = torch.tensor(norm_params["clip_limit"], dtype=torch.float32, device=device) if "clip_limit" in norm_params else None
    model.set_norm_params(mean, std, scale_factor)

    run_name = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))

    use_latent = consistency_target in ("latent", "latent_plus_soft_code")
    use_soft = consistency_target in ("soft_primary_code", "latent_plus_soft_code")

    codebook_weight = get_primary_codebook_weight(model) if use_soft else None
    if use_soft and codebook_weight is None:
        print("[warning] 未找到 primary codebook，soft_primary_code consistency 将跳过。")

    rng = np.random.default_rng(seed)
    warn_state = {
        "warned_codebook_shape": False,
        "warned_codebook_dim": False,
    }

    history: List[Dict[str, float]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        if epoch > epochs * 0.8:
            model.rvq.dropout = 0.0

        warmup_weight = 1.0
        if int(consistency_warmup_epochs) > 0:
            warmup_weight = min(1.0, float(epoch) / float(consistency_warmup_epochs))

        total_loss_sum = 0.0
        base_weight_sum = 0.0
        weighted_consistency_sum = 0.0
        recon_sum = 0.0
        vq_sum = 0.0
        vel_sum = 0.0
        acc_sum = 0.0
        kin_smooth_sum = 0.0
        turn_global_sum = 0.0
        turn_yaw_sum = 0.0
        latent_sum = 0.0
        soft_sum = 0.0
        supcon_sum = 0.0
        hard_code_sum = 0.0
        n_batches = 0

        for batch in dataloader:
            x_norm = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                base_loss, base_terms, _ = _base_taae_loss(model=model, x_norm=x_norm, epoch=epoch)

                # 动态采样 consistency batch（原始物理空间），再按保存参数归一化。
                c_trajs_np, c_labels_np = sample_consistency_batch(
                    trajs=all_trajs,
                    group_id_per_sample=group_id_per_sample,
                    group_to_indices=group_to_indices,
                    positive_groups_per_step=positive_groups_per_step,
                    positives_per_group=positives_per_group,
                    negative_groups_per_step=negative_groups_per_step,
                    seed_or_rng=rng,
                )

                latent_loss = x_norm.new_zeros(())
                soft_loss = x_norm.new_zeros(())
                supcon_loss = x_norm.new_zeros(())
                hard_code_loss = x_norm.new_zeros(())

                if c_trajs_np.shape[0] > 1:
                    c_x_phys = torch.tensor(c_trajs_np, dtype=torch.float32, device=device)
                    c_labels = torch.tensor(c_labels_np, dtype=torch.long, device=device)
                    c_x_norm = _normalize_by_saved_params(
                        trajs=c_x_phys,
                        mean=mean,
                        std=std,
                        scale_factor=scale_factor,
                        clip_limit=clip_limit,
                    )
                    z = model.encode(c_x_norm)
                    z_pooled = flatten_or_pool_latent(z)

                    if use_latent:
                        latent_loss = _latent_consistency_loss(z_pooled=z_pooled, labels=c_labels)
                    if use_soft:
                        soft_loss = _soft_primary_code_consistency_loss(
                            z_pooled=z_pooled,
                            labels=c_labels,
                            codebook_weight=codebook_weight,
                            temperature=soft_code_temperature,
                            warn_state=warn_state,
                        )
                    supcon_loss = supervised_contrastive_loss(
                        features=z_pooled,
                        labels=c_labels,
                        temperature=contrastive_temperature,
                    )
                    if float(lambda_hard_code_consistency) > 0.0:
                        hard_code_loss = _hard_code_consistency_loss(
                            model=model,
                            z_or_z_pooled=z_pooled,
                            labels=c_labels,
                            num_layers=hard_code_consistency_layers,
                            temperature=hard_code_temperature,
                        )

                consistency_term = (
                    float(lambda_latent_consistency) * latent_loss
                    + float(lambda_soft_code_consistency) * soft_loss
                    + float(lambda_supcon) * supcon_loss
                    + float(lambda_hard_code_consistency) * hard_code_loss
                )
                total_loss = base_loss + warmup_weight * consistency_term

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += float(total_loss.item())
            base_weight_sum += float(base_loss.item())
            weighted_consistency_sum += float((warmup_weight * consistency_term).item())
            recon_sum += float(base_terms["recon_loss"].item())
            vq_sum += float(base_terms["vq_loss"].item())
            vel_sum += float(base_terms["vel_loss"].item())
            acc_sum += float(base_terms["acc_loss"].item())
            kin_smooth_sum += float(base_terms["kin_smooth_loss"].item())
            turn_global_sum += float(base_terms["turn_global_loss"].item())
            turn_yaw_sum += float(base_terms["turn_yaw_loss"].item())
            latent_sum += float(latent_loss.item())
            soft_sum += float(soft_loss.item())
            supcon_sum += float(supcon_loss.item())
            hard_code_sum += float(hard_code_loss.item())
            n_batches += 1

        scheduler.step()

        denom = max(1, n_batches)
        avg_recon = recon_sum / denom
        avg_vq = vq_sum / denom
        avg_vel = vel_sum / denom
        avg_acc = acc_sum / denom
        avg_kin_smooth = kin_smooth_sum / denom
        avg_turn_global = turn_global_sum / denom
        avg_turn_yaw = turn_yaw_sum / denom
        kin_smooth_weight = 1e-2 if epoch > 30 else 0.0
        row = {
            "epoch": int(epoch + 1),
            "total_loss": total_loss_sum / denom,
            "base_weight_loss": base_weight_sum / denom,
            "weighted_consistency_loss": weighted_consistency_sum / denom,
            "recon_loss": avg_recon,
            "vq_loss": avg_vq,
            "vel_loss": avg_vel,
            "acc_loss": avg_acc,
            "kin_smooth_loss": avg_kin_smooth,
            "turn_global_loss": avg_turn_global,
            "turn_yaw_loss": avg_turn_yaw,
            "weight_recon": 10.0 * avg_recon,
            "weight_vq": 5.0 * avg_vq,
            "weight_vel": 0.5 * avg_vel,
            "weight_acc": 0.05 * avg_acc,
            "weight_kin_smooth": kin_smooth_weight * avg_kin_smooth,
            "weight_turn_global": avg_turn_global,
            "weight_turn_yaw": 2.0 * avg_turn_yaw,
            "latent_consistency_loss": latent_sum / denom,
            "soft_code_consistency_loss": soft_sum / denom,
            "supcon_loss": supcon_sum / denom,
            "hard_code_consistency_loss": hard_code_sum / denom,
            "weight_hard_code_consistency": float(lambda_hard_code_consistency) * (hard_code_sum / denom),
            "warmup_weight": float(warmup_weight),
        }
        history.append(row)

        writer.add_scalar("loss/total", row["total_loss"], epoch + 1)
        writer.add_scalar("loss/recon", row["recon_loss"], epoch + 1)
        writer.add_scalar("loss/vq", row["vq_loss"], epoch + 1)
        writer.add_scalar("loss/weight_recon", row["weight_recon"], epoch + 1)
        writer.add_scalar("loss/weight_vq", row["weight_vq"], epoch + 1)
        writer.add_scalar("loss/weight_vel", row["weight_vel"], epoch + 1)
        writer.add_scalar("loss/weight_acc", row["weight_acc"], epoch + 1)
        writer.add_scalar("loss/weight_kin_smooth", row["weight_kin_smooth"], epoch + 1)
        writer.add_scalar("loss/weight_turn_global", row["weight_turn_global"], epoch + 1)
        writer.add_scalar("loss/weight_turn_yaw", row["weight_turn_yaw"], epoch + 1)
        writer.add_scalar("loss/weight", row["base_weight_loss"], epoch + 1)
        writer.add_scalar("loss/weight_consistency", row["weighted_consistency_loss"], epoch + 1)
        writer.add_scalar("loss/latent_consistency", row["latent_consistency_loss"], epoch + 1)
        writer.add_scalar("loss/soft_code_consistency", row["soft_code_consistency_loss"], epoch + 1)
        writer.add_scalar("loss/supcon", row["supcon_loss"], epoch + 1)
        writer.add_scalar("loss/hard_code_consistency", row["hard_code_consistency_loss"], epoch + 1)
        writer.add_scalar("loss/weight_hard_code_consistency", row["weight_hard_code_consistency"], epoch + 1)
        writer.add_scalar("loss/warmup_weight", row["warmup_weight"], epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(
                f"[SimilarConsistency] Epoch {epoch+1:03d} | total={row['total_loss']:.5f} | "
                f"weight={row['base_weight_loss']:.5f} | consistency_w={row['weighted_consistency_loss']:.5f} | "
                f"recon={row['recon_loss']:.5f} | vq={row['vq_loss']:.5f} | "
                f"latent={row['latent_consistency_loss']:.5f} | "
                f"soft={row['soft_code_consistency_loss']:.5f} | "
                f"supcon={row['supcon_loss']:.5f} | hard={row['hard_code_consistency_loss']:.5f} | "
                f"warmup={row['warmup_weight']:.3f}"
            )

    torch.save(model.state_dict(), os.path.join(save_dir, f"{data_type}_rvq_taae_model.pth"))
    writer.close()
    print(f"Similar-consistency Training Done. Model saved to {save_dir}")
    return history


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

    norm_params = load_norm_params(norm_path, device)
    model.set_norm_params(norm_params["mean"], norm_params["std"], norm_params["scale_factor"])
    model.eval()
    return model, norm_params, model_path


def _encode_codes_for_indices(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    indices: np.ndarray,
    norm_params: Dict[str, torch.Tensor],
    batch_size: int,
) -> np.ndarray:
    subset = np.asarray(trajs[indices], dtype=np.float32)
    x_norm = normalize_trajs(
        subset,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
    )
    loader = DataLoader(
        TensorDataset(x_norm),
        batch_size=min(max(1, int(batch_size)), max(1, int(x_norm.shape[0]))),
        shuffle=False,
    )
    all_codes: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(norm_params["mean"].device)
            z = model.encode(x)
            _, _, codes = model.rvq(z)
            all_codes.append(codes.cpu().numpy())
    return np.concatenate(all_codes, axis=0).astype(np.int64)


def evaluate_group_level_similar_or(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    group_id_per_sample: np.ndarray,
    representative_indices: np.ndarray,
    group_to_indices: List[np.ndarray],
    norm_params: Dict[str, torch.Tensor],
    max_groups_eval: int,
    max_members_per_group: int,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    """按 group 评估 Similar-Traj OR（核心指标）。"""
    group_sizes = np.asarray([idxs.size for idxs in group_to_indices], dtype=np.int64)
    valid_groups = np.where(group_sizes >= 2)[0]
    if valid_groups.size == 0:
        return {
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }

    rng = np.random.default_rng(seed)
    max_g = min(int(max_groups_eval), int(valid_groups.size))

    # 覆盖大小分布：按 group size 排序后均匀抽取。
    ordered = valid_groups[np.argsort(group_sizes[valid_groups])]
    if ordered.size <= max_g:
        chosen_groups = ordered
    else:
        pick = np.linspace(0, ordered.size - 1, max_g).round().astype(np.int64)
        chosen_groups = ordered[pick]
        rng.shuffle(chosen_groups)

    eval_indices = []
    group_entries: List[Dict[str, object]] = []
    for gid in chosen_groups.tolist():
        members = group_to_indices[int(gid)]
        anchor = int(representative_indices[int(gid)])
        others = members[members != anchor]
        if others.size == 0:
            continue
        take = min(int(max_members_per_group), int(others.size))
        # 采样组内成员（不包含 anchor）。
        if take < others.size:
            chosen_members = rng.choice(others, size=take, replace=False)
        else:
            chosen_members = others

        local = np.concatenate([[anchor], chosen_members.astype(np.int64)], axis=0)
        start = len(eval_indices)
        eval_indices.extend(local.tolist())
        end = len(eval_indices)
        group_entries.append(
            {
                "group_id": int(gid),
                "group_size": int(members.size),
                "anchor_idx": int(anchor),
                "slice": (start, end),
            }
        )

    if len(eval_indices) == 0:
        return {
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }

    eval_indices_np = np.asarray(eval_indices, dtype=np.int64)
    codes_all = _encode_codes_for_indices(
        model=model,
        trajs=trajs,
        indices=eval_indices_np,
        norm_params=norm_params,
        batch_size=batch_size,
    )

    per_group_or = []
    layer_or_collect = []
    or1_collect = []
    or3_collect = []
    orall_collect = []
    exact_collect = []

    for g in group_entries:
        s, e = g["slice"]
        codes = codes_all[s:e]
        if codes.shape[0] <= 1:
            continue
        anchor = codes[0]
        members = codes[1:]
        same = (members == anchor[None, :]).astype(np.float32)
        layer_or = same.mean(axis=0) * 100.0
        num_layers = int(layer_or.shape[0])

        or1 = float(layer_or[0]) if num_layers >= 1 else 0.0
        or3 = float(np.mean(layer_or[: min(3, num_layers)])) if num_layers > 0 else 0.0
        orall = float(np.mean(layer_or)) if num_layers > 0 else 0.0
        exact = float((same.sum(axis=1) == num_layers).mean() * 100.0)

        or1_collect.append(or1)
        or3_collect.append(or3)
        orall_collect.append(orall)
        exact_collect.append(exact)
        layer_or_collect.append(layer_or)

        per_group_or.append(
            {
                "group_id": int(g["group_id"]),
                "group_size": int(g["group_size"]),
                "anchor_idx": int(g["anchor_idx"]),
                "num_members_eval": int(members.shape[0]),
                "or_at_1": or1,
                "or_at_3": or3,
                "or_at_all": orall,
                "exact_all_layers_match_pct": exact,
                "layer_or": [float(v) for v in layer_or.tolist()],
            }
        )

    if not per_group_or:
        return {
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }

    layer_or_mean = np.mean(np.stack(layer_or_collect, axis=0), axis=0)
    return {
        "num_groups_evaluated": int(len(per_group_or)),
        "or_at_1": float(np.mean(or1_collect)),
        "or_at_3": float(np.mean(or3_collect)),
        "or_at_all": float(np.mean(orall_collect)),
        "exact_all_layers_match_pct": float(np.mean(exact_collect)),
        "layer_or_mean": [float(v) for v in layer_or_mean.tolist()],
        "per_group_or": per_group_or,
    }


def evaluate_selected_group_or(
    model: TrajRVQTransformer,
    trajs: np.ndarray,
    representative_indices: np.ndarray,
    group_to_indices: List[np.ndarray],
    norm_params: Dict[str, torch.Tensor],
    group_size: int,
    batch_size: int,
    seed: int,
) -> Optional[Dict[str, object]]:
    """单个诊断 group 的 OR（可视化/排障用，不是主指标）。"""
    rng = np.random.default_rng(seed)
    group_sizes = np.asarray([idxs.size for idxs in group_to_indices], dtype=np.int64)
    candidates = np.where(group_sizes >= 2)[0]
    if candidates.size == 0:
        return None

    # 优先找 size 足够大的组，否则选最大组。
    enough = candidates[group_sizes[candidates] >= max(2, int(group_size))]
    if enough.size > 0:
        gid = int(enough[np.argmax(group_sizes[enough])])
    else:
        gid = int(candidates[np.argmax(group_sizes[candidates])])

    members = group_to_indices[gid]
    anchor = int(representative_indices[gid])
    others = members[members != anchor]
    if others.size == 0:
        return None

    take = min(max(1, int(group_size) - 1), int(others.size))
    chosen = rng.choice(others, size=take, replace=False) if take < others.size else others
    eval_indices = np.concatenate([[anchor], chosen.astype(np.int64)], axis=0)

    codes = _encode_codes_for_indices(
        model=model,
        trajs=trajs,
        indices=eval_indices,
        norm_params=norm_params,
        batch_size=batch_size,
    )
    anchor_codes = codes[0]
    member_codes = codes[1:]
    same = (member_codes == anchor_codes[None, :]).astype(np.float32)
    layer_or = same.mean(axis=0) * 100.0
    l = int(layer_or.shape[0])

    return {
        "group_id": int(gid),
        "group_size": int(members.size),
        "anchor_idx": int(anchor),
        "sample_indices": [int(v) for v in eval_indices.tolist()],
        "or_at_1": float(layer_or[0]) if l >= 1 else 0.0,
        "or_at_3": float(np.mean(layer_or[: min(3, l)])) if l > 0 else 0.0,
        "or_at_all": float(np.mean(layer_or)) if l > 0 else 0.0,
        "exact_all_layers_match_pct": float((same.sum(axis=1) == l).mean() * 100.0),
        "layer_or": [float(v) for v in layer_or.tolist()],
        "codes": [[int(x) for x in row.tolist()] for row in codes],
    }


def save_random_group_visualizations(
    trajs: np.ndarray,
    representative_indices: np.ndarray,
    group_to_indices: List[np.ndarray],
    save_dir: str,
    num_groups: int,
    max_members_per_group: int,
    seed: int,
) -> Dict[str, object]:
    """随机抽取若干 group 可视化，检查分组质量。"""
    if plt is None:
        print("[warning] matplotlib 不可用，跳过 group 可视化。")
        return {"enabled": False, "reason": "matplotlib_not_available", "saved_count": 0, "vis_dir": save_dir}

    os.makedirs(save_dir, exist_ok=True)
    group_sizes = np.asarray([idxs.size for idxs in group_to_indices], dtype=np.int64)
    valid = np.where(group_sizes >= 2)[0]
    if valid.size == 0 or int(num_groups) <= 0:
        return {"enabled": False, "reason": "no_valid_groups", "saved_count": 0, "vis_dir": save_dir}

    rng = np.random.default_rng(seed)
    take_g = min(int(num_groups), int(valid.size))
    chosen_gids = rng.choice(valid, size=take_g, replace=False)
    chosen_gids = np.sort(chosen_gids.astype(np.int64))

    rows: List[Dict[str, object]] = []
    saved_count = 0
    for k, gid in enumerate(chosen_gids.tolist(), start=1):
        members = group_to_indices[int(gid)]
        anchor_idx = int(representative_indices[int(gid)])

        # 每组最多画 max_members_per_group 条，且包含 representative。
        if members.size > int(max_members_per_group):
            others = members[members != anchor_idx]
            take_other = max(0, int(max_members_per_group) - 1)
            if take_other > 0 and others.size > 0:
                chosen_other = rng.choice(others, size=min(take_other, int(others.size)), replace=False)
                vis_indices = np.concatenate([[anchor_idx], chosen_other.astype(np.int64)], axis=0)
            else:
                vis_indices = np.asarray([anchor_idx], dtype=np.int64)
        else:
            vis_indices = members.astype(np.int64)
            if anchor_idx not in vis_indices:
                vis_indices = np.concatenate([[anchor_idx], vis_indices], axis=0)
        vis_indices = np.unique(vis_indices).astype(np.int64)

        clips = np.asarray(trajs[vis_indices], dtype=np.float32)
        xy_global = integrate_to_global(clips)

        fig = plt.figure(figsize=(7, 6), dpi=140)
        ax = fig.add_subplot(111)
        for i in range(xy_global.shape[0]):
            x = xy_global[i, :, 0]
            y = xy_global[i, :, 1]
            idx_global = int(vis_indices[i])
            if idx_global == anchor_idx:
                ax.plot(x, y, "-", linewidth=2.2, label=f"anchor:{idx_global}")
                ax.scatter([x[0]], [y[0]], s=24, marker="o")
            else:
                ax.plot(x, y, "-", linewidth=1.0, alpha=0.75)

        ax.set_title(f"group_{k:02d}_gid{gid}_size{int(members.size)}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8)

        fname = f"group_{k:02d}_gid{gid}_size{int(members.size)}.png"
        fpath = os.path.join(save_dir, fname)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)

        saved_count += 1
        rows.append(
            {
                "order": int(k),
                "group_id": int(gid),
                "group_size": int(members.size),
                "anchor_idx": int(anchor_idx),
                "num_members_visualized": int(vis_indices.size),
                "figure_path": fpath,
                "sample_indices": ",".join([str(int(v)) for v in vis_indices.tolist()]),
            }
        )

    _write_csv(rows, os.path.join(save_dir, "group_visualization_samples.csv"))
    print(f"[group-vis] saved {saved_count} figures to: {save_dir}")
    return {"enabled": True, "saved_count": int(saved_count), "vis_dir": save_dir}


def _print_dataset_summary(stats: Dict[str, np.ndarray], num_steps: int) -> None:
    raw_count = int(stats["total_path_length"].shape[0])
    stationary_like_count = int(np.sum(stats["stationary_like"]))
    stationary_like_ratio = float(stationary_like_count / max(1, raw_count))

    print("=" * 80)
    print("Dataset Summary")
    print(f"raw_count: {raw_count}")
    print(f"stationary_like_count: {stationary_like_count}")
    print(f"stationary_like_ratio: {stationary_like_ratio:.6f}")
    print(f"trajectory_length: {num_steps}")
    print("=" * 80)


def _print_grouping_summary(
    grouping_method: str,
    grouping_stage: str,
    group_feature: str,
    requested_num_groups: int,
    actual_num_groups: int,
    representative_count: int,
    group_sizes: np.ndarray,
) -> None:
    p = _percentiles(group_sizes.astype(np.float32))
    print("Grouping Summary")
    print(f"grouping_method: {grouping_method}")
    print(f"grouping_stage: {grouping_stage}")
    print(f"group_feature: {group_feature}")
    print(f"requested_num_groups: {requested_num_groups}")
    print(f"actual_num_groups: {actual_num_groups}")
    print(f"representative_count: {representative_count}")
    print(
        "group_size p0/p25/p50/p75/p95/p99/max: "
        f"{p['p0']:.2f}/{p['p25']:.2f}/{p['p50']:.2f}/{p['p75']:.2f}/{p['p95']:.2f}/{p['p99']:.2f}/{p['max']:.2f}"
    )
    print("=" * 80)


def _print_training_summary(
    experiment_mode: str,
    train_unique_count: int,
    train_data_source: str,
    epochs: int,
    batch_size: int,
) -> None:
    print("Training Summary")
    print(f"experiment_mode: {experiment_mode}")
    print(f"train_unique_count: {train_unique_count}")
    print(f"train_data_source: {train_data_source}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative/full/consistency tokenizer experiments.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])

    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="representative_group",
        choices=["representative_group", "full_data", "similar_consistency"],
    )

    parser.add_argument("--num-representative-groups", type=int, default=80000)
    parser.add_argument(
        "--grouping-method",
        type=str,
        default="minibatch_kmeans",
        choices=["minibatch_kmeans", "kinematic_bins"],
    )
    parser.add_argument(
        "--grouping-stage",
        type=str,
        default="scenario_first",
        choices=["scenario_first", "global"],
        help="scenario_first: 先按场景切分再场景内分组；global: 全量直接分组。",
    )
    parser.add_argument(
        "--group-feature",
        type=str,
        default="kinematic_plus_shape",
        choices=["kinematic", "shape", "kinematic_plus_shape"],
    )
    parser.add_argument("--shape-downsample-steps", type=int, default=10)
    parser.add_argument("--feature-xy-weight", type=float, default=1.0)
    parser.add_argument("--feature-yaw-weight", type=float, default=3.0)

    parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=1) # 为了结果稳定，建议设置较大值（如 10 或 20），但会增加计算成本。默认值 1 是为了快速测试。
    parser.add_argument("--kmeans-random-state", type=int, default=42)

    parser.add_argument("--full-train-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--consistency-train-base",
        type=str,
        default="representative_group",
        choices=["representative_group", "full_data"],
    )
    parser.add_argument("--positive-groups-per-step", type=int, default=8)
    parser.add_argument("--positives-per-group", type=int, default=4)
    parser.add_argument("--negative-groups-per-step", type=int, default=16)

    parser.add_argument("--lambda-latent-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-soft-code-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-supcon", type=float, default=0.05)
    parser.add_argument("--lambda-hard-code-consistency", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--soft-code-temperature", type=float, default=0.2)
    parser.add_argument("--hard-code-temperature", type=float, default=0.2)
    parser.add_argument("--hard-code-consistency-layers", type=int, default=1)
    parser.add_argument("--consistency-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--consistency-target",
        type=str,
        default="latent_plus_soft_code",
        choices=["latent", "soft_primary_code", "latent_plus_soft_code"],
    )

    parser.add_argument("--report-similar-or", action="store_true")
    parser.add_argument("--report-noise-or", action="store_true")
    parser.add_argument("--noise-std-xy", type=float, default=0.05)
    parser.add_argument("--noise-std-yaw", type=float, default=0.01)

    parser.add_argument("--max-groups-eval", type=int, default=1000)
    parser.add_argument("--max-members-per-group", type=int, default=8)

    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--report-selected-group-or", action="store_true")
    parser.add_argument("--num-group-visualizations", type=int, default=0)
    parser.add_argument("--max-members-per-group-vis", type=int, default=12)

    parser.add_argument("--save-root", type=str, default="./work_dirs/tokenizer/similar_single_train")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--grouping-cache-dir", type=str, default=None)
    parser.add_argument("--disable-grouping-cache", action="store_true")
    parser.add_argument("--grouping-cache-key", type=str, default="")

    args = parser.parse_args()

    _set_seed(int(args.seed))

    run_name = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(args.save_root, f"{args.data_type}_{args.experiment_mode}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    trajs = load_sampled_datas(args.data_path)
    if args.data_type == "history":
        trajs = trajs[:, :14, :]
    trajs = np.asarray(trajs, dtype=np.float32)

    n_total = int(trajs.shape[0])
    requested_num_groups = int(max(1, min(args.num_representative_groups, n_total)))

    motion_stats = compute_motion_stats(trajs, dt=float(args.dt))
    _print_dataset_summary(motion_stats, num_steps=int(trajs.shape[1]))

    grouping = None
    grouping_loaded_from_cache = False

    # 分组 cache：优先读 output_dir 下已有结果；其次读全局 grouping_cache。
    cache_key = str(args.grouping_cache_key).strip()
    if not cache_key:
        cache_key = build_grouping_cache_key(
            data_path=args.data_path,
            data_type=args.data_type,
            n_total=n_total,
            num_steps=int(trajs.shape[1]),
            requested_num_groups=requested_num_groups,
            grouping_method=args.grouping_method,
            grouping_stage=args.grouping_stage,
            group_feature=args.group_feature,
            shape_downsample_steps=int(args.shape_downsample_steps),
            feature_xy_weight=float(args.feature_xy_weight),
            feature_yaw_weight=float(args.feature_yaw_weight),
            dt=float(args.dt),
            kmeans_batch_size=int(args.kmeans_batch_size),
            kmeans_max_iter=int(args.kmeans_max_iter),
            kmeans_n_init=int(args.kmeans_n_init),
            kmeans_random_state=int(args.kmeans_random_state),
            seed=int(args.seed),
        )

    grouping_cache_root = args.grouping_cache_dir or os.path.join(args.save_root, "grouping_cache")
    grouping_cache_dir = os.path.join(grouping_cache_root, cache_key)

    if not args.disable_grouping_cache:
        grouping = try_load_grouping_from_cache(output_dir, n_total=n_total)
        if grouping is not None:
            grouping_loaded_from_cache = True
            print(f"[cache] Loaded grouping from output_dir: {output_dir}")
        else:
            grouping = try_load_grouping_from_cache(grouping_cache_dir, n_total=n_total)
            if grouping is not None:
                grouping_loaded_from_cache = True
                print(f"[cache] Loaded grouping from cache_dir: {grouping_cache_dir}")

    if grouping is None:
        # 分组核心流程：按配置选择全局分组或“先场景后分组”。
        grouping = find_representative_groups(
            trajs=trajs,
            dt=float(args.dt),
            num_groups=requested_num_groups,
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
            seed=int(args.seed),
            motion_stats=motion_stats,
        )

        if not args.disable_grouping_cache:
            cache_meta = {
                "grouping_method_used": grouping["grouping_method_used"],
                "grouping_stage_used": grouping.get("grouping_stage_used", args.grouping_stage),
                "grouping_method_requested": args.grouping_method,
                "grouping_stage_requested": args.grouping_stage,
                "group_feature": args.group_feature,
                "cache_key": cache_key,
                "raw_count": int(n_total),
                "num_steps": int(trajs.shape[1]),
                "requested_num_groups": int(requested_num_groups),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_grouping_cache(output_dir, grouping, meta=cache_meta)
            save_grouping_cache(grouping_cache_dir, grouping, meta=cache_meta)
            print(f"[cache] Saved grouping cache to: {grouping_cache_dir}")
    else:
        # 缓存命中时也同步一份到当前 output_dir，保证本次实验目录完整。
        if not args.disable_grouping_cache:
            cache_meta = {
                "grouping_method_used": grouping["grouping_method_used"],
                "grouping_stage_used": grouping.get("grouping_stage_used", "cache"),
                "grouping_method_requested": args.grouping_method,
                "grouping_stage_requested": args.grouping_stage,
                "group_feature": args.group_feature,
                "cache_key": cache_key,
                "raw_count": int(n_total),
                "num_steps": int(trajs.shape[1]),
                "requested_num_groups": int(requested_num_groups),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "loaded_from_cache": True,
            }
            save_grouping_cache(output_dir, grouping, meta=cache_meta)

    group_id_per_sample = np.asarray(grouping["group_id_per_sample"], dtype=np.int32)
    representative_indices = np.asarray(grouping["representative_indices"], dtype=np.int64)
    group_sizes = np.asarray(grouping["group_sizes"], dtype=np.int64)
    rep_dist = np.asarray(grouping["representative_distance_to_center"], dtype=np.float32)
    group_to_indices: List[np.ndarray] = grouping["group_to_indices"]
    actual_num_groups = int(grouping["actual_num_groups"])
    grouping_method_used = str(grouping["grouping_method_used"])
    grouping_stage_used = str(grouping.get("grouping_stage_used", args.grouping_stage))
    scenario_partition_summary = grouping.get("scenario_partition_summary", [])

    if representative_indices.shape[0] < int(0.8 * requested_num_groups):
        print(
            "[warning] representative_count 低于 requested_num_groups 的 80%，"
            f"requested={requested_num_groups}, actual={representative_indices.shape[0]}"
        )

    _print_grouping_summary(
        grouping_method=grouping_method_used,
        grouping_stage=grouping_stage_used,
        group_feature=args.group_feature,
        requested_num_groups=requested_num_groups,
        actual_num_groups=actual_num_groups,
        representative_count=int(representative_indices.shape[0]),
        group_sizes=group_sizes,
    )

    representative_indices_path = os.path.join(output_dir, "representative_indices.npy")
    group_id_per_sample_path = os.path.join(output_dir, "group_id_per_sample.npy")
    group_sizes_path = os.path.join(output_dir, "group_sizes.npy")

    np.save(representative_indices_path, representative_indices)
    np.save(group_id_per_sample_path, group_id_per_sample)
    np.save(group_sizes_path, group_sizes)

    group_summary = {
        "grouping_method": grouping_method_used,
        "grouping_stage": grouping_stage_used,
        "group_feature": args.group_feature,
        "requested_num_groups": int(requested_num_groups),
        "actual_num_groups": int(actual_num_groups),
        "representative_count": int(representative_indices.shape[0]),
        "loaded_from_cache": bool(grouping_loaded_from_cache),
        "cache_key": cache_key,
        "grouping_cache_dir": grouping_cache_dir,
        "scenario_partition_summary": scenario_partition_summary,
        "group_size_percentiles": _percentiles(group_sizes.astype(np.float32)),
        "representative_distance_to_center_percentiles": _percentiles(rep_dist.astype(np.float32)),
        "representative_indices_path": representative_indices_path,
        "group_id_per_sample_path": group_id_per_sample_path,
        "group_sizes_path": group_sizes_path,
    }

    group_vis_info = None
    if int(args.num_group_visualizations) > 0:
        group_vis_dir = os.path.join(output_dir, "group_visualizations")
        group_vis_info = save_random_group_visualizations(
            trajs=trajs,
            representative_indices=representative_indices,
            group_to_indices=group_to_indices,
            save_dir=group_vis_dir,
            num_groups=int(args.num_group_visualizations),
            max_members_per_group=int(args.max_members_per_group_vis),
            seed=int(args.seed),
        )
        group_summary["group_visualization"] = group_vis_info

    _write_json(os.path.join(output_dir, "group_summary.json"), group_summary)

    if args.experiment_mode == "representative_group":
        train_indices = representative_indices.copy()
        train_trajs = trajs[train_indices]
        train_data_source = "representative_group"
        train_batch_size = int(args.batch_size)
    elif args.experiment_mode == "full_data":
        train_indices = np.arange(n_total, dtype=np.int64)
        train_trajs = trajs
        train_data_source = "full_data"
        train_batch_size = int(args.full_train_batch_size)
    elif args.experiment_mode == "similar_consistency":
        if args.consistency_train_base == "representative_group":
            train_indices = representative_indices.copy()
            train_trajs = trajs[train_indices]
            train_data_source = "representative_group"
            train_batch_size = int(args.batch_size)
        else:
            train_indices = np.arange(n_total, dtype=np.int64)
            train_trajs = trajs
            train_data_source = "full_data"
            train_batch_size = int(args.full_train_batch_size)
    else:
        raise ValueError(f"Unsupported experiment_mode: {args.experiment_mode}")

    train_indices_path = os.path.join(output_dir, "train_indices.npy")
    np.save(train_indices_path, train_indices)

    _print_training_summary(
        experiment_mode=args.experiment_mode,
        train_unique_count=int(train_indices.shape[0]),
        train_data_source=train_data_source,
        epochs=int(args.epochs),
        batch_size=int(train_batch_size),
    )

    robust_training_losses = None
    if args.experiment_mode in ("representative_group", "full_data"):
        train_rvq_taae(
            data_array=train_trajs,
            save_dir=output_dir,
            data_type=args.data_type,
            batch_size=train_batch_size,
            num_layers=int(args.num_layers),
            num_transformer_layers=int(args.num_transformer_layers),
            epochs=int(args.epochs),
        )
    else:
        robust_training_losses = train_rvq_taae_with_group_consistency(
            base_train_trajs=train_trajs,
            all_trajs=trajs,
            group_id_per_sample=group_id_per_sample,
            group_to_indices=group_to_indices,
            save_dir=output_dir,
            data_type=args.data_type,
            batch_size=train_batch_size,
            num_layers=int(args.num_layers),
            num_transformer_layers=int(args.num_transformer_layers),
            epochs=int(args.epochs),
            positive_groups_per_step=int(args.positive_groups_per_step),
            positives_per_group=int(args.positives_per_group),
            negative_groups_per_step=int(args.negative_groups_per_step),
            lambda_latent_consistency=float(args.lambda_latent_consistency),
            lambda_soft_code_consistency=float(args.lambda_soft_code_consistency),
            lambda_supcon=float(args.lambda_supcon),
            lambda_hard_code_consistency=float(args.lambda_hard_code_consistency),
            contrastive_temperature=float(args.contrastive_temperature),
            soft_code_temperature=float(args.soft_code_temperature),
            hard_code_temperature=float(args.hard_code_temperature),
            consistency_warmup_epochs=int(args.consistency_warmup_epochs),
            consistency_target=args.consistency_target,
            hard_code_consistency_layers=int(args.hard_code_consistency_layers),
            seed=int(args.seed),
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

    # reconstruction 指标：在训练数据上评估。
    recon_trajs, _ = reconstruct_trajs(
        model=model,
        trajs=train_trajs,
        mean=norm_params["mean"],
        std=norm_params["std"],
        scale_factor=norm_params["scale_factor"],
        clip_limit=norm_params["clip_limit"],
        batch_size=max(1, int(args.eval_batch_size)),
    )
    reconstruction_metrics = scenario_metrics(train_trajs, recon_trajs, dt=float(args.dt))

    group_level_similar_or = None
    group_level_similar_or_path = os.path.join(output_dir, "group_level_similar_or_summary.json")
    if args.report_similar_or:
        group_level_similar_or = evaluate_group_level_similar_or(
            model=model,
            trajs=trajs,
            group_id_per_sample=group_id_per_sample,
            representative_indices=representative_indices,
            group_to_indices=group_to_indices,
            norm_params=norm_params,
            max_groups_eval=int(args.max_groups_eval),
            max_members_per_group=int(args.max_members_per_group),
            batch_size=max(1, int(args.eval_batch_size)),
            seed=int(args.seed),
        )
    else:
        group_level_similar_or = {
            "enabled": False,
            "num_groups_evaluated": 0,
            "or_at_1": 0.0,
            "or_at_3": 0.0,
            "or_at_all": 0.0,
            "exact_all_layers_match_pct": 0.0,
            "layer_or_mean": [],
            "per_group_or": [],
        }
    _write_json(group_level_similar_or_path, group_level_similar_or)

    selected_group_or = None
    if args.report_selected_group_or:
        selected_group_or = evaluate_selected_group_or(
            model=model,
            trajs=trajs,
            representative_indices=representative_indices,
            group_to_indices=group_to_indices,
            norm_params=norm_params,
            group_size=int(args.group_size),
            batch_size=max(1, int(args.eval_batch_size)),
            seed=int(args.seed),
        )

    tokenizer_health = None
    if args.report_noise_or:
        eval_norm = normalize_trajs(
            train_trajs,
            mean=norm_params["mean"],
            std=norm_params["std"],
            scale_factor=norm_params["scale_factor"],
            clip_limit=norm_params["clip_limit"],
        )
        eval_loader = DataLoader(
            TensorDataset(eval_norm),
            batch_size=min(max(1, int(args.eval_batch_size)), max(1, int(train_trajs.shape[0]))),
            shuffle=False,
        )
        util, overlap = evaluate_tokenizer_health(
            model=model,
            dataloader=eval_loader,
            device=device,
            noise_std_xy=float(args.noise_std_xy),
            noise_std_yaw=float(args.noise_std_yaw),
            clip_limit=norm_params["clip_limit"],
            vis_dt=float(args.dt),
        )
        tokenizer_health = {
            "avg_codebook_utilization_pct": float(util),
            "avg_overlap_rate_pct": float(overlap),
            "noise_std_xy": float(args.noise_std_xy),
            "noise_std_yaw": float(args.noise_std_yaw),
        }

    if robust_training_losses is not None:
        _write_csv(robust_training_losses, os.path.join(output_dir, "train_loss_history.csv"))

    dataset_summary = {
        "raw_count": int(n_total),
        "stationary_like_count": int(np.sum(motion_stats["stationary_like"])),
        "stationary_like_ratio": float(np.mean(motion_stats["stationary_like"].astype(np.float32))),
        "motion_stats_percentiles": {
            "total_path_length": _percentiles(motion_stats["total_path_length"]),
            "mean_speed": _percentiles(motion_stats["mean_speed"]),
            "abs_yaw_sum": _percentiles(motion_stats["abs_yaw_sum"]),
        },
    }

    training_summary = {
        "experiment_mode": args.experiment_mode,
        "train_unique_count": int(train_indices.shape[0]),
        "train_indices_path": train_indices_path,
        "train_data_source": train_data_source,
        "epochs": int(args.epochs),
        "batch_size": int(train_batch_size),
        "consistency_train_base": args.consistency_train_base,
        "positive_groups_per_step": int(args.positive_groups_per_step),
        "positives_per_group": int(args.positives_per_group),
        "negative_groups_per_step": int(args.negative_groups_per_step),
        "lambda_latent_consistency": float(args.lambda_latent_consistency),
        "lambda_soft_code_consistency": float(args.lambda_soft_code_consistency),
        "lambda_supcon": float(args.lambda_supcon),
        "lambda_hard_code_consistency": float(args.lambda_hard_code_consistency),
        "contrastive_temperature": float(args.contrastive_temperature),
        "soft_code_temperature": float(args.soft_code_temperature),
        "hard_code_temperature": float(args.hard_code_temperature),
        "consistency_warmup_epochs": int(args.consistency_warmup_epochs),
        "consistency_target": args.consistency_target,
        "hard_code_consistency_layers": int(args.hard_code_consistency_layers),
    }

    summary = {
        "dataset": dataset_summary,
        "grouping": group_summary,
        "training": training_summary,
        "metrics": {
            "reconstruction_metrics": reconstruction_metrics,
            "group_level_similar_or": group_level_similar_or,
            "selected_group_or": selected_group_or,
            "tokenizer_health": tokenizer_health,
            "robust_training_losses": robust_training_losses,
        },
    }

    _write_json(os.path.join(output_dir, "dataset_training_summary.json"), {
        "dataset": dataset_summary,
        "training": training_summary,
    })
    summary_path = os.path.join(output_dir, "similar_single_train_summary.json")
    _write_json(summary_path, summary)

    print("=" * 80)
    print("Experiment Done")
    print(f"model_path: {model_path}")
    print(f"summary_path: {summary_path}")

    if group_level_similar_or is not None:
        print("Group-level Similar-Traj OR:")
        print(f"  OR@1: {float(group_level_similar_or.get('or_at_1', 0.0)):.4f}")
        print(f"  OR@3: {float(group_level_similar_or.get('or_at_3', 0.0)):.4f}")
        print(f"  OR@All: {float(group_level_similar_or.get('or_at_all', 0.0)):.4f}")
        print(f"  exact_all_layers_match_pct: {float(group_level_similar_or.get('exact_all_layers_match_pct', 0.0)):.4f}")

    if tokenizer_health is not None:
        print("Noise OR:")
        print(f"  avg_codebook_utilization_pct: {float(tokenizer_health['avg_codebook_utilization_pct']):.4f}")
        print(f"  avg_overlap_rate_pct: {float(tokenizer_health['avg_overlap_rate_pct']):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# python /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/experiment_similar_traj_single_train.py \
#   --experiment-mode similar_consistency \
#   --consistency-train-base representative_group \
#   --grouping-method minibatch_kmeans \
#   --grouping-stage scenario_first \
#   --group-feature kinematic_plus_shape \
#   --num-representative-groups 80000 \
#   --shape-downsample-steps 6 \
#   --kmeans-batch-size 100000 \
#   --kmeans-max-iter 20 \
#   --kmeans-n-init 1 \
#   --positive-groups-per-step 16 \
#   --positives-per-group 6 \
#   --negative-groups-per-step 8 \
#   --lambda-latent-consistency 0.1 \
#   --lambda-soft-code-consistency 0.1 \
#   --lambda-supcon 0.0 \
#   --lambda-hard-code-consistency 0.05 \
#   --contrastive-temperature 0.1 \
#   --soft-code-temperature 0.2 \
#   --hard-code-temperature 0.2 \
#   --hard-code-consistency-layers 1 \
#   --consistency-warmup-epochs 100 \
#   --consistency-target latent_plus_soft_code \
#   --num-layers 6 \
#   --epochs 500 \
#   --report-similar-or \
#   --report-noise-or \
#   --num-group-visualizations 6 \
#   --max-members-per-group-vis 12 \
#   --output-dir /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn/work_dirs/tokenizer/similar_consistency_repbase
