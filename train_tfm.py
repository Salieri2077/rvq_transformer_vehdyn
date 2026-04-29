import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from rvq_model import ResidualVQ
from utils import (
    preprocess_and_save_norm_params,
    load_sampled_datas,
    frequency_smoothness_loss,
    acceleration_smoothness_loss,
)
# from tokenizer.rvq.rvq_mlp.train import integrate_trajectory_keyframe_torch


def signed_velocity_loss_from_dxdydyaw(pred_u: torch.Tensor, gt_u: torch.Tensor, dt: float = 0.1):
    """
    基于 dxdydyaw 计算带符号速度并施加约束（Batch 版本）。

    思路：
      - dxdydyaw 中的 dx, dy 表示相对于上一帧 ego 坐标系的位移增量
      - vx = dx / dt 保留符号，dx < 0 表示倒车
      - vy = dy / dt 表示横向速度

    Args:
        pred_u: [B, T, 3] - 预测的 dxdydyaw
        gt_u:   [B, T, 3] - GT 的 dxdydyaw
        dt:     时间步长（秒），只影响速度的绝对数值，损失相对权重可通过 loss weight 调整

    Returns:
        标量 velocity_loss：预测/GT 带符号速度的 MSE 误差
    """
    assert pred_u.shape == gt_u.shape
    assert pred_u.shape[-1] == 3

    vx_pred = pred_u[:, :, 0] / dt
    vy_pred = pred_u[:, :, 1] / dt
    vx_gt = gt_u[:, :, 0] / dt
    vy_gt = gt_u[:, :, 1] / dt

    return F.mse_loss(vx_pred, vx_gt) + 0.2 * F.mse_loss(vy_pred, vy_gt)


def signed_acceleration_loss_from_dxdydyaw(pred_u: torch.Tensor, gt_u: torch.Tensor, dt: float = 0.1):
    """
    基于 dxdydyaw 计算带符号加速度并施加约束（Batch 版本）。

    思路：
      - 先由 dx, dy 计算带符号的 vx / vy
      - 再对 vx / vy 做一阶差分得到 ax / ay
    """
    assert pred_u.shape == gt_u.shape
    assert pred_u.shape[-1] == 3

    vx_pred = pred_u[:, :, 0] / dt
    vx_gt = gt_u[:, :, 0] / dt
    vy_pred = pred_u[:, :, 1] / dt
    vy_gt = gt_u[:, :, 1] / dt

    ax_pred = (vx_pred[:, 1:] - vx_pred[:, :-1]) / dt
    ax_gt = (vx_gt[:, 1:] - vx_gt[:, :-1]) / dt
    ay_pred = (vy_pred[:, 1:] - vy_pred[:, :-1]) / dt
    ay_gt = (vy_gt[:, 1:] - vy_gt[:, :-1]) / dt

    return F.mse_loss(ax_pred, ax_gt) + 0.2 * F.mse_loss(ay_pred, ay_gt)


def integrate_to_global_torch(trajs: torch.Tensor) -> torch.Tensor:
    dx = trajs[:, :, 0]
    dy = trajs[:, :, 1]
    dyaw = trajs[:, :, 2]
    yaw = torch.cumsum(dyaw, dim=1)
    prev_yaw = torch.zeros_like(yaw)
    prev_yaw[:, 1:] = yaw[:, :-1]

    dx_global = dx * torch.cos(prev_yaw) - dy * torch.sin(prev_yaw)
    dy_global = dx * torch.sin(prev_yaw) + dy * torch.cos(prev_yaw)

    x_global = torch.cumsum(dx_global, dim=1)
    y_global = torch.cumsum(dy_global, dim=1)
    return torch.stack([x_global, y_global], dim=-1)


def turn_global_yaw_loss(
    pred_phys: torch.Tensor,
    gt_phys: torch.Tensor,
    turn_threshold: float = 0.35,
):
    net_yaw = torch.sum(gt_phys[:, :, 2], dim=1)
    turn_mask = torch.abs(net_yaw) > turn_threshold
    if not torch.any(turn_mask):
        zero = pred_phys.sum() * 0.0
        return zero, zero, turn_mask

    pred_turn = pred_phys[turn_mask]
    gt_turn = gt_phys[turn_mask]

    pred_xy = integrate_to_global_torch(pred_turn)
    gt_xy = integrate_to_global_torch(gt_turn)
    global_xy_loss = F.mse_loss(pred_xy, gt_xy)

    pred_yaw = torch.cumsum(pred_turn[:, :, 2], dim=1)
    gt_yaw = torch.cumsum(gt_turn[:, :, 2], dim=1)
    cumulative_yaw_loss = F.mse_loss(pred_yaw, gt_yaw)
    return global_xy_loss, cumulative_yaw_loss, turn_mask


def vel_aug(trajs: np.ndarray, dt: float = 0.2, high_speed_threshold_kmh: float = 75.0) -> np.ndarray:
    """
    对轨迹做基于速度的重采样：
    - 对每条轨迹计算最大速度（基于相邻帧的位移增量）
    - 当最大速度超过 high_speed_threshold_kmh 时，将这条轨迹视为高速样本
    - 返回「原始数据 + 所有高速轨迹的拷贝」，实现对高速样本的过采样

    Args:
        trajs: [N, T, 3] numpy 数组，dxdydyaw（单位：米 / 弧度）
        dt: 时间间隔（秒），默认 0.2
        high_speed_threshold_kmh: 判定为高速轨迹的阈值（km/h）

    Returns:
        aug_trajs: [N + N_high, T, 3]，包含原始轨迹和高速轨迹的重复样本
    """
    assert trajs.ndim == 3 and trajs.shape[-1] == 3, "trajs should be [N, T, 3]"

    # 计算每条轨迹在每个时间步的速度（基于 dx, dy）
    dx = trajs[:, 1:, 0] - trajs[:, :-1, 0]  # [N, T-1]
    dy = trajs[:, 1:, 1] - trajs[:, :-1, 1]  # [N, T-1]

    # 速度 (m/s) -> km/h
    speed_mps = np.sqrt(dx * dx + dy * dy) / dt
    speed_kmh = speed_mps * 3.6  # [N, T-1]

    # 每条轨迹的最大速度
    max_speed_kmh = np.max(speed_kmh, axis=1)  # [N]

    # 找出高速轨迹
    high_speed_mask = max_speed_kmh > high_speed_threshold_kmh
    if not np.any(high_speed_mask):
        return trajs

    high_speed_trajs = trajs[high_speed_mask]  # [N_high, T, 3]

    # 原始数据 + 高速轨迹拷贝
    aug_trajs = np.concatenate([trajs, high_speed_trajs], axis=0)
    return aug_trajs

class ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, D, T]
        return x + self.norm(self.relu(self.conv(x)))


class TrajRVQTransformer(nn.Module):
    """
    运动学 RVQ Transformer：Encoder 不变，Decoder 用运动学公式 rollout。
    decoder 预测 v(速度) / κ(曲率) / dy(横向残差) profile，
    再通过 dx = v*dt, dyaw = v*κ*dt 恢复 dxdydyaw。
    """

    def __init__(
        self,
        input_steps: int = 25,
        input_dim: int = 3,
        num_layers: int = 10,
        vocab_size: int = 1024,
        d_model: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dt: float = 0.2,
    ):
        super().__init__()

        self.input_steps = input_steps
        self.input_dim = input_dim
        self.input_flat_dim = input_steps * input_dim
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dt = dt

        # --- Encoder: 展平轨迹后用 TransformerEncoder 处理 ---
        self.input_proj = nn.Linear(self.input_flat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # --- RVQ 瓶颈 ---
        self.rvq = ResidualVQ(
            num_quantizers=num_layers,
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            dropout=0.2,
            commitment_cost=0.25,
        )

        # --- Decoder: Transformer + 运动学 heads ---
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_transformer_layers
        )

        # 运动学参数 heads（输出物理空间的 v / κ / dy）
        self.v_head = nn.Linear(d_model, input_steps)
        self.kappa_head = nn.Linear(d_model, input_steps)
        self.dy_head = nn.Linear(d_model, input_steps)

        # v_head 偏置初始化为 ~10 m/s（典型行驶速度），加速收敛
        nn.init.constant_(self.v_head.bias, 10.0)

        # 归一化参数 buffers（通过 set_norm_params 设置，会随 state_dict 保存/加载）
        self.register_buffer('norm_mean', torch.zeros(1, 1, 3))
        self.register_buffer('norm_std', torch.ones(1, 1, 3))
        self.register_buffer('norm_scale', torch.ones(1, 1, 3))

    def set_norm_params(self, mean, std, scale_factor):
        """设置归一化参数（训练前从数据集计算得到，会随 state_dict 保存/加载）"""
        self.norm_mean.copy_(mean)
        self.norm_std.copy_(std)
        self.norm_scale.copy_(scale_factor)

    def _kinematic_decode(self, h_dec):
        """
        运动学解码：从 decoder hidden 预测 v/κ/dy profile，用公式 rollout 成 dxdydyaw。

        物理公式（body 系）:
            dx  = v * dt        纵向位移
            dyaw = v * κ * dt   航向角变化 = 速度 × 曲率 × 时间
            dy  ≈ residual      横向滑移（正常驾驶极小）

        Args:
            h_dec: [B, D] - transformer decoder 输出
        Returns:
            x_norm:  [B, T, 3] - 归一化空间的 dxdydyaw
            v:       [B, T]    - 带符号的纵向速度 profile (m/s)，允许为负，表示倒车
            kappa:   [B, T]    - 曲率 profile (1/m)，用于外部 smoothness loss
        """
        v = 40.0 * torch.tanh(self.v_head(h_dec) / 40.0)       # [B, T] 带符号纵向速度，限幅避免速度爆炸
        kappa = torch.tanh(self.kappa_head(h_dec)) * 0.5        # [B, T] 曲率，[-0.5, 0.5] 1/m
        dy_phys = self.dy_head(h_dec) * 0.01                    # [B, T] 横向，量级 ~mm

        dx_phys = v * self.dt                                   # [B, T]
        dyaw_phys = v * kappa * self.dt                         # [B, T]

        x_phys = torch.stack([dx_phys, dy_phys, dyaw_phys], dim=-1)  # [B, T, 3]

        # 物理空间 -> 归一化空间（与 preprocess_and_save_norm_params 的逆操作对应）
        x_norm = (x_phys - self.norm_mean) / (self.norm_std + 1e-8) / self.norm_scale

        return x_norm, v, kappa

    def encode(self, x: torch.Tensor):
        """
        x: [B, T, C] (归一化空间)
        返回: z: [B, D] - 轨迹级 latent
        """
        B, T, C = x.shape
        assert T == self.input_steps and C == self.input_dim

        x_flat = x.view(B, self.input_flat_dim)
        h = self.input_proj(x_flat)
        h = h.unsqueeze(1)
        h = h + self.pos_embed
        h = self.transformer_encoder(h)
        h = h.squeeze(1)
        z = self.to_latent(h)
        return z

    def decode_from_codes(self, codes: torch.Tensor):
        """
        推理时使用：从 token codes 恢复归一化空间的轨迹
        codes: [B, num_layers]
        返回: x_recon: [B, T, 3] (归一化空间)
        """
        z_q = self.rvq.decode_from_codes(codes)

        h_dec = z_q.unsqueeze(1)
        h_dec = h_dec + self.decoder_pos_embed
        h_dec = self.transformer_decoder(h_dec)
        h_dec = h_dec.squeeze(1)

        x_recon, _, _ = self._kinematic_decode(h_dec)
        return x_recon

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C] (归一化空间)
        返回:
            x_recon: [B, T, C] (归一化空间)
            vq_loss: 标量 VQ 损失
            codes:   [B, num_quantizers]
            v:       [B, T] 速度 profile (m/s)
            kappa:   [B, T] 曲率 profile (1/m)
        """
        z = self.encode(x)
        z_q, vq_loss, codes = self.rvq(z)

        h_dec = z_q.unsqueeze(1)
        h_dec = h_dec + self.decoder_pos_embed
        h_dec = self.transformer_decoder(h_dec)
        h_dec = h_dec.squeeze(1)

        x_recon, v, kappa = self._kinematic_decode(h_dec)

        return x_recon, vq_loss, codes, v, kappa


def train_rvq_taae(
    data_array: np.ndarray,
    save_dir: str = "./work_dirs/tokenizer/rvq_taae_0205",
    data_type: str = "pred",
    batch_size: int = 4096,
    num_layers: int = 15,
    num_transformer_layers: int = 2,
    epochs: int = 500,
):
    """
    使用 TAAE 结构训练 RVQ 模型，整体流程与 train.py 中的 train_rvq 类似，
    方便直接对比效果。

    Args:
        data_array: [M, T, 3] numpy array (dxdydyaw)
        save_dir:   模型与归一化参数保存目录
        data_type:  'pred' 或 'history'，用于区分不同长度 / 使用场景
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_steps = data_array.shape[1]

    # 1. 归一化预处理 + 保存归一化参数
    data_normalized = preprocess_and_save_norm_params(data_array, save_dir, data_type)

    # 2. 准备 DataLoader（优化数据加载速度）
    dataset = TensorDataset(torch.FloatTensor(data_normalized))
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # 增加 worker 数量，充分利用 CPU
        pin_memory=True,  # 加速 GPU 传输
        prefetch_factor=2,  # 预取数据
        persistent_workers=True,  # 保持 worker 进程，避免重复创建
    )

    # 3. 初始化模型 (TAAE 版本)
    model = TrajRVQTransformer(
        input_steps=num_steps,
        input_dim=data_array.shape[2],
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,  # 128
        nhead=4,  # 4
        num_transformer_layers=num_transformer_layers,
    ).to(device)
    
    # 使用 torch.compile 加速（PyTorch 2.0+，可提升 20-30% 速度）
    # 注意：Flash Attention 要求 dropout=0，但训练时需要 dropout，所以使用 "default" 模式
    # 或者完全禁用 compile（如果遇到问题）
    use_compile = False
    if use_compile:
        try:
            # 使用 "default" 模式，避免 Flash Attention 的限制
            # "reduce-overhead" 模式会尝试使用 Flash Attention，但要求 dropout=0
            model = torch.compile(model, mode="default")
            print("Model compiled with torch.compile (default mode)")
        except Exception as e:
            print(f"torch.compile failed: {e}, using normal model")
            print("  Note: This is often due to Flash Attention requiring dropout=0")
            use_compile = False
    else:
        print("torch.compile disabled, using normal model")
    
    # 混合精度训练（FP16/BF16）- 可提升 1.5-2x 速度
    # A100 支持 BF16，性能更好
    use_amp = True
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        # A100/H100 等 Ampere+ 架构，使用 BF16
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 不需要 scaler
        dtype = torch.bfloat16
        print("Using BF16 mixed precision training")
    else:
        # 较老的 GPU，使用 FP16
        scaler = torch.cuda.amp.GradScaler()
        dtype = torch.float16
        print("Using FP16 mixed precision training")

    # 学习率设置
    initial_lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    # Warmup + CosineAnnealingLR
    # 前 warmup_epochs 个 epoch 使用线性 warmup，从 warmup_start_lr 线性增长到 initial_lr
    # 然后切换到 CosineAnnealingLR
    warmup_epochs = 5
    warmup_start_lr = 1e-5

    # Warmup 阶段：线性增长
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_lr / initial_lr,  # 起始学习率比例
        end_factor=1.0,  # 结束学习率比例（即 initial_lr）
        total_iters=warmup_epochs,
    )

    # CosineAnnealingLR 阶段：余弦退火
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6  # 剩余 epoch 数  # 最小学习率
    )

    # 组合两个调度器：先 warmup，然后 cosine annealing
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],  # 在第 warmup_epochs 个 epoch 后切换到 cosine
    )
    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")
    with open(norm_path, 'rb') as f:
        norm_params = pickle.load(f)
    
    # 将归一化参数转为 torch tensor 并移到 device
    mean = torch.tensor(norm_params['mean'], device=device, dtype=torch.float32)  # [1, 1, C]
    std = torch.tensor(norm_params['std'], device=device, dtype=torch.float32)  # [1, 1, C]
    scale_factor = torch.tensor(norm_params['scale_factor'], device=device, dtype=torch.float32)  # [1, 1, C]

    # 将归一化参数注入模型（用于运动学 decoder 的物理空间 <-> 归一化空间转换）
    model.set_norm_params(mean, std, scale_factor)
    print(f"Norm params set: mean={mean.squeeze().cpu().numpy()}, "
          f"std={std.squeeze().cpu().numpy()}, scale={scale_factor.squeeze().cpu().numpy()}")

    print("Start Training (Kinematic RVQ Transformer)...")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))
    # 最大允许误差
    MAX_LATERAL = 1  # 米
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_vel_loss = 0.0
        total_acc_loss = 0.0
        total_kin_smooth_loss = 0.0
        total_turn_global_loss = 0.0
        total_turn_yaw_loss = 0.0
        total_turn_samples = 0
        total_traj_error = 0.0
        total_endpoint_error = 0.0
        total_vrr_count = 0
        total_samples = 0

        if epoch > epochs * 0.8:
            model.rvq.dropout = 0.0

        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                x_recon, vq_loss, _, v, kappa = model(x)

                # Loss 1: Reconstruction MSE（归一化空间）
                mse_dxdy = F.mse_loss(x_recon[..., :2], x[..., :2])
                mse_dyaw = F.mse_loss(x_recon[..., 2], x[..., 2])
                recon_loss = mse_dxdy + 14.0 * mse_dyaw

                # Loss 2: 真实动态监督
                pred_phys_for_loss = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
                gt_phys_for_loss = (x * model.norm_scale * model.norm_std) + model.norm_mean
                vel_loss = signed_velocity_loss_from_dxdydyaw(pred_phys_for_loss, gt_phys_for_loss, dt=model.dt)
                acc_loss = signed_acceleration_loss_from_dxdydyaw(pred_phys_for_loss, gt_phys_for_loss, dt=model.dt)

                pred_phys_for_loss = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
                gt_phys_for_loss = (x * model.norm_scale * model.norm_std) + model.norm_mean
                turn_global_loss, turn_yaw_loss, turn_mask = turn_global_yaw_loss(
                    pred_phys_for_loss,
                    gt_phys_for_loss,
                    turn_threshold=0.35,
                )

                # Loss 3: 运动学参数平滑性（直接约束物理量，比频域 smoothness 更直观）
                # signed v 的差分 ≈ 纵向加速度，kappa 的差分 ≈ 曲率变化率（方向盘转速）
                acc = (v[:, 1:] - v[:, :-1]) / model.dt          # [B, T-1] 带符号纵向加速度 (m/s²)
                kappa_rate = (kappa[:, 1:] - kappa[:, :-1]) / model.dt  # [B, T-1] 曲率变化率
                kin_smooth_loss = acc.pow(2).mean() + kappa_rate.pow(2).mean()
                ### 上述不太对，应该再求一次导数，约束acc和kappa_rate ###
                # acc_rate = (acc[:, 1:] - acc[:, :-1]) / model.dt  # [B, T-2] 加加速度 (m/s³)，即 jerk
                # kappa_acc = (kappa_rate[:, 1:] - kappa_rate[:, :-1]) / model.dt  # [B, T-2] 曲率加速度 (1/m/s²)
                # kin_smooth_loss = acc_rate.pow(2).mean() + kappa_acc.pow(2).mean()
                # Loss weights
                recon_loss_weight = 10.0
                vq_loss_weight = 5.0
                vel_loss_weight = 0.5
                acc_loss_weight = 0.05
                kin_smooth_weight = 1e-2 if epoch > 30 else 0.0
                turn_global_weight = 1.0
                turn_yaw_weight = 2.0

                loss = (
                    recon_loss_weight * recon_loss
                    + vq_loss_weight * vq_loss
                    + vel_loss_weight * vel_loss
                    + acc_loss_weight * acc_loss
                    + kin_smooth_weight * kin_smooth_loss
                    + turn_global_weight * turn_global_loss
                    + turn_yaw_weight * turn_yaw_loss
                )

            if dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_vel_loss += vel_loss.item()
            total_acc_loss += acc_loss.item()
            total_kin_smooth_loss += kin_smooth_loss.item()
            total_turn_global_loss += turn_global_loss.item()
            total_turn_yaw_loss += turn_yaw_loss.item()
            total_turn_samples += turn_mask.sum().item()
            
            # 计算轨迹误差
            with torch.no_grad():
                # 1. 恢复到物理空间
                pred_phys = (x_recon * model.norm_scale * model.norm_std) + model.norm_mean
                gt_phys = (x * model.norm_scale * model.norm_std) + model.norm_mean
                
                dx_pred, dy_pred, dyaw_pred = pred_phys[..., 0], pred_phys[..., 1], pred_phys[..., 2]
                dx_gt, dy_gt, dyaw_gt = gt_phys[..., 0], gt_phys[..., 1], gt_phys[..., 2]
                
                # 2. 正确的运动学积分 (包含 Yaw 的旋转矩阵)
                # 累加得到每一步的绝对航向角 (假设初始朝向为 0)
                yaw_pred = torch.cumsum(dyaw_pred, dim=1)
                yaw_gt = torch.cumsum(dyaw_gt, dim=1)
                
                # 将车身坐标系的 dx, dy 投影到全局 XY 坐标系
                # 由于第一步是基于当前朝向，我们需要把前一帧的 yaw 作为当前帧的投影基准
                # 为了简化计算并对齐维度，我们直接用当前 step 的 yaw 做近似投影
                dx_global_pred = dx_pred * torch.cos(yaw_pred) - dy_pred * torch.sin(yaw_pred)
                dy_global_pred = dx_pred * torch.sin(yaw_pred) + dy_pred * torch.cos(yaw_pred)
                
                dx_global_gt = dx_gt * torch.cos(yaw_gt) - dy_gt * torch.sin(yaw_gt)
                dy_global_gt = dx_gt * torch.sin(yaw_gt) + dy_gt * torch.cos(yaw_gt)
                
                # 全局坐标累加得到真实的 X, Y
                x_pred_global = torch.cumsum(dx_global_pred, dim=1)
                y_pred_global = torch.cumsum(dy_global_pred, dim=1)
                
                x_gt_global = torch.cumsum(dx_global_gt, dim=1)
                y_gt_global = torch.cumsum(dy_global_gt, dim=1)
                
                # 3. 计算每个时间步的欧式距离误差 [B, T]
                step_dist_error = torch.sqrt((x_pred_global - x_gt_global)**2 + (y_pred_global - y_gt_global)**2)
                
                # 计算 ADE (Average Displacement Error) 和 FDE (Final Displacement Error)
                traj_error = step_dist_error.mean(dim=1).mean().item() # 平均轨迹误差
                endpoint_dist = step_dist_error[:, -1].mean().item()   # 第 25 步的误差
                
                # 4. 计算真正的 VRR (Valid Reconstruction Rate)
                # 定义：整条轨迹中最大的位移误差是否小于阈值 MAX_LATERAL
                max_dist_error_per_traj = step_dist_error.max(dim=1)[0] # [B]
                valid_count = (max_dist_error_per_traj < MAX_LATERAL).sum().item()
                
                total_traj_error += traj_error
                total_endpoint_error += endpoint_dist
                total_vrr_count += valid_count
                total_samples += x.shape[0]

        scheduler.step()

        # 对应上述weight的tensorborad展示
        avg_recon = total_recon_loss / len(dataloader)
        avg_vq = total_vq_loss / len(dataloader)
        avg_vel = total_vel_loss / len(dataloader)
        avg_acc = total_acc_loss / len(dataloader)
        avg_kin = total_kin_smooth_loss / len(dataloader)
        avg_turn_global = total_turn_global_loss / len(dataloader)
        avg_turn_yaw = total_turn_yaw_loss / len(dataloader)
        turn_ratio = total_turn_samples / total_samples if total_samples > 0 else 0.0
        avg_weight = (
            10.0 * avg_recon
            + 5.0 * avg_vq
            + 0.5 * avg_vel
            + 0.05 * avg_acc
            + (1e-2 if epoch > 30 else 0.0) * avg_kin
            + 1.0 * avg_turn_global
            + 2.0 * avg_turn_yaw
        )
        writer.add_scalar("loss/weight_recon", 10.0 * avg_recon, epoch + 1)
        writer.add_scalar("loss/weight_vq", 5.0 * avg_vq, epoch + 1)
        writer.add_scalar("loss/weight_vel", 0.5 * avg_vel, epoch + 1)
        writer.add_scalar("loss/weight_acc", 0.05 * avg_acc, epoch + 1)
        writer.add_scalar("loss/weight_kin_smooth", (1e-2 if epoch > 30 else 0.0) * avg_kin, epoch + 1)
        writer.add_scalar("loss/weight_turn_global", 1.0 * avg_turn_global, epoch + 1)
        writer.add_scalar("loss/weight_turn_yaw", 2.0 * avg_turn_yaw, epoch + 1)
        writer.add_scalar("loss/weight", avg_weight, epoch + 1)

        if (epoch + 1) % 10 == 0:
            avg_traj = total_traj_error / len(dataloader)
            avg_endpoint = total_endpoint_error / len(dataloader)
            vrr = total_vrr_count / total_samples if total_samples > 0 else 0.0
            # 打印 v/kappa 的统计信息，方便调试
            with torch.no_grad():
                v_mean = v.mean().item()
                v_min = v.min().item()
                v_max = v.max().item()
                v_abs_mean = v.abs().mean().item()
                kappa_abs_mean = kappa.abs().mean().item()
            print(
                f"[KinRVQ] Epoch {epoch+1:03d} | Recon: {avg_recon:.5f} | "
                f"VQ: {avg_vq:.5f} | Vel: {avg_vel:.5f} | Acc: {avg_acc:.5f} | KinSmooth: {avg_kin:.5f} | "
                f"TurnGlobal: {avg_turn_global:.5f} | TurnYaw: {avg_turn_yaw:.5f} | TurnRatio: {turn_ratio:.3f} | "
                f"TrajErr: {avg_traj:.4f} m | EndErr: {avg_endpoint:.4f} m | VRR: {vrr:.4f} | "
                f"v_mean: {v_mean:.2f} m/s | v_min: {v_min:.2f} | v_max: {v_max:.2f} | "
                f"v_abs_mean: {v_abs_mean:.2f} | κ_abs: {kappa_abs_mean:.4f}"
            )

    # 5. 保存模型
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, f"{data_type}_rvq_taae_model.pth"),
    )
    print(f"TAAE Training Done. Model saved to {save_dir}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RVQ TAAE tokenizer.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_kin_0311")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    sampled_trajs = load_sampled_datas(args.data_path)
    if args.data_type == "history":
        sampled_trajs = sampled_trajs[:, :14, :]
    if args.max_samples > 0:
        sampled_trajs = sampled_trajs[: args.max_samples]

    print(
        f"Train config | data_type={args.data_type} | num_layers={args.num_layers} | "
        f"num_transformer_layers={args.num_transformer_layers} | "
        f"batch_size={args.batch_size} | epochs={args.epochs} | save_dir={args.save_dir}"
    )
    print(f"Dataset shape: {sampled_trajs.shape}")

    train_rvq_taae(
        sampled_trajs,
        save_dir=args.save_dir,
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_transformer_layers=args.num_transformer_layers,
        epochs=args.epochs,
    )
