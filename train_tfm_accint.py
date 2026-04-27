import os
import pickle
from datetime import datetime
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# 兼容从仓库根目录或当前目录两种启动方式
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rvq_model import ResidualVQ
from utils import preprocess_and_save_norm_params, load_sampled_datas


def integrate_local_to_global_with_yaw(dxdydyaw: torch.Tensor):
    """
    将 body 系 dxdydyaw 积分为全局轨迹。

    Args:
        dxdydyaw: [B, T, 3], 每一步 body 坐标系位移与航向增量

    Returns:
        pos_xy_global: [B, T, 2], 全局 XY
        yaw:           [B, T],    绝对航向（初始为 0）
        disp_xy_global:[B, T, 2], 每一步全局位移
    """
    dx = dxdydyaw[..., 0]
    dy = dxdydyaw[..., 1]
    dyaw = dxdydyaw[..., 2]

    yaw = torch.cumsum(dyaw, dim=1)
    prev_yaw = torch.zeros_like(yaw)
    prev_yaw[:, 1:] = yaw[:, :-1]

    dx_global = dx * torch.cos(prev_yaw) - dy * torch.sin(prev_yaw)
    dy_global = dx * torch.sin(prev_yaw) + dy * torch.cos(prev_yaw)

    x_global = torch.cumsum(dx_global, dim=1)
    y_global = torch.cumsum(dy_global, dim=1)
    pos_xy_global = torch.stack([x_global, y_global], dim=-1)
    disp_xy_global = torch.stack([dx_global, dy_global], dim=-1)

    return pos_xy_global, yaw, disp_xy_global


def compute_dynamics_from_dxdydyaw(dxdydyaw: torch.Tensor, dt: float):
    """
    从 body-frame dxdydyaw 计算动力学监督目标。

    Args:
        dxdydyaw: [B, T, 3], body 系每步增量 (dx, dy, dyaw)
        dt:       时间步长

    Returns:
        dict, 至少包含:
            pos_xy_global: [B, T, 2]
            disp_xy_global:[B, T, 2]
            yaw:           [B, T]
            dyaw:          [B, T]
            vel_xy:        [B, T, 2]
            speed:         [B, T]
            yaw_rate:      [B, T]
            acc_xy:        [B, T, 2]
            acc_yaw:       [B, T]
            v0_xy:         [B, 2]
            w0:            [B]
    """
    pos_xy_global, yaw, disp_xy_global = integrate_local_to_global_with_yaw(dxdydyaw)
    dyaw = dxdydyaw[..., 2]

    vel_xy = disp_xy_global / (dt + 1e-8)
    speed = torch.sqrt((vel_xy[..., 0] ** 2 + vel_xy[..., 1] ** 2) + 1e-6)
    v0_xy = vel_xy[:, 0, :]

    yaw_rate = dyaw / (dt + 1e-8)
    w0 = yaw_rate[:, 0]

    # 约定：acc[0] 不作为可观测差分加速度（固定为 0），
    # 只从 t>=1 监督/使用 acc[t] = (vel[t]-vel[t-1])/dt
    acc_xy = torch.zeros_like(vel_xy)
    acc_yaw = torch.zeros_like(yaw_rate)
    if dxdydyaw.shape[1] > 1:
        acc_xy[:, 1:, :] = (vel_xy[:, 1:, :] - vel_xy[:, :-1, :]) / (dt + 1e-8)
        acc_yaw[:, 1:] = (yaw_rate[:, 1:] - yaw_rate[:, :-1]) / (dt + 1e-8)

    return {
        "pos_xy_global": pos_xy_global,
        "disp_xy_global": disp_xy_global,
        "yaw": yaw,
        "dyaw": dyaw,
        "vel_xy": vel_xy,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "acc_xy": acc_xy,
        "acc_yaw": acc_yaw,
        "v0_xy": v0_xy,
        "w0": w0,
    }


class AccFirstRVQTokenizer(nn.Module):
    """
    思路：
    1) Decoder 预测加速度序列（ax, ay, alpha）与初始速度（v0x, v0y, w0）；
    2) 通过积分恢复速度、位移、yaw；
    3) 再反投影到 body 系得到 dxdydyaw。

    训练主目标放在动力学一致性（acc/vel/yaw-rate），轨迹路径作为辅助项。
    """

    def __init__(
        self,
        input_steps: int = 25,
        input_dim: int = 3,
        num_layers: int = 15,
        vocab_size: int = 1024,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dt: float = 0.2,
    ):
        super().__init__()

        self.input_steps = input_steps
        self.input_dim = input_dim
        self.input_flat_dim = input_steps * input_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dt = dt

        # -------- Encoder --------
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

        # -------- RVQ Bottleneck --------
        self.rvq = ResidualVQ(
            num_quantizers=num_layers,
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            dropout=0.2,
            commitment_cost=0.25,
        )

        # -------- Decoder --------
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

        # 预测加速度与初始速度
        self.acc_xy_head = nn.Linear(d_model, input_steps * 2)
        self.acc_yaw_head = nn.Linear(d_model, input_steps)
        self.v0_xy_head = nn.Linear(d_model, 2)
        self.w0_head = nn.Linear(d_model, 1)

        # 参数范围（物理先验）
        self.acc_xy_scale = 8.0      # m/s^2
        self.acc_yaw_scale = 1.5     # rad/s^2
        self.v0_scale = 45.0         # m/s
        self.w0_scale = 1.5          # rad/s

        # 归一化参数 buffer（state_dict 可保存）
        self.register_buffer("norm_mean", torch.zeros(1, 1, 3))
        self.register_buffer("norm_std", torch.ones(1, 1, 3))
        self.register_buffer("norm_scale", torch.ones(1, 1, 3))

    def set_norm_params(self, mean, std, scale_factor):
        self.norm_mean.copy_(mean)
        self.norm_std.copy_(std)
        self.norm_scale.copy_(scale_factor)

    def to_phys(self, x_norm: torch.Tensor) -> torch.Tensor:
        return (x_norm * self.norm_scale * self.norm_std) + self.norm_mean

    def to_norm(self, x_phys: torch.Tensor) -> torch.Tensor:
        return (x_phys - self.norm_mean) / (self.norm_std + 1e-8) / (self.norm_scale + 1e-8)

    def encode(self, x: torch.Tensor):
        """x: [B, T, 3] (归一化空间)"""
        bsz, t, c = x.shape
        assert t == self.input_steps and c == self.input_dim

        x_flat = x.view(bsz, self.input_flat_dim)
        h = self.input_proj(x_flat).unsqueeze(1)
        h = h + self.pos_embed
        h = self.transformer_encoder(h)
        z = self.to_latent(h.squeeze(1))
        return z

    def _decode_heads(self, h_dec: torch.Tensor):
        bsz = h_dec.shape[0]

        acc_xy = self.acc_xy_head(h_dec).view(bsz, self.input_steps, 2)
        acc_xy = torch.tanh(acc_xy) * self.acc_xy_scale

        acc_yaw = self.acc_yaw_head(h_dec)
        acc_yaw = torch.tanh(acc_yaw) * self.acc_yaw_scale

        v0_xy = self.v0_xy_head(h_dec)
        v0_xy = torch.tanh(v0_xy) * self.v0_scale

        w0 = self.w0_head(h_dec).squeeze(-1)
        w0 = torch.tanh(w0) * self.w0_scale

        return acc_xy, acc_yaw, v0_xy, w0

    def _rollout_with_dynamics(
        self,
        acc_xy: torch.Tensor,
        acc_yaw: torch.Tensor,
        v0_xy: torch.Tensor,
        w0: torch.Tensor,
    ):
        """
        根据加速度与初始速度积分得到速度、位移和姿态。
        """
        dt = self.dt
        t_steps = acc_xy.shape[1]
        # 约定：vel[0] 由 v0 直接定义；acc[t] 仅用于从 vel[t-1] 积分到 vel[t] (t>=1)
        vel_xy = torch.zeros_like(acc_xy)
        vel_xy[:, 0, :] = v0_xy
        if t_steps > 1:
            vel_xy[:, 1:, :] = v0_xy.unsqueeze(1) + dt * torch.cumsum(acc_xy[:, 1:, :], dim=1)
        disp_xy_global = vel_xy * dt
        pos_xy_global = torch.cumsum(disp_xy_global, dim=1)

        yaw_rate = torch.zeros_like(acc_yaw)
        yaw_rate[:, 0] = w0
        if t_steps > 1:
            yaw_rate[:, 1:] = w0.unsqueeze(1) + dt * torch.cumsum(acc_yaw[:, 1:], dim=1)
        dyaw = yaw_rate * dt
        yaw = torch.cumsum(dyaw, dim=1)

        prev_yaw = torch.zeros_like(yaw)
        prev_yaw[:, 1:] = yaw[:, :-1]

        # 全局位移 -> body 位移
        dx_local = (
            disp_xy_global[..., 0] * torch.cos(prev_yaw)
            + disp_xy_global[..., 1] * torch.sin(prev_yaw)
        )
        dy_local = (
            -disp_xy_global[..., 0] * torch.sin(prev_yaw)
            + disp_xy_global[..., 1] * torch.cos(prev_yaw)
        )

        x_phys = torch.stack([dx_local, dy_local, dyaw], dim=-1)  # [B, T, 3]
        x_norm = self.to_norm(x_phys)

        speed = torch.sqrt((vel_xy[..., 0] ** 2 + vel_xy[..., 1] ** 2) + 1e-6)  # [B, T]

        aux = {
            "acc_xy": acc_xy,
            "acc_yaw": acc_yaw,
            "v0_xy": v0_xy,
            "w0": w0,
            "vel_xy": vel_xy,
            "speed": speed,
            "yaw_rate": yaw_rate,
            "dyaw": dyaw,
            "disp_xy_global": disp_xy_global,
            "pos_xy_global": pos_xy_global,
            "yaw": yaw,
        }
        return x_norm, aux

    def _decode_from_latent(self, z_q: torch.Tensor):
        h_dec = z_q.unsqueeze(1)
        h_dec = h_dec + self.decoder_pos_embed
        h_dec = self.transformer_decoder(h_dec).squeeze(1)

        acc_xy, acc_yaw, v0_xy, w0 = self._decode_heads(h_dec)
        return self._rollout_with_dynamics(acc_xy, acc_yaw, v0_xy, w0)

    def decode_from_codes(self, codes: torch.Tensor, return_aux: bool = False):
        z_q = self.rvq.decode_from_codes(codes)
        x_recon, aux = self._decode_from_latent(z_q)
        if return_aux:
            return x_recon, aux
        return x_recon

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        z_q, vq_loss, codes = self.rvq(z)
        x_recon, aux = self._decode_from_latent(z_q)
        return x_recon, vq_loss, codes, aux


def train_rvq_accint(
    data_array: np.ndarray,
    save_dir: str = "./work_dirs/tokenizer/rvq_tfm_accfirst",
    data_type: str = "pred",
    batch_size: int = 4096,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_steps = data_array.shape[1]

    # 1) 归一化 + 保存参数
    data_normalized = preprocess_and_save_norm_params(data_array, save_dir, data_type)

    # 2) DataLoader
    dataset = TensorDataset(torch.FloatTensor(data_normalized))
    num_workers = min(4, os.cpu_count() or 1)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        dataloader_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # 3) 模型
    model = AccFirstRVQTokenizer(
        input_steps=num_steps,
        input_dim=data_array.shape[2],
        num_layers=15,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dt=0.2,
    ).to(device)

    # 混合精度
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
        print("Using FP32 training (CPU)")

    # 优化器与调度器
    epochs = 500
    initial_lr = 1e-3

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    warmup_epochs = 5
    warmup_start_lr = 1e-5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_lr / initial_lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # 注入归一化参数
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

    print("Start Training (Acceleration-first RVQ Tokenizer)...")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))

    max_lateral = 1.0

    for epoch in range(epochs):
        model.train()

        total_acc_xy = 0.0
        total_acc_yaw = 0.0
        total_v0 = 0.0
        total_w0 = 0.0
        total_vel = 0.0
        total_speed = 0.0
        total_yaw_rate = 0.0
        total_yaw = 0.0
        total_vq = 0.0
        total_traj = 0.0
        total_final_pos = 0.0
        total_recon = 0.0
        total_jerk_smooth = 0.0
        total_speed_weight = 0.0

        total_ade = 0.0
        total_fde = 0.0
        total_vrr_count = 0
        total_samples = 0
        total_pred_speed_mean = 0.0
        total_gt_speed_mean = 0.0
        total_pred_acc_mean = 0.0
        total_gt_acc_mean = 0.0

        if epoch > epochs * 0.6:
            model.rvq.dropout = 0.0

        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                x_recon, vq_loss, _, aux = model(x)

                # ------- GT 转物理空间并计算动力学监督 -------
                gt_phys = model.to_phys(x)
                gt_dyn = compute_dynamics_from_dxdydyaw(gt_phys, dt=model.dt)

                # 速度/转向/加速度感知权重：提高动态更激烈样本的话语权
                gt_avg_speed = gt_dyn["speed"].mean(dim=1)
                gt_final_yaw = gt_dyn["yaw"][:, -1]
                gt_mean_abs_acc = torch.sqrt(
                    (gt_dyn["acc_xy"][..., 0] ** 2 + gt_dyn["acc_xy"][..., 1] ** 2) + 1e-6
                ).mean(dim=1)
                speed_weight = 1.0 + torch.clamp(gt_avg_speed / 15.0, min=0.0, max=2.5)
                turn_weight = 1.0 + torch.clamp(gt_final_yaw.abs() / 1.2, min=0.0, max=1.5)
                acc_weight = 1.0 + torch.clamp(gt_mean_abs_acc / 2.0, min=0.0, max=1.5)
                sample_weight = torch.clamp(
                    0.45 * speed_weight + 0.35 * turn_weight + 0.20 * acc_weight, min=1.0, max=4.0
                )

                # ------- 1) acceleration-first 主损失 -------
                if model.input_steps > 1:
                    # acc[0] 不参与监督，避免与 v0/vel[0] 的定义产生 off-by-one 冲突
                    acc_xy_mse_per_sample = (
                        aux["acc_xy"][:, 1:, :] - gt_dyn["acc_xy"][:, 1:, :]
                    ).pow(2).mean(dim=(1, 2))
                    acc_yaw_mse_per_sample = (
                        aux["acc_yaw"][:, 1:] - gt_dyn["acc_yaw"][:, 1:]
                    ).pow(2).mean(dim=1)
                else:
                    acc_xy_mse_per_sample = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                    acc_yaw_mse_per_sample = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                acc_xy_loss = (acc_xy_mse_per_sample * sample_weight).mean()
                acc_yaw_loss = (acc_yaw_mse_per_sample * sample_weight).mean()

                # ------- 2) 初始状态损失 -------
                v0_mse_per_sample = (aux["v0_xy"] - gt_dyn["v0_xy"]).pow(2).mean(dim=1)
                v0_loss = (v0_mse_per_sample * sample_weight).mean()

                w0_mse_per_sample = (aux["w0"] - gt_dyn["w0"]).pow(2)
                w0_loss = (w0_mse_per_sample * sample_weight).mean()

                # ------- 3) 速度一致性 -------
                vel_mse_per_sample = (aux["vel_xy"] - gt_dyn["vel_xy"]).pow(2).mean(dim=(1, 2))
                vel_loss = (vel_mse_per_sample * sample_weight).mean()

                speed_mse_per_sample = (aux["speed"] - gt_dyn["speed"]).pow(2).mean(dim=1)
                speed_loss = (speed_mse_per_sample * sample_weight).mean()

                # ------- 4) yaw 一致性 -------
                yaw_rate_mse_per_sample = (aux["yaw_rate"] - gt_dyn["yaw_rate"]).pow(2).mean(dim=1)
                yaw_rate_loss = (yaw_rate_mse_per_sample * sample_weight).mean()

                yaw_mse_per_sample = (aux["yaw"] - gt_dyn["yaw"]).pow(2).mean(dim=1)
                yaw_loss = (yaw_mse_per_sample * sample_weight).mean()

                # ------- 5) 路径辅助 -------
                traj_err2_per_sample = (aux["pos_xy_global"] - gt_dyn["pos_xy_global"]).pow(2).mean(dim=(1, 2))
                traj_global_loss = (traj_err2_per_sample * sample_weight).mean()

                final_pos_err2 = (
                    aux["pos_xy_global"][:, -1, :] - gt_dyn["pos_xy_global"][:, -1, :]
                ).pow(2).mean(dim=1)
                final_pos_loss = (final_pos_err2 * sample_weight).mean()

                # ------- 6) local dxdydyaw 辅助 -------
                mse_dxdy_per_sample = (x_recon[..., :2] - x[..., :2]).pow(2).mean(dim=(1, 2))
                mse_dyaw_per_sample = (x_recon[..., 2] - x[..., 2]).pow(2).mean(dim=1)
                recon_loss = ((mse_dxdy_per_sample + 14.0 * mse_dyaw_per_sample) * sample_weight).mean()

                # ------- 7) jerk 平滑 -------
                if model.input_steps > 1:
                    jerk_xy = (aux["acc_xy"][:, 1:, :] - aux["acc_xy"][:, :-1, :]) / model.dt
                    jerk_yaw = (aux["acc_yaw"][:, 1:] - aux["acc_yaw"][:, :-1]) / model.dt
                    jerk_smooth_loss = jerk_xy.pow(2).mean() + jerk_yaw.pow(2).mean()
                else:
                    jerk_smooth_loss = aux["acc_xy"].sum() * 0.0

                # 分阶段 loss 权重：保持 acc 相关项为主导，路径项为辅助
                if epoch < 30:
                    acc_xy_w = 8.0
                    acc_yaw_w = 4.0
                    v0_w = 3.0
                    w0_w = 2.0
                    vel_w = 2.0
                    speed_w = 1.5
                    yaw_rate_w = 1.5
                    yaw_w = 0.8
                    traj_w = 1.0
                    final_pos_w = 1.0
                    recon_w = 2.0
                    vq_w = 1.0
                    jerk_w = 0.0
                elif epoch < 100:
                    acc_xy_w = 6.0
                    acc_yaw_w = 3.0
                    v0_w = 2.0
                    w0_w = 1.5
                    vel_w = 2.0
                    speed_w = 1.2
                    yaw_rate_w = 1.2
                    yaw_w = 1.0
                    traj_w = 1.5
                    final_pos_w = 1.0
                    recon_w = 2.0
                    vq_w = 3.0
                    jerk_w = 0.0
                else:
                    acc_xy_w = 5.0
                    acc_yaw_w = 2.5
                    v0_w = 1.5
                    w0_w = 1.0
                    vel_w = 1.5
                    speed_w = 1.0
                    yaw_rate_w = 1.0
                    yaw_w = 1.0
                    traj_w = 2.0
                    final_pos_w = 1.5
                    recon_w = 2.0
                    vq_w = 5.0
                    jerk_w = 5e-4

                loss = (
                    acc_xy_w * acc_xy_loss
                    + acc_yaw_w * acc_yaw_loss
                    + v0_w * v0_loss
                    + w0_w * w0_loss
                    + vel_w * vel_loss
                    + speed_w * speed_loss
                    + yaw_rate_w * yaw_rate_loss
                    + yaw_w * yaw_loss
                    + traj_w * traj_global_loss
                    + final_pos_w * final_pos_loss
                    + recon_w * recon_loss
                    + vq_w * vq_loss
                    + jerk_w * jerk_smooth_loss
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

            total_acc_xy += acc_xy_loss.item()
            total_acc_yaw += acc_yaw_loss.item()
            total_v0 += v0_loss.item()
            total_w0 += w0_loss.item()
            total_vel += vel_loss.item()
            total_speed += speed_loss.item()
            total_yaw_rate += yaw_rate_loss.item()
            total_yaw += yaw_loss.item()
            total_vq += vq_loss.item()
            total_traj += traj_global_loss.item()
            total_final_pos += final_pos_loss.item()
            total_recon += recon_loss.item()
            total_jerk_smooth += jerk_smooth_loss.item()
            total_speed_weight += sample_weight.mean().item()

            # 指标：ADE / FDE / VRR
            with torch.no_grad():
                step_err = torch.sqrt(
                    ((aux["pos_xy_global"] - gt_dyn["pos_xy_global"]).pow(2).sum(dim=-1)) + 1e-6
                )  # [B, T]
                ade = step_err.mean(dim=1).mean().item()
                fde = step_err[:, -1].mean().item()
                valid = (step_err.max(dim=1)[0] < max_lateral).sum().item()
                pred_acc_abs = torch.sqrt(
                    (aux["acc_xy"][..., 0] ** 2 + aux["acc_xy"][..., 1] ** 2) + 1e-6
                ).mean().item()
                gt_acc_abs = torch.sqrt(
                    (gt_dyn["acc_xy"][..., 0] ** 2 + gt_dyn["acc_xy"][..., 1] ** 2) + 1e-6
                ).mean().item()

                total_ade += ade
                total_fde += fde
                total_vrr_count += valid
                total_samples += x.shape[0]
                total_pred_speed_mean += aux["speed"].mean().item()
                total_gt_speed_mean += gt_dyn["speed"].mean().item()
                total_pred_acc_mean += pred_acc_abs
                total_gt_acc_mean += gt_acc_abs

        scheduler.step()

        avg_acc_xy = total_acc_xy / len(dataloader)
        avg_acc_yaw = total_acc_yaw / len(dataloader)
        avg_v0 = total_v0 / len(dataloader)
        avg_w0 = total_w0 / len(dataloader)
        avg_vel = total_vel / len(dataloader)
        avg_speed = total_speed / len(dataloader)
        avg_yaw_rate = total_yaw_rate / len(dataloader)
        avg_yaw = total_yaw / len(dataloader)
        avg_vq = total_vq / len(dataloader)
        avg_traj = total_traj / len(dataloader)
        avg_final_pos = total_final_pos / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_jerk_smooth = total_jerk_smooth / len(dataloader)
        avg_speed_weight = total_speed_weight / len(dataloader)
        avg_ade = total_ade / len(dataloader)
        avg_fde = total_fde / len(dataloader)
        vrr = total_vrr_count / total_samples if total_samples > 0 else 0.0
        avg_pred_speed_mean = total_pred_speed_mean / len(dataloader)
        avg_gt_speed_mean = total_gt_speed_mean / len(dataloader)
        avg_pred_acc_mean = total_pred_acc_mean / len(dataloader)
        avg_gt_acc_mean = total_gt_acc_mean / len(dataloader)

        if epoch < 30:
            acc_xy_w = 8.0
            acc_yaw_w = 4.0
            v0_w = 3.0
            w0_w = 2.0
            vel_w = 2.0
            speed_w = 1.5
            yaw_rate_w = 1.5
            yaw_w = 0.8
            traj_w = 1.0
            final_pos_w = 1.0
            recon_w = 2.0
            vq_w = 1.0
            jerk_w = 0.0
        elif epoch < 100:
            acc_xy_w = 6.0
            acc_yaw_w = 3.0
            v0_w = 2.0
            w0_w = 1.5
            vel_w = 2.0
            speed_w = 1.2
            yaw_rate_w = 1.2
            yaw_w = 1.0
            traj_w = 1.5
            final_pos_w = 1.0
            recon_w = 2.0
            vq_w = 3.0
            jerk_w = 0.0
        else:
            acc_xy_w = 5.0
            acc_yaw_w = 2.5
            v0_w = 1.5
            w0_w = 1.0
            vel_w = 1.5
            speed_w = 1.0
            yaw_rate_w = 1.0
            yaw_w = 1.0
            traj_w = 2.0
            final_pos_w = 1.5
            recon_w = 2.0
            vq_w = 5.0
            jerk_w = 5e-4

        weighted_loss = (
            acc_xy_w * avg_acc_xy
            + acc_yaw_w * avg_acc_yaw
            + v0_w * avg_v0
            + w0_w * avg_w0
            + vel_w * avg_vel
            + speed_w * avg_speed
            + yaw_rate_w * avg_yaw_rate
            + yaw_w * avg_yaw
            + traj_w * avg_traj
            + final_pos_w * avg_final_pos
            + recon_w * avg_recon
            + vq_w * avg_vq
            + jerk_w * avg_jerk_smooth
        )

        writer.add_scalar("loss/acc_xy", avg_acc_xy, epoch + 1)
        writer.add_scalar("loss/acc_yaw", avg_acc_yaw, epoch + 1)
        writer.add_scalar("loss/v0", avg_v0, epoch + 1)
        writer.add_scalar("loss/w0", avg_w0, epoch + 1)
        writer.add_scalar("loss/vel", avg_vel, epoch + 1)
        writer.add_scalar("loss/speed", avg_speed, epoch + 1)
        writer.add_scalar("loss/yaw_rate", avg_yaw_rate, epoch + 1)
        writer.add_scalar("loss/yaw", avg_yaw, epoch + 1)
        writer.add_scalar("loss/traj_global", avg_traj, epoch + 1)
        writer.add_scalar("loss/final_pos", avg_final_pos, epoch + 1)
        writer.add_scalar("loss/recon", avg_recon, epoch + 1)
        writer.add_scalar("loss/vq", avg_vq, epoch + 1)
        writer.add_scalar("loss/jerk_smooth", avg_jerk_smooth, epoch + 1)
        writer.add_scalar("loss/weighted", weighted_loss, epoch + 1)

        if (epoch + 1) % 10 == 0:
            print(
                f"[AccFirstRVQ] Epoch {epoch+1:03d} | "
                f"AccXY: {avg_acc_xy:.5f} | AccYaw: {avg_acc_yaw:.5f} | "
                f"V0: {avg_v0:.5f} | W0: {avg_w0:.5f} | "
                f"Vel: {avg_vel:.5f} | Speed: {avg_speed:.5f} | "
                f"YawRate: {avg_yaw_rate:.5f} | Yaw: {avg_yaw:.5f} | "
                f"Traj: {avg_traj:.5f} | FinalPos: {avg_final_pos:.5f} | "
                f"Recon: {avg_recon:.5f} | VQ: {avg_vq:.5f} | Jerk: {avg_jerk_smooth:.5f} | "
                f"ADE: {avg_ade:.4f} m | FDE: {avg_fde:.4f} m | VRR: {vrr:.4f} | "
                f"pred_speed_mean: {avg_pred_speed_mean:.2f} m/s | gt_speed_mean: {avg_gt_speed_mean:.2f} m/s | "
                f"pred_acc_mean: {avg_pred_acc_mean:.2f} m/s^2 | gt_acc_mean: {avg_gt_acc_mean:.2f} m/s^2 | "
                f"sample_w: {avg_speed_weight:.2f}"
            )

    # 保存模型
    model_path = os.path.join(save_dir, f"{data_type}_rvq_accint_model.pth")
    torch.save(model.state_dict(), model_path)
    writer.close()
    print(f"Acceleration-first RVQ training done. Model saved to {model_path}")


if __name__ == "__main__":
    batch_size = 4096
    sampled_trajs = load_sampled_datas()

    save_dir = "./work_dirs/tokenizer/rvq_tfm_accint_0423"
    data_type = "pred"  # 'pred' or 'history'
    print("data_type:", data_type)

    if data_type == "history":
        sampled_trajs = sampled_trajs[:, :14, :]

    train_rvq_accint(sampled_trajs, save_dir, data_type, batch_size)
