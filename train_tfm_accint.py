import os
import pickle
from datetime import datetime
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AccBoundaryRVQTokenizer(nn.Module):
    """
    思路：
    1) Decoder 只预测加速度序列（ax, ay, alpha）和末端边界（end_x, end_y, end_yaw）；
    2) 利用末端边界反解初速度（v0x, v0y, w0）；
    3) 积分得到速度、位移、航向，再反投影到 body 系得到 dxdydyaw。

    这样可把“首尾位置约束”直接注入重建过程。
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

        # 预测加速度与末端边界
        self.acc_xy_head = nn.Linear(d_model, input_steps * 2)
        self.acc_yaw_head = nn.Linear(d_model, input_steps)
        self.end_xy_head = nn.Linear(d_model, 2)
        self.end_yaw_head = nn.Linear(d_model, 1)

        # 参数范围（物理先验）
        self.acc_xy_scale = 8.0      # m/s^2
        self.acc_yaw_scale = 1.2     # rad/s^2
        self.end_xy_scale = 250.0    # m
        self.end_yaw_scale = 4.5     # rad

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

        end_xy = self.end_xy_head(h_dec)
        end_xy = torch.tanh(end_xy) * self.end_xy_scale

        end_yaw = self.end_yaw_head(h_dec).squeeze(-1)
        end_yaw = torch.tanh(end_yaw) * self.end_yaw_scale

        return acc_xy, acc_yaw, end_xy, end_yaw

    def _rollout_with_boundary(
        self,
        acc_xy: torch.Tensor,
        acc_yaw: torch.Tensor,
        end_xy: torch.Tensor,
        end_yaw: torch.Tensor,
    ):
        """
        根据加速度和末端边界反解初速度，然后积分得到轨迹。

        离散形式：
            v_t = v0 + dt * cumsum(a)_t
            p_T = sum_t (v_t * dt)
        =>  v0 = (p_T - dt^2 * sum_t cumsum(a)_t) / (T * dt)
        """
        dt = self.dt
        t_steps = self.input_steps
        denom = t_steps * dt + 1e-8

        # XY 部分（全局系）
        acc_xy_cum = torch.cumsum(acc_xy, dim=1)                # [B, T, 2]
        acc_xy_term = (dt * dt) * torch.sum(acc_xy_cum, dim=1)  # [B, 2]
        v0_xy = (end_xy - acc_xy_term) / denom                  # [B, 2]

        vel_xy = v0_xy.unsqueeze(1) + dt * acc_xy_cum           # [B, T, 2]
        disp_xy_global = vel_xy * dt                             # [B, T, 2]
        pos_xy_global = torch.cumsum(disp_xy_global, dim=1)      # [B, T, 2]

        # Yaw 部分
        acc_yaw_cum = torch.cumsum(acc_yaw, dim=1)               # [B, T]
        acc_yaw_term = (dt * dt) * torch.sum(acc_yaw_cum, dim=1) # [B]
        w0 = (end_yaw - acc_yaw_term) / denom                    # [B]

        yaw_rate = w0.unsqueeze(1) + dt * acc_yaw_cum            # [B, T]
        dyaw = yaw_rate * dt                                     # [B, T]
        yaw = torch.cumsum(dyaw, dim=1)                          # [B, T]

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

        speed = torch.norm(vel_xy, dim=-1)  # [B, T]

        aux = {
            "acc_xy": acc_xy,
            "acc_yaw": acc_yaw,
            "v0_xy": v0_xy,
            "w0": w0,
            "vel_xy": vel_xy,
            "speed": speed,
            "disp_xy_global": disp_xy_global,
            "pos_xy_global": pos_xy_global,
            "yaw": yaw,
            "end_xy": end_xy,
            "end_yaw": end_yaw,
        }
        return x_norm, aux

    def _decode_from_latent(self, z_q: torch.Tensor):
        h_dec = z_q.unsqueeze(1)
        h_dec = h_dec + self.decoder_pos_embed
        h_dec = self.transformer_decoder(h_dec).squeeze(1)

        acc_xy, acc_yaw, end_xy, end_yaw = self._decode_heads(h_dec)
        return self._rollout_with_boundary(acc_xy, acc_yaw, end_xy, end_yaw)

    def decode_from_codes(self, codes: torch.Tensor):
        z_q = self.rvq.decode_from_codes(codes)
        x_recon, _ = self._decode_from_latent(z_q)
        return x_recon

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        z_q, vq_loss, codes = self.rvq(z)
        x_recon, aux = self._decode_from_latent(z_q)
        return x_recon, vq_loss, codes, aux


def train_rvq_accint(
    data_array: np.ndarray,
    save_dir: str = "./work_dirs/tokenizer/rvq_tfm_accint",
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
    model = AccBoundaryRVQTokenizer(
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

    print("Start Training (Acceleration-Integration RVQ Tokenizer)...")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))

    max_lateral = 1.0

    for epoch in range(epochs):
        model.train()

        total_recon = 0.0
        total_vq = 0.0
        total_end_xy = 0.0
        total_end_yaw = 0.0
        total_traj = 0.0
        total_speed_boundary = 0.0
        total_speed_profile = 0.0
        total_acc_smooth = 0.0

        total_ade = 0.0
        total_fde = 0.0
        total_vrr_count = 0
        total_samples = 0

        if epoch > epochs * 0.8:
            model.rvq.dropout = 0.0

        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                x_recon, vq_loss, _, aux = model(x)

                # ------- GT 转物理空间并积分出边界 -------
                gt_phys = model.to_phys(x)
                gt_pos_xy, gt_yaw, _ = integrate_local_to_global_with_yaw(gt_phys)
                gt_end_xy = gt_pos_xy[:, -1, :]
                gt_end_yaw = gt_yaw[:, -1]

                # ------- 1) 重建损失（归一化空间） -------
                mse_dxdy = F.mse_loss(x_recon[..., :2], x[..., :2])
                mse_dyaw = F.mse_loss(x_recon[..., 2], x[..., 2])
                recon_loss = mse_dxdy + 14.0 * mse_dyaw

                # ------- 2) 末端边界损失 -------
                end_xy_loss = F.mse_loss(aux["end_xy"], gt_end_xy)
                end_yaw_loss = F.mse_loss(aux["end_yaw"], gt_end_yaw)

                # ------- 3) 全局轨迹损失 -------
                traj_global_loss = F.mse_loss(aux["pos_xy_global"], gt_pos_xy)

                # ------- 4) 速度约束（由首尾位置得到平均速度） -------
                target_avg_speed = torch.norm(gt_end_xy, dim=-1) / (model.input_steps * model.dt + 1e-8)
                pred_avg_speed = aux["speed"].mean(dim=1)
                speed_boundary_loss = F.mse_loss(pred_avg_speed, target_avg_speed)

                # 速度 profile 约束（辅助稳定收敛）
                pred_phys = model.to_phys(x_recon)
                pred_speed = torch.sqrt(pred_phys[..., 0] ** 2 + pred_phys[..., 1] ** 2 + 1e-6) / model.dt
                gt_speed = torch.sqrt(gt_phys[..., 0] ** 2 + gt_phys[..., 1] ** 2 + 1e-6) / model.dt
                speed_profile_loss = F.mse_loss(pred_speed, gt_speed)

                # ------- 5) 加速度平滑项 -------
                if model.input_steps > 1:
                    jerk_xy = (aux["acc_xy"][:, 1:, :] - aux["acc_xy"][:, :-1, :]) / model.dt
                    jerk_yaw = (aux["acc_yaw"][:, 1:] - aux["acc_yaw"][:, :-1]) / model.dt
                    acc_smooth_loss = jerk_xy.pow(2).mean() + jerk_yaw.pow(2).mean()
                else:
                    acc_smooth_loss = aux["acc_xy"].sum() * 0.0

                # loss 权重
                recon_w = 10.0
                vq_w = 5.0
                end_xy_w = 2.0
                end_yaw_w = 1.0
                traj_w = 2.0
                speed_boundary_w = 1.0
                speed_profile_w = 0.5
                acc_smooth_w = 1e-3 if epoch > 20 else 0.0

                loss = (
                    recon_w * recon_loss
                    + vq_w * vq_loss
                    + end_xy_w * end_xy_loss
                    + end_yaw_w * end_yaw_loss
                    + traj_w * traj_global_loss
                    + speed_boundary_w * speed_boundary_loss
                    + speed_profile_w * speed_profile_loss
                    + acc_smooth_w * acc_smooth_loss
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

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_end_xy += end_xy_loss.item()
            total_end_yaw += end_yaw_loss.item()
            total_traj += traj_global_loss.item()
            total_speed_boundary += speed_boundary_loss.item()
            total_speed_profile += speed_profile_loss.item()
            total_acc_smooth += acc_smooth_loss.item()

            # 指标：ADE / FDE / VRR
            with torch.no_grad():
                step_err = torch.norm(aux["pos_xy_global"] - gt_pos_xy, dim=-1)  # [B, T]
                ade = step_err.mean(dim=1).mean().item()
                fde = step_err[:, -1].mean().item()
                valid = (step_err.max(dim=1)[0] < max_lateral).sum().item()

                total_ade += ade
                total_fde += fde
                total_vrr_count += valid
                total_samples += x.shape[0]

        scheduler.step()

        avg_recon = total_recon / len(dataloader)
        avg_vq = total_vq / len(dataloader)
        avg_end_xy = total_end_xy / len(dataloader)
        avg_end_yaw = total_end_yaw / len(dataloader)
        avg_traj = total_traj / len(dataloader)
        avg_speed_boundary = total_speed_boundary / len(dataloader)
        avg_speed_profile = total_speed_profile / len(dataloader)
        avg_acc_smooth = total_acc_smooth / len(dataloader)

        weighted_loss = (
            10.0 * avg_recon
            + 5.0 * avg_vq
            + 2.0 * avg_end_xy
            + 1.0 * avg_end_yaw
            + 2.0 * avg_traj
            + 1.0 * avg_speed_boundary
            + 0.5 * avg_speed_profile
            + (1e-3 if epoch > 20 else 0.0) * avg_acc_smooth
        )

        writer.add_scalar("loss/recon", avg_recon, epoch + 1)
        writer.add_scalar("loss/vq", avg_vq, epoch + 1)
        writer.add_scalar("loss/end_xy", avg_end_xy, epoch + 1)
        writer.add_scalar("loss/end_yaw", avg_end_yaw, epoch + 1)
        writer.add_scalar("loss/traj_global", avg_traj, epoch + 1)
        writer.add_scalar("loss/speed_boundary", avg_speed_boundary, epoch + 1)
        writer.add_scalar("loss/speed_profile", avg_speed_profile, epoch + 1)
        writer.add_scalar("loss/acc_smooth", avg_acc_smooth, epoch + 1)
        writer.add_scalar("loss/weighted", weighted_loss, epoch + 1)

        if (epoch + 1) % 10 == 0:
            avg_ade = total_ade / len(dataloader)
            avg_fde = total_fde / len(dataloader)
            vrr = total_vrr_count / total_samples if total_samples > 0 else 0.0
            with torch.no_grad():
                mean_speed = aux["speed"].mean().item()
                mean_end_dist = torch.norm(aux["end_xy"], dim=-1).mean().item()
                mean_acc = aux["acc_xy"].norm(dim=-1).mean().item()

            print(
                f"[AccIntRVQ] Epoch {epoch+1:03d} | "
                f"Recon: {avg_recon:.5f} | VQ: {avg_vq:.5f} | "
                f"EndXY: {avg_end_xy:.5f} | EndYaw: {avg_end_yaw:.5f} | "
                f"Traj: {avg_traj:.5f} | SpeedB: {avg_speed_boundary:.5f} | "
                f"SpeedP: {avg_speed_profile:.5f} | AccSmooth: {avg_acc_smooth:.5f} | "
                f"ADE: {avg_ade:.4f} m | FDE: {avg_fde:.4f} m | VRR: {vrr:.4f} | "
                f"speed_mean: {mean_speed:.2f} m/s | end_dist_mean: {mean_end_dist:.2f} m | "
                f"|acc|_mean: {mean_acc:.2f}"
            )

    # 保存模型
    model_path = os.path.join(save_dir, f"{data_type}_rvq_accint_model.pth")
    torch.save(model.state_dict(), model_path)
    writer.close()
    print(f"Acceleration-Integration RVQ training done. Model saved to {model_path}")


if __name__ == "__main__":
    batch_size = 4096
    sampled_trajs = load_sampled_datas()

    save_dir = "./work_dirs/tokenizer/rvq_tfm_accint_0423"
    data_type = "pred"  # 'pred' or 'history'
    print("data_type:", data_type)

    if data_type == "history":
        sampled_trajs = sampled_trajs[:, :14, :]

    train_rvq_accint(sampled_trajs, save_dir, data_type, batch_size)
