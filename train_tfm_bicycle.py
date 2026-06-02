import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Support both "python train_tfm_bicycle.py" from this directory and imports
# from the repository root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rvq_model import ResidualVQ
from utils import preprocess_and_save_norm_params, load_sampled_datas


def _as_batch_vector(value, ref: torch.Tensor, batch_size: int, default: float):
    """Convert scalar/[B]/[B,1] initial state to [B] on ref's device."""
    if value is None:
        return torch.full((batch_size,), default, device=ref.device, dtype=ref.dtype)

    if torch.is_tensor(value):
        out = value.to(device=ref.device, dtype=ref.dtype)
    else:
        out = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)

    if out.dim() == 0:
        out = out.expand(batch_size)
    elif out.dim() == 2 and out.shape[1] == 1:
        out = out[:, 0]

    if out.dim() != 1 or out.shape[0] != batch_size:
        raise ValueError(f"Initial state should be scalar, [B], or [B, 1], got {tuple(out.shape)}")
    return out


def bicycle_rollout_from_controls(
    acc: torch.Tensor,
    yaw_rate: torch.Tensor,
    v0: torch.Tensor,
    yaw0=None,
    x0=None,
    y0=None,
    dt: float = 0.2,
):
    """
    Differentiable unicycle-like rollout from acceleration and yaw-rate controls.

    Args:
        acc:      [B, T] longitudinal acceleration in m/s^2.
        yaw_rate: [B, T] yaw rate in rad/s.
        v0:       [B] or [B, 1] initial speed before the first rollout step.
        yaw0:     optional scalar/[B]/[B,1] initial yaw, defaults to 0.
        x0, y0:   optional scalar/[B]/[B,1] initial global position, defaults to 0.
        dt:       time step in seconds.

    Returns:
        x, y, yaw, v: each [B, T].
    """
    if acc.shape != yaw_rate.shape or acc.dim() != 2:
        raise ValueError("acc and yaw_rate should both have shape [B, T]")

    batch_size = acc.shape[0]

    # Keep integration in float32 under AMP to make cumsum/trig more stable.
    acc_f = acc.float()
    yaw_rate_f = yaw_rate.float()
    v0_f = _as_batch_vector(v0, acc_f, batch_size, 0.0).float()
    yaw0_f = _as_batch_vector(yaw0, acc_f, batch_size, 0.0).float()
    x0_f = _as_batch_vector(x0, acc_f, batch_size, 0.0).float()
    y0_f = _as_batch_vector(y0, acc_f, batch_size, 0.0).float()

    # Make the indexing explicit:
    #   v[:, 0] = v0
    #   v[:, t] = v[:, t-1] + acc[:, t] * dt, t >= 1
    # This matches the GT target construction where gt_v[:, 0]=dx[:, 0]/dt
    # and gt_acc[:, 0] is fixed to 0 instead of being used to update v0.
    v = torch.empty_like(acc_f)
    v[:, 0] = v0_f
    if acc_f.shape[1] > 1:
        v[:, 1:] = v0_f.unsqueeze(1) + torch.cumsum(acc_f[:, 1:], dim=1) * dt
    v = torch.clamp(v, min=-40.0, max=40.0)

    yaw = yaw0_f.unsqueeze(1) + torch.cumsum(yaw_rate_f, dim=1) * dt

    dx_global = v * torch.cos(yaw) * dt
    dy_global = v * torch.sin(yaw) * dt

    x = x0_f.unsqueeze(1) + torch.cumsum(dx_global, dim=1)
    y = y0_f.unsqueeze(1) + torch.cumsum(dy_global, dim=1)
    return x, y, yaw, v


def compute_gt_bicycle_targets_from_dxdydyaw(gt_phys: torch.Tensor, dt: float = 0.2):
    """
    Build physical supervision targets from body-frame dxdydyaw.

    Args:
        gt_phys: [B, T, 3] physical-space dxdydyaw.
        dt: time step in seconds.

    Returns:
        Dict with [B, T] tensors: gt_x, gt_y, gt_yaw, gt_v, gt_acc, gt_yaw_rate.
    """
    if gt_phys.dim() != 3 or gt_phys.shape[-1] != 3:
        raise ValueError("gt_phys should have shape [B, T, 3]")

    dx = gt_phys[..., 0]
    dy = gt_phys[..., 1]
    dyaw = gt_phys[..., 2]

    gt_yaw = torch.cumsum(dyaw, dim=1)

    prev_yaw = torch.zeros_like(gt_yaw)
    prev_yaw[:, 1:] = gt_yaw[:, :-1]

    dx_global = dx * torch.cos(prev_yaw) - dy * torch.sin(prev_yaw)
    dy_global = dx * torch.sin(prev_yaw) + dy * torch.cos(prev_yaw)

    gt_x = torch.cumsum(dx_global, dim=1)
    gt_y = torch.cumsum(dy_global, dim=1)

    denom = dt + 1e-8
    gt_v = dx / denom
    gt_acc = torch.zeros_like(gt_v)
    if gt_v.shape[1] > 1:
        gt_acc[:, 1:] = (gt_v[:, 1:] - gt_v[:, :-1]) / denom
    gt_yaw_rate = dyaw / denom

    return {
        "gt_x": gt_x,
        "gt_y": gt_y,
        "gt_yaw": gt_yaw,
        "gt_v": gt_v,
        "gt_acc": gt_acc,
        "gt_yaw_rate": gt_yaw_rate,
    }


def global_xyyaw_to_dxdydyaw(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor):
    """
    Convert global x/y/yaw profiles back to body-frame dxdydyaw.

    Args:
        x, y, yaw: [B, T] global position and yaw profiles.

    Returns:
        pred_phys: [B, T, 3] body-frame dxdydyaw.
    """
    if x.shape != y.shape or x.shape != yaw.shape or x.dim() != 2:
        raise ValueError("x, y, and yaw should all have shape [B, T]")

    dx_global = torch.zeros_like(x)
    dy_global = torch.zeros_like(y)
    dx_global[:, 0] = x[:, 0]
    dy_global[:, 0] = y[:, 0]
    if x.shape[1] > 1:
        dx_global[:, 1:] = x[:, 1:] - x[:, :-1]
        dy_global[:, 1:] = y[:, 1:] - y[:, :-1]

    prev_yaw = torch.zeros_like(yaw)
    if yaw.shape[1] > 1:
        prev_yaw[:, 1:] = yaw[:, :-1]

    dx_body = dx_global * torch.cos(prev_yaw) + dy_global * torch.sin(prev_yaw)
    dy_body = -dx_global * torch.sin(prev_yaw) + dy_global * torch.cos(prev_yaw)

    dyaw = torch.zeros_like(yaw)
    dyaw[:, 0] = yaw[:, 0]
    if yaw.shape[1] > 1:
        dyaw[:, 1:] = yaw[:, 1:] - yaw[:, :-1]

    return torch.stack([dx_body, dy_body, dyaw], dim=-1)


def get_rollout_horizon(epoch: int, epochs: int, total_steps: int):
    """
    Progressive rollout horizon.

    这个 bicycle 版本不是直接预测 dxdydyaw，而是预测 acc/yaw_rate 后
    递推积分得到 x/y/yaw。积分时间越长，早期小误差越容易累积放大。
    因此训练时先只监督短 horizon，再平滑扩展到完整 25 steps≈5s。

    这里不用 5/10/15/T 的硬阶梯，而是用 smoothstep 曲线：
      - epoch=0 时 H≈5，即先学 1s 短期积分；
      - 中间随训练进度平滑增长；
      - 最后 H=T，完整监督 5s rollout。
    H 仍然需要取整，因为它用于 tensor slicing。
    """
    min_steps = min(total_steps, 5)
    if total_steps <= min_steps:
        return total_steps

    denom = max(epochs - 1, 1)
    progress = min(max(epoch / denom, 0.0), 1.0)
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    horizon = round(min_steps + (total_steps - min_steps) * smooth_progress)
    return int(min(max(horizon, min_steps), total_steps))


class TrajRVQBicycleTransformer(nn.Module):
    """
    RVQ tokenizer whose decoder predicts acc/yaw_rate and reconstructs
    dxdydyaw through a differentiable bicycle-like rollout.

    与 train_tfm.py 的主要区别：
      - train_tfm.py decoder 直接输出 v/kappa/dy，再组合成 dxdydyaw；
      - 这里 decoder 只输出两个逐时刻标量控制量 acc 和 yaw_rate；
      - acc/yaw_rate 结合初始状态 v0，通过可微分 rollout 积分成
        v/yaw/x/y，再转换回 body-frame dxdydyaw 做重建监督。
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
        acc_max: float = 8.0,
        yaw_rate_max: float = 1.0,
    ):
        super().__init__()

        self.input_steps = input_steps
        self.input_dim = input_dim
        self.input_flat_dim = input_steps * input_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dt = dt
        self.acc_max = acc_max
        self.yaw_rate_max = yaw_rate_max

        # Encoder: same structure as TrajRVQTransformer.
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
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # RVQ bottleneck.
        self.rvq = ResidualVQ(
            num_quantizers=num_layers,
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            dropout=0.2,
            commitment_cost=0.25,
        )

        # Decoder: transformer trunk plus control heads.
        # 这里不再直接预测 dx/dy/dyaw，也不预测 v/kappa/dy。
        # decoder 只负责输出每个时间步的两个标量控制量：
        #   acc:      [B, T] 纵向加速度
        #   yaw_rate: [B, T] 航向角速度
        # 后续轨迹由初始速度 v0 和这两个控制量递推积分得到。
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
            decoder_layer,
            num_layers=num_transformer_layers,
        )

        self.acc_head = nn.Linear(d_model, input_steps)
        self.yaw_rate_head = nn.Linear(d_model, input_steps)

        # Normalization buffers saved with state_dict.
        self.register_buffer("norm_mean", torch.zeros(1, 1, 3))
        self.register_buffer("norm_std", torch.ones(1, 1, 3))
        self.register_buffer("norm_scale", torch.ones(1, 1, 3))

    def set_norm_params(self, mean, std, scale_factor):
        """Set normalization parameters produced by preprocess_and_save_norm_params."""
        mean_t = torch.as_tensor(mean, device=self.norm_mean.device, dtype=self.norm_mean.dtype)
        std_t = torch.as_tensor(std, device=self.norm_std.device, dtype=self.norm_std.dtype)
        scale_t = torch.as_tensor(scale_factor, device=self.norm_scale.device, dtype=self.norm_scale.dtype)
        self.norm_mean.copy_(mean_t)
        self.norm_std.copy_(std_t)
        self.norm_scale.copy_(scale_t)

    def to_phys(self, x_norm: torch.Tensor):
        return (x_norm * self.norm_scale * self.norm_std) + self.norm_mean

    def to_norm(self, x_phys: torch.Tensor):
        return (x_phys - self.norm_mean) / (self.norm_std + 1e-8) / (self.norm_scale + 1e-8)

    def encode(self, x_norm: torch.Tensor):
        """
        Args:
            x_norm: [B, T, 3] normalized dxdydyaw.

        Returns:
            z: [B, D] trajectory-level latent.
        """
        batch_size, steps, dim = x_norm.shape
        if steps != self.input_steps or dim != self.input_dim:
            raise ValueError(
                f"Expected input shape [B, {self.input_steps}, {self.input_dim}], "
                f"got {tuple(x_norm.shape)}"
            )

        x_flat = x_norm.view(batch_size, self.input_flat_dim)
        h = self.input_proj(x_flat).unsqueeze(1)
        h = h + self.pos_embed
        h = self.transformer_encoder(h).squeeze(1)
        return self.to_latent(h)

    def _decode_controls(self, h_dec: torch.Tensor):
        # 控制量做 tanh 限幅，避免训练早期 acc/yaw_rate 过大导致
        # 积分出的速度和位置爆炸。
        acc = self.acc_max * torch.tanh(self.acc_head(h_dec) / self.acc_max)
        yaw_rate = self.yaw_rate_max * torch.tanh(
            self.yaw_rate_head(h_dec) / self.yaw_rate_max
        )
        return acc, yaw_rate

    def _decode_from_latent(self, z_q: torch.Tensor, v0=None):
        h_dec = z_q.unsqueeze(1)
        h_dec = h_dec + self.decoder_pos_embed
        h_dec = self.transformer_decoder(h_dec).squeeze(1)

        acc, yaw_rate = self._decode_controls(h_dec)

        # 从控制量 rollout 到全局轨迹：
        #   v_t   = v_{t-1} + acc_t * dt
        #   yaw_t = yaw_{t-1} + yaw_rate_t * dt
        #   x/y   = x/y + v_t * [cos(yaw_t), sin(yaw_t)] * dt
        # 训练 forward 中 v0 来自 GT 初始速度；纯 token decode 时如果
        # 外部不传 v0，则 rollout 函数会退化为 v0=0。
        x_global, y_global, yaw, v = bicycle_rollout_from_controls(
            acc=acc,
            yaw_rate=yaw_rate,
            v0=v0,
            dt=self.dt,
        )
        # 评估和下游仍然需要 dxdydyaw，因此把全局 x/y/yaw 再旋回
        # body-frame delta，并使用训练保存的 norm 参数转回 normalized 空间。
        pred_phys = global_xyyaw_to_dxdydyaw(x_global, y_global, yaw)
        x_recon_norm = self.to_norm(pred_phys)

        return {
            "x_recon": x_recon_norm,
            "pred_phys": pred_phys,
            "acc": acc.float(),
            "yaw_rate": yaw_rate.float(),
            "v": v,
            "yaw": yaw,
            "x_global": x_global,
            "y_global": y_global,
        }

    def decode_from_codes(self, codes: torch.Tensor, v0=None, return_dict: bool = False):
        """
        Decode token codes to normalized dxdydyaw.

        Training forward can infer v0 from the input trajectory. Pure token
        decoding has no context, so v0 defaults to 0 unless the caller passes it.
        The default return value is [B, T, 3] to match existing eval helpers.
        """
        z_q = self.rvq.decode_from_codes(codes)
        out = self._decode_from_latent(z_q, v0=v0)
        if return_dict:
            return out
        return out["x_recon"]

    def forward(self, x_norm: torch.Tensor, v0=None):
        """
        Args:
            x_norm: [B, T, 3] normalized dxdydyaw.
            v0: optional [B] or [B,1]. If None, use gt signed speed at step 0.

        Returns:
            Dict with x_recon, pred_phys, vq_loss, codes, acc, yaw_rate, v, yaw,
            x_global, and y_global.
        """
        if v0 is None:
            gt_phys = self.to_phys(x_norm)
            gt_targets = compute_gt_bicycle_targets_from_dxdydyaw(gt_phys, dt=self.dt)
            v0 = gt_targets["gt_v"][:, 0]

        z = self.encode(x_norm)
        z_q, vq_loss, codes = self.rvq(z)
        out = self._decode_from_latent(z_q, v0=v0)
        out["vq_loss"] = vq_loss
        out["codes"] = codes
        return out


def _build_dataloader(data_normalized: np.ndarray, batch_size: int):
    dataset = TensorDataset(torch.FloatTensor(data_normalized))
    num_workers = min(4, os.cpu_count() or 1)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    return DataLoader(dataset, **kwargs)


def _make_scheduler(optimizer, epochs: int, initial_lr: float):
    warmup_epochs = min(5, max(1, epochs))
    warmup_start_lr = 1e-5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_lr / initial_lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    if epochs <= warmup_epochs:
        return warmup_scheduler

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def train_rvq_bicycle(
    data_array: np.ndarray,
    save_dir: str = "./work_dirs/tokenizer/rvq_tfm_bicycle_0526",
    data_type: str = "pred",
    batch_size: int = 4096,
    num_layers: int = 15,
    num_transformer_layers: int = 2,
    epochs: int = 500,
    dt: float = 0.2,
    acc_max: float = 8.0,
    yaw_rate_max: float = 1.0,
):
    """
    Train the acc/yaw_rate rollout RVQ tokenizer.

    Args:
        data_array: [N, T, 3] physical-space dxdydyaw.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_array = np.asarray(data_array, dtype=np.float32)
    num_steps = data_array.shape[1]
    input_dim = data_array.shape[2]

    data_normalized = preprocess_and_save_norm_params(data_array, save_dir, data_type)
    dataloader = _build_dataloader(data_normalized, batch_size=batch_size)

    model = TrajRVQBicycleTransformer(
        input_steps=num_steps,
        input_dim=input_dim,
        num_layers=num_layers,
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=num_transformer_layers,
        dt=dt,
        acc_max=acc_max,
        yaw_rate_max=yaw_rate_max,
    ).to(device)

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
        print("Using FP32 training")

    initial_lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = _make_scheduler(optimizer, epochs=epochs, initial_lr=initial_lr)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard", run_name))
    print("Start Training (Bicycle RVQ Transformer)...")

    recon_loss_weight = 10.0
    xy_loss_weight = 1.0
    yaw_loss_weight = 2.0
    v_loss_weight = 0.5
    control_loss_weight = 0.2
    vq_loss_weight = 5.0
    yaw_rate_weight = 1.0

    max_lateral = 1.0

    for epoch in range(epochs):
        model.train()
        # Progressive horizon: 模型每次 forward 仍然输出完整 T 步，
        # 但 loss 先只看前 H 步。这样先把 1s/2s 的短期积分学稳，
        # 再逐渐扩展到完整 5s，降低长时间积分误差累积带来的训练难度。
        horizon = get_rollout_horizon(epoch, epochs, num_steps)
        smooth_loss_weight = 1e-3 if epoch > 30 else 0.0

        total_recon = 0.0
        total_xy = 0.0
        total_yaw = 0.0
        total_v = 0.0
        total_control = 0.0
        total_smooth = 0.0
        total_vq = 0.0
        total_loss = 0.0

        total_ade = 0.0
        total_fde = 0.0
        total_vrr_count = 0
        total_samples = 0
        total_acc_mean = 0.0
        total_acc_abs = 0.0
        total_yaw_rate_abs = 0.0
        total_v_abs = 0.0
        running_v_min = float("inf")
        running_v_max = float("-inf")

        for batch in dataloader:
            x_norm = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(x_norm)
                gt_phys = model.to_phys(x_norm)
                gt = compute_gt_bicycle_targets_from_dxdydyaw(gt_phys, dt=dt)

                # 轨迹相关 loss 都只截取当前 curriculum horizon。
                # H 会随 epoch 从 5/10/15 逐步增长到 T。
                pred_recon_h = out["x_recon"][:, :horizon]
                target_recon_h = x_norm[:, :horizon]

                mse_dxdy = F.mse_loss(pred_recon_h[..., :2], target_recon_h[..., :2])
                mse_dyaw = F.mse_loss(pred_recon_h[..., 2], target_recon_h[..., 2])
                recon_loss = mse_dxdy + 14.0 * mse_dyaw

                xy_loss = (
                    F.mse_loss(out["x_global"][:, :horizon], gt["gt_x"][:, :horizon])
                    + F.mse_loss(out["y_global"][:, :horizon], gt["gt_y"][:, :horizon])
                )
                yaw_loss = F.mse_loss(out["yaw"][:, :horizon], gt["gt_yaw"][:, :horizon])
                v_loss = F.mse_loss(out["v"][:, :horizon], gt["gt_v"][:, :horizon])
                control_loss = (
                    F.mse_loss(out["acc"][:, :horizon], gt["gt_acc"][:, :horizon])
                    + yaw_rate_weight
                    * F.mse_loss(out["yaw_rate"][:, :horizon], gt["gt_yaw_rate"][:, :horizon])
                )

                if horizon > 1:
                    jerk = (out["acc"][:, 1:horizon] - out["acc"][:, : horizon - 1]) / dt
                    yaw_acc = (
                        out["yaw_rate"][:, 1:horizon] - out["yaw_rate"][:, : horizon - 1]
                    ) / dt
                    smooth_loss = jerk.pow(2).mean() + yaw_acc.pow(2).mean()
                else:
                    smooth_loss = out["acc"].sum() * 0.0

                vq_loss = out["vq_loss"]
                loss = (
                    recon_loss_weight * recon_loss
                    + xy_loss_weight * xy_loss
                    + yaw_loss_weight * yaw_loss
                    + v_loss_weight * v_loss
                    + control_loss_weight * control_loss
                    + smooth_loss_weight * smooth_loss
                    + vq_loss_weight * vq_loss
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
            total_xy += xy_loss.item()
            total_yaw += yaw_loss.item()
            total_v += v_loss.item()
            total_control += control_loss.item()
            total_smooth += smooth_loss.item()
            total_vq += vq_loss.item()
            total_loss += loss.item()

            with torch.no_grad():
                step_dist = torch.sqrt(
                    (out["x_global"][:, :horizon] - gt["gt_x"][:, :horizon]).pow(2)
                    + (out["y_global"][:, :horizon] - gt["gt_y"][:, :horizon]).pow(2)
                    + 1e-6
                )
                batch_size_actual = x_norm.shape[0]
                total_ade += step_dist.mean(dim=1).sum().item()
                total_fde += step_dist[:, -1].sum().item()
                total_vrr_count += (step_dist.max(dim=1)[0] < max_lateral).sum().item()
                total_samples += batch_size_actual

                total_acc_mean += out["acc"][:, :horizon].mean().item()
                total_acc_abs += out["acc"][:, :horizon].abs().mean().item()
                total_yaw_rate_abs += out["yaw_rate"][:, :horizon].abs().mean().item()
                total_v_abs += out["v"][:, :horizon].abs().mean().item()
                running_v_min = min(running_v_min, out["v"][:, :horizon].min().item())
                running_v_max = max(running_v_max, out["v"][:, :horizon].max().item())

        scheduler.step()

        num_batches = len(dataloader)
        avg_recon = total_recon / num_batches
        avg_xy = total_xy / num_batches
        avg_yaw = total_yaw / num_batches
        avg_v = total_v / num_batches
        avg_control = total_control / num_batches
        avg_smooth = total_smooth / num_batches
        avg_vq = total_vq / num_batches
        avg_loss = total_loss / num_batches

        avg_ade = total_ade / total_samples if total_samples > 0 else 0.0
        avg_fde = total_fde / total_samples if total_samples > 0 else 0.0
        vrr = total_vrr_count / total_samples if total_samples > 0 else 0.0
        avg_acc_mean = total_acc_mean / num_batches
        avg_acc_abs = total_acc_abs / num_batches
        avg_yaw_rate_abs = total_yaw_rate_abs / num_batches
        avg_v_abs = total_v_abs / num_batches

        writer.add_scalar("loss/recon", avg_recon, epoch + 1)
        writer.add_scalar("loss/xy", avg_xy, epoch + 1)
        writer.add_scalar("loss/yaw", avg_yaw, epoch + 1)
        writer.add_scalar("loss/v", avg_v, epoch + 1)
        writer.add_scalar("loss/control", avg_control, epoch + 1)
        writer.add_scalar("loss/smooth", avg_smooth, epoch + 1)
        writer.add_scalar("loss/vq", avg_vq, epoch + 1)
        writer.add_scalar("loss/total", avg_loss, epoch + 1)
        writer.add_scalar("metrics/ade", avg_ade, epoch + 1)
        writer.add_scalar("metrics/fde", avg_fde, epoch + 1)
        writer.add_scalar("metrics/vrr_1m", vrr, epoch + 1)
        writer.add_scalar("stats/acc_abs_mean", avg_acc_abs, epoch + 1)
        writer.add_scalar("stats/yaw_rate_abs_mean", avg_yaw_rate_abs, epoch + 1)
        writer.add_scalar("stats/v_abs_mean", avg_v_abs, epoch + 1)
        writer.add_scalar("stats/acc_mean", avg_acc_mean, epoch + 1)
        writer.add_scalar("stats/v_min", running_v_min, epoch + 1)
        writer.add_scalar("stats/v_max", running_v_max, epoch + 1)
        writer.add_scalar("train/rollout_horizon", horizon, epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"[BiRVQ] Epoch {epoch+1:03d} | H: {horizon:02d} | "
                f"Recon: {avg_recon:.5f} | XY: {avg_xy:.5f} | "
                f"Yaw: {avg_yaw:.5f} | V: {avg_v:.5f} | "
                f"Ctrl: {avg_control:.5f} | Smooth: {avg_smooth:.5f} | "
                f"VQ: {avg_vq:.5f} | TrajErr: {avg_ade:.4f} | "
                f"EndErr: {avg_fde:.4f} | VRR: {vrr:.4f} | "
                f"acc_mean: {avg_acc_mean:.3f} | acc_abs: {avg_acc_abs:.3f} | "
                f"yaw_rate_abs: {avg_yaw_rate_abs:.3f} | "
                f"v_min: {running_v_min:.2f} | v_max: {running_v_max:.2f}"
            )

    model_path = os.path.join(save_dir, f"{data_type}_rvq_bicycle_model.pth")
    torch.save(model.state_dict(), model_path)

    config = {
        "model_type": "TrajRVQBicycleTransformer",
        "dt": dt,
        "num_layers": num_layers,
        "vocab_size": 1024,
        "d_model": 128,
        "num_transformer_layers": num_transformer_layers,
        "acc_max": acc_max,
        "yaw_rate_max": yaw_rate_max,
        "input_steps": num_steps,
        "input_dim": input_dim,
    }
    with open(os.path.join(save_dir, f"{data_type}_rvq_bicycle_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    writer.close()
    print(f"Bicycle RVQ training done. Model saved to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train acc/yaw_rate bicycle RVQ tokenizer.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./work_dirs/tokenizer/rvq_tfm_bicycle_0526")
    parser.add_argument("--data-type", type=str, default="pred", choices=["pred", "history"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--acc-max", type=float, default=8.0)
    parser.add_argument("--yaw-rate-max", type=float, default=1.0)
    args = parser.parse_args()

    sampled_trajs = load_sampled_datas(args.data_path)
    sampled_trajs = np.asarray(sampled_trajs, dtype=np.float32)
    if args.data_type == "history":
        sampled_trajs = sampled_trajs[:, :14, :]
    if args.max_samples > 0:
        sampled_trajs = sampled_trajs[: args.max_samples]

    print(
        f"Train config | data_type={args.data_type} | num_layers={args.num_layers} | "
        f"num_transformer_layers={args.num_transformer_layers} | batch_size={args.batch_size} | "
        f"epochs={args.epochs} | dt={args.dt} | acc_max={args.acc_max} | "
        f"yaw_rate_max={args.yaw_rate_max} | save_dir={args.save_dir}"
    )
    print(f"Dataset shape: {sampled_trajs.shape}")

    train_rvq_bicycle(
        sampled_trajs,
        save_dir=args.save_dir,
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_transformer_layers=args.num_transformer_layers,
        epochs=args.epochs,
        dt=args.dt,
        acc_max=args.acc_max,
        yaw_rate_max=args.yaw_rate_max,
    )
