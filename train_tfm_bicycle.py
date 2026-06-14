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


def get_rollout_horizon_float(
    epoch: int,
    epochs: int,
    total_steps: int,
    full_horizon_ratio: float = 0.7,
):
    """
    Continuous progressive rollout horizon.

    这个 bicycle 版本不是直接预测 dxdydyaw，而是预测 acc/yaw_rate 后
    递推积分得到 x/y/yaw。积分时间越长，早期小误差越容易累积放大。
    因此训练时先强调短 horizon，再平滑扩展到完整 25 steps≈5s。

    这里用 smoothstep 曲线：
      - epoch=0 时 H≈5，即先学 1s 短期积分；
      - 中间随训练进度平滑增长；
      - 默认 70% 训练进度到达 H=T，后 30% 固定完整 5s 训练。
    """
    min_steps = min(total_steps, 5)
    if total_steps <= min_steps:
        return float(total_steps)

    warmup_epochs = max((epochs - 1) * full_horizon_ratio, 1.0)
    progress = min(max(epoch / warmup_epochs, 0.0), 1.0)
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    return float(min_steps + (total_steps - min_steps) * smooth_progress)


def get_rollout_horizon(
    epoch: int,
    epochs: int,
    total_steps: int,
    full_horizon_ratio: float = 0.7,
):
    """Integer horizon used only for logging/metrics labels."""
    horizon = round(
        get_rollout_horizon_float(
            epoch,
            epochs,
            total_steps,
            full_horizon_ratio=full_horizon_ratio,
        )
    )
    min_steps = min(total_steps, 5)
    return int(min(max(horizon, min_steps), total_steps))


def get_timestep_loss_weights(
    epoch: int,
    epochs: int,
    total_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    full_horizon_ratio: float = 0.7,
    future_min_weight: float = 0.05,
):
    """
    Smooth per-timestep curriculum weights for trajectory losses.

    旧做法是 x[:, :H] hard slicing，H 以外 timestep 完全没有梯度；
    这里让所有 25 步从一开始都参与训练，只是远期 timestep 权重较小。
    随着 horizon_float 增大，远期权重连续上升，避免某个 epoch
    第一次“看到”新 step 时产生明显 loss 台阶。

    Returns:
        weights: [T], first steps near 1, far future >= future_min_weight.
    """
    horizon = get_rollout_horizon_float(
        epoch,
        epochs,
        total_steps,
        full_horizon_ratio=full_horizon_ratio,
    )
    step_ids = torch.arange(1, total_steps + 1, device=device, dtype=dtype)
    horizon_t = torch.tensor(horizon, device=device, dtype=dtype)

    # Linear soft boundary: steps before horizon have weight 1; the next step
    # ramps in continuously; farther future keeps a small non-zero weight.
    visible = torch.clamp(horizon_t - step_ids + 1.0, min=0.0, max=1.0)
    return future_min_weight + (1.0 - future_min_weight) * visible


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    """MSE over [B, T, ...] with per-timestep weights [T]."""
    err = (pred - target).pow(2)
    view_shape = [1, weights.shape[0]] + [1] * (err.dim() - 2)
    w = weights.view(*view_shape)
    denom = weights.sum() * err.shape[0]
    for size in err.shape[2:]:
        denom = denom * size
    return (err * w).sum() / denom.clamp_min(1e-8)


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
    full_horizon_ratio: float = 0.7,
    future_loss_min_weight: float = 0.05,
    final_traj_only: bool = False,
    late_soft_global_focus: bool = False,
    late_xy_loss_weight: float = 2.0,
    late_yaw_loss_weight: float = 3.0,
    late_v_loss_weight: float = 0.1,
    late_control_loss_weight: float = 0.05,
    late_smooth_loss_weight: float = 2e-3,
    late_endpoint_xy_weight: float = 2.0,
    late_tail_xy_weight: float = 1.0,
    late_tail_start_ratio: float = 0.6,
):
    """
    Train the acc/yaw_rate rollout RVQ tokenizer.

    Args:
        data_array: [N, T, 3] physical-space dxdydyaw.
    """
    if final_traj_only and late_soft_global_focus:
        raise ValueError("--final-traj-only and --late-soft-global-focus are mutually exclusive.")
    if not 0.0 <= late_tail_start_ratio < 1.0:
        raise ValueError("late_tail_start_ratio should be in [0, 1).")

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
    tail_start_idx = min(max(int(np.ceil(late_tail_start_ratio * num_steps)), 0), num_steps - 1)
    # Curriculum settings:
    # 1) full_horizon_ratio=0.7: 前 70% epoch 平滑扩展到完整 25 steps，
    #    后 30% 固定完整 5s 训练。
    # 2) future_loss_min_weight>0: 远期 timestep 从一开始就有少量梯度，
    #    避免 hard slicing 带来的“新 step 第一次被看到”台阶。
    for epoch in range(epochs):
        model.train()
        # Progressive horizon 现在只作为“主监督区域”的日志指标；
        # 实际 trajectory loss 使用全 T 步 per-timestep weights。
        horizon = get_rollout_horizon(
            epoch,
            epochs,
            num_steps,
            full_horizon_ratio=full_horizon_ratio,
        )
        horizon_float = get_rollout_horizon_float(
            epoch,
            epochs,
            num_steps,
            full_horizon_ratio=full_horizon_ratio,
        )
        smooth_loss_weight = 1e-3 if epoch > 30 else 0.0
        # Optional late-stage fine-tuning: after the curriculum has reached the
        # full horizon, optimize trajectory reconstruction directly and stop
        # applying the auxiliary v/acc/yaw_rate/smooth losses.
        late_stage_active = epoch >= int(epochs * full_horizon_ratio)
        final_traj_only_active = final_traj_only and late_stage_active
        late_soft_global_focus_active = late_soft_global_focus and late_stage_active
        effective_xy_loss_weight = late_xy_loss_weight if late_soft_global_focus_active else xy_loss_weight
        effective_yaw_loss_weight = late_yaw_loss_weight if late_soft_global_focus_active else yaw_loss_weight
        effective_v_loss_weight = late_v_loss_weight if late_soft_global_focus_active else v_loss_weight
        effective_control_loss_weight = (
            late_control_loss_weight if late_soft_global_focus_active else control_loss_weight
        )
        effective_smooth_loss_weight = (
            late_smooth_loss_weight if late_soft_global_focus_active else smooth_loss_weight
        )

        total_recon = 0.0
        total_xy = 0.0
        total_yaw = 0.0
        total_v = 0.0
        total_control = 0.0
        total_smooth = 0.0
        total_endpoint_xy = 0.0
        total_tail_xy = 0.0
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

                step_weights = get_timestep_loss_weights(
                    epoch,
                    epochs,
                    num_steps,
                    device=x_norm.device,
                    dtype=x_norm.dtype,
                    full_horizon_ratio=full_horizon_ratio,
                    future_min_weight=future_loss_min_weight,
                )

                # Trajectory losses use all T steps with smooth timestep weights.
                # Early training still focuses on the first ~1s, but future steps
                # are never completely invisible.
                mse_dxdy = weighted_mse_loss(out["x_recon"][..., :2], x_norm[..., :2], step_weights)
                mse_dyaw = weighted_mse_loss(out["x_recon"][..., 2], x_norm[..., 2], step_weights)
                recon_loss = mse_dxdy + 14.0 * mse_dyaw

                xy_loss = (
                    weighted_mse_loss(out["x_global"], gt["gt_x"], step_weights)
                    + weighted_mse_loss(out["y_global"], gt["gt_y"], step_weights)
                )
                yaw_loss = weighted_mse_loss(out["yaw"], gt["gt_yaw"], step_weights)
                endpoint_xy_loss = (
                    F.mse_loss(out["x_global"][:, -1], gt["gt_x"][:, -1])
                    + F.mse_loss(out["y_global"][:, -1], gt["gt_y"][:, -1])
                )
                tail_xy_loss = (
                    F.mse_loss(out["x_global"][:, tail_start_idx:], gt["gt_x"][:, tail_start_idx:])
                    + F.mse_loss(out["y_global"][:, tail_start_idx:], gt["gt_y"][:, tail_start_idx:])
                )

                # Dynamics/control losses are supervised on the full horizon from
                # the start. This teaches future acc/yaw_rate/v profiles before
                # the trajectory loss weight becomes large there.
                v_loss = F.mse_loss(out["v"], gt["gt_v"])
                control_loss = (
                    F.mse_loss(out["acc"], gt["gt_acc"])
                    + yaw_rate_weight
                    * F.mse_loss(out["yaw_rate"], gt["gt_yaw_rate"])
                )

                if num_steps > 1:
                    jerk = (out["acc"][:, 1:] - out["acc"][:, :-1]) / dt
                    yaw_acc = (out["yaw_rate"][:, 1:] - out["yaw_rate"][:, :-1]) / dt
                    smooth_loss = jerk.pow(2).mean() + yaw_acc.pow(2).mean()
                else:
                    smooth_loss = out["acc"].sum() * 0.0

                vq_loss = out["vq_loss"]
                traj_loss = (
                    recon_loss_weight * recon_loss
                    + effective_xy_loss_weight * xy_loss
                    + effective_yaw_loss_weight * yaw_loss
                )
                aux_loss = (
                    effective_v_loss_weight * v_loss
                    + effective_control_loss_weight * control_loss
                    + effective_smooth_loss_weight * smooth_loss
                )
                late_global_loss = (
                    late_endpoint_xy_weight * endpoint_xy_loss
                    + late_tail_xy_weight * tail_xy_loss
                ) if late_soft_global_focus_active else endpoint_xy_loss * 0.0
                if final_traj_only_active:
                    loss = traj_loss + vq_loss_weight * vq_loss
                else:
                    loss = traj_loss + aux_loss + late_global_loss + vq_loss_weight * vq_loss

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
            total_endpoint_xy += endpoint_xy_loss.item()
            total_tail_xy += tail_xy_loss.item()
            total_vq += vq_loss.item()
            total_loss += loss.item()

            with torch.no_grad():
                # Metrics are reported on the full 5s horizon so they stay
                # comparable across epochs while the curriculum weights change.
                step_dist = torch.sqrt(
                    (out["x_global"] - gt["gt_x"]).pow(2)
                    + (out["y_global"] - gt["gt_y"]).pow(2)
                    + 1e-6
                )
                batch_size_actual = x_norm.shape[0]
                total_ade += step_dist.mean(dim=1).sum().item()
                total_fde += step_dist[:, -1].sum().item()
                total_vrr_count += (step_dist.max(dim=1)[0] < max_lateral).sum().item()
                total_samples += batch_size_actual

                total_acc_mean += out["acc"].mean().item()
                total_acc_abs += out["acc"].abs().mean().item()
                total_yaw_rate_abs += out["yaw_rate"].abs().mean().item()
                total_v_abs += out["v"].abs().mean().item()
                running_v_min = min(running_v_min, out["v"].min().item())
                running_v_max = max(running_v_max, out["v"].max().item())

        scheduler.step()

        num_batches = len(dataloader)
        avg_recon = total_recon / num_batches
        avg_xy = total_xy / num_batches
        avg_yaw = total_yaw / num_batches
        avg_v = total_v / num_batches
        avg_control = total_control / num_batches
        avg_smooth = total_smooth / num_batches
        avg_endpoint_xy = total_endpoint_xy / num_batches
        avg_tail_xy = total_tail_xy / num_batches
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
        writer.add_scalar("loss/endpoint_xy", avg_endpoint_xy, epoch + 1)
        writer.add_scalar("loss/tail_xy", avg_tail_xy, epoch + 1)
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
        writer.add_scalar("train/rollout_horizon_float", horizon_float, epoch + 1)
        writer.add_scalar("train/future_loss_min_weight", future_loss_min_weight, epoch + 1)
        writer.add_scalar("train/full_horizon_ratio", full_horizon_ratio, epoch + 1)
        writer.add_scalar("train/final_traj_only_active", 1.0 if final_traj_only_active else 0.0, epoch + 1)
        writer.add_scalar(
            "train/late_soft_global_focus_active",
            1.0 if late_soft_global_focus_active else 0.0,
            epoch + 1,
        )
        writer.add_scalar("train/effective_xy_loss_weight", effective_xy_loss_weight, epoch + 1)
        writer.add_scalar("train/effective_yaw_loss_weight", effective_yaw_loss_weight, epoch + 1)
        writer.add_scalar("train/effective_v_loss_weight", effective_v_loss_weight, epoch + 1)
        writer.add_scalar("train/effective_control_loss_weight", effective_control_loss_weight, epoch + 1)
        writer.add_scalar("train/effective_smooth_loss_weight", effective_smooth_loss_weight, epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"[BiRVQ] Epoch {epoch+1:03d} | H: {horizon:02d} ({horizon_float:.1f}) | "
                f"Recon: {avg_recon:.5f} | XY: {avg_xy:.5f} | "
                f"Yaw: {avg_yaw:.5f} | V: {avg_v:.5f} | "
                f"Ctrl: {avg_control:.5f} | Smooth: {avg_smooth:.5f} | "
                f"EndXY: {avg_endpoint_xy:.5f} | TailXY: {avg_tail_xy:.5f} | "
                f"VQ: {avg_vq:.5f} | TrajErr: {avg_ade:.4f} | "
                f"EndErr: {avg_fde:.4f} | VRR: {vrr:.4f} | "
                f"FinalTrajOnly: {int(final_traj_only_active)} | "
                f"LateSoft: {int(late_soft_global_focus_active)} | "
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
        "full_horizon_ratio": full_horizon_ratio,
        "future_loss_min_weight": future_loss_min_weight,
        "final_traj_only": final_traj_only,
        "late_soft_global_focus": late_soft_global_focus,
        "late_xy_loss_weight": late_xy_loss_weight,
        "late_yaw_loss_weight": late_yaw_loss_weight,
        "late_v_loss_weight": late_v_loss_weight,
        "late_control_loss_weight": late_control_loss_weight,
        "late_smooth_loss_weight": late_smooth_loss_weight,
        "late_endpoint_xy_weight": late_endpoint_xy_weight,
        "late_tail_xy_weight": late_tail_xy_weight,
        "late_tail_start_ratio": late_tail_start_ratio,
        "late_tail_start_idx": tail_start_idx,
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
    parser.add_argument("--full-horizon-ratio", type=float, default=0.7)
    parser.add_argument("--future-loss-min-weight", type=float, default=0.05)
    parser.add_argument(
        "--final-traj-only",
        action="store_true",
        help=(
            "After full_horizon_ratio of training, optimize only recon/xy/yaw "
            "trajectory losses plus VQ loss; v/control/smooth remain logged but "
            "are not backpropagated."
        ),
    )
    parser.add_argument(
        "--late-soft-global-focus",
        action="store_true",
        help=(
            "After full_horizon_ratio of training, increase global trajectory "
            "and endpoint/tail losses while keeping smaller dynamics losses."
        ),
    )
    parser.add_argument("--late-xy-loss-weight", type=float, default=2.0)
    parser.add_argument("--late-yaw-loss-weight", type=float, default=3.0)
    parser.add_argument("--late-v-loss-weight", type=float, default=0.1)
    parser.add_argument("--late-control-loss-weight", type=float, default=0.05)
    parser.add_argument("--late-smooth-loss-weight", type=float, default=2e-3)
    parser.add_argument("--late-endpoint-xy-weight", type=float, default=2.0)
    parser.add_argument("--late-tail-xy-weight", type=float, default=1.0)
    parser.add_argument("--late-tail-start-ratio", type=float, default=0.6)
    args = parser.parse_args()
    if args.final_traj_only and args.late_soft_global_focus:
        parser.error("--final-traj-only and --late-soft-global-focus are mutually exclusive.")

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
        f"yaw_rate_max={args.yaw_rate_max} | full_horizon_ratio={args.full_horizon_ratio} | "
        f"future_loss_min_weight={args.future_loss_min_weight} | "
        f"final_traj_only={args.final_traj_only} | "
        f"late_soft_global_focus={args.late_soft_global_focus} | "
        f"late_endpoint_xy_weight={args.late_endpoint_xy_weight} | "
        f"late_tail_xy_weight={args.late_tail_xy_weight} | "
        f"late_tail_start_ratio={args.late_tail_start_ratio} | save_dir={args.save_dir}"
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
        full_horizon_ratio=args.full_horizon_ratio,
        future_loss_min_weight=args.future_loss_min_weight,
        final_traj_only=args.final_traj_only,
        late_soft_global_focus=args.late_soft_global_focus,
        late_xy_loss_weight=args.late_xy_loss_weight,
        late_yaw_loss_weight=args.late_yaw_loss_weight,
        late_v_loss_weight=args.late_v_loss_weight,
        late_control_loss_weight=args.late_control_loss_weight,
        late_smooth_loss_weight=args.late_smooth_loss_weight,
        late_endpoint_xy_weight=args.late_endpoint_xy_weight,
        late_tail_xy_weight=args.late_tail_xy_weight,
        late_tail_start_ratio=args.late_tail_start_ratio,
    )
