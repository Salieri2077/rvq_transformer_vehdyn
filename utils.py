import numpy as np
import pickle
import os
import json
import csv

from scipy.fft import dct, idct

import torch
import torch.nn.functional as F

DEFAULT_BASE_DATA_PATH = "/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas.npy"
DEFAULT_AUGMENTED_DATA_PATH = (
    "/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy"
)


def to_py(obj):
    """递归转成 Python 原生类型，确保 json.dump 不报错。"""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return to_py(obj.detach().cpu().numpy())
    return obj


def percentiles(arr: np.ndarray) -> dict:
    """返回常用百分位统计（p0/p25/p50/p75/p95/p99/max）。"""
    if arr.size == 0:
        return {k: 0.0 for k in ["p0", "p25", "p50", "p75", "p95", "p99", "max"]}
    q = np.percentile(arr, [0, 25, 50, 75, 95, 99])
    return {
        "p0": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(np.max(arr)),
    }


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_py(data), f, indent=2)


def write_csv(rows: list, path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_traj_array(data_path: str) -> np.ndarray:
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if isinstance(data, dict) and "trajs" in data:
        return np.asarray(data["trajs"])
    return np.asarray(data)


def load_traj_array(data_path: str, dtype=np.float32) -> np.ndarray:
    """
    通用轨迹加载接口，兼容:
      1) npy 直接是 [N, T, 3]
      2) npy 为 dict 且包含 'trajs'
    """
    arr = _load_traj_array(data_path)
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected trajectory array shape [N, T, 3], got {arr.shape}")
    return arr


def resolve_default_data_path() -> str:
    """优先使用增强数据，若不存在则回退到基础数据。"""
    if os.path.exists(DEFAULT_AUGMENTED_DATA_PATH):
        return DEFAULT_AUGMENTED_DATA_PATH
    return DEFAULT_BASE_DATA_PATH


def load_sampled_datas(data_path: str = None):
    if data_path is None:
        data_path = resolve_default_data_path()
    return load_traj_array(data_path)


def load_test_datas(data_path: str = None):
    if data_path is None:
        data_path = resolve_default_data_path()
    return load_traj_array(data_path)


def load_norm_params_torch(norm_path: str, device: torch.device):
    """
    加载归一化参数并转为 torch.Tensor。
    返回字段:
      mean, std, scale_factor, clip_limit(None 或 Tensor)
    """
    with open(norm_path, "rb") as f:
        norm_params = pickle.load(f)
    out = {
        "mean": torch.tensor(norm_params["mean"], dtype=torch.float32, device=device),
        "std": torch.tensor(norm_params["std"], dtype=torch.float32, device=device),
        "scale_factor": torch.tensor(norm_params["scale_factor"], dtype=torch.float32, device=device),
        "clip_limit": None,
    }
    if "clip_limit" in norm_params:
        out["clip_limit"] = torch.tensor(norm_params["clip_limit"], dtype=torch.float32, device=device)
    return out


def normalize_trajs_torch(
    trajs: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
    clip_limit: torch.Tensor = None,
) -> torch.Tensor:
    x = torch.tensor(trajs, dtype=torch.float32, device=mean.device)
    x_norm = (x - mean) / (std + 1e-8)
    if clip_limit is not None:
        x_norm = torch.clamp(x_norm, -clip_limit, clip_limit)
    return x_norm / (scale_factor + 1e-8)


def denormalize_trajs_torch(
    x_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_factor: torch.Tensor,
) -> torch.Tensor:
    return x_norm * scale_factor * (std + 1e-8) + mean


def token_sequence_to_str(tokens) -> str:
    """把一条轨迹的 RVQ token 序列保存成紧凑可读的 CSV 字段。"""
    values = np.asarray(tokens).reshape(-1)
    return "[" + ",".join(str(int(v)) for v in values.tolist()) + "]"


def finite_diff_keep_shape(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape[-1] <= 1:
        return np.zeros_like(x)
    d = np.diff(x, axis=-1) / (dt + 1e-8)
    return np.concatenate([d[..., :1], d], axis=-1)


def moving_average_1d_time(x: np.ndarray, window: int) -> np.ndarray:
    """
    对最后一维做简单滑动平均，保持形状不变。
    x: [..., T]
    """
    if window <= 1:
        return x
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = np.convolve(flat[i], kernel, mode="same")
    return out.reshape(x.shape)


def compute_extended_kinematic_quantities(
    profile: dict,
    dt: float,
    curvature_clip: float = 2.0,
    jerk_smooth_window: int = 3,
):
    """
    从 compute_kinematic_profiles 的输出计算扩展物理量。
    统一在多个 eval 里复用。
    """
    eps = 1e-8
    vx = profile["vx"]
    vy = profile["vy"]
    ax = profile["ax"]
    ay = profile["ay"]
    speed = profile["speed"]
    yaw_rate = profile["yaw_rate"]
    disp_xy = profile["disp_xy"]

    curvature_raw = (vx * ay - vy * ax) / (np.power(speed, 3) + eps)
    curvature_raw = np.where(speed < 0.2, 0.0, curvature_raw)
    if curvature_clip is not None and curvature_clip > 0:
        curvature_raw = np.clip(curvature_raw, -curvature_clip, curvature_clip)

    lat_acc = (speed ** 2) * curvature_raw
    abs_lat_acc = np.abs(lat_acc)

    ax_for_jerk = moving_average_1d_time(ax, window=jerk_smooth_window)
    ay_for_jerk = moving_average_1d_time(ay, window=jerk_smooth_window)
    jerk_x = finite_diff_keep_shape(ax_for_jerk, dt=dt)
    jerk_y = finite_diff_keep_shape(ay_for_jerk, dt=dt)
    jerk_abs = np.sqrt(jerk_x**2 + jerk_y**2 + 1e-6)

    curvature_rate = finite_diff_keep_shape(curvature_raw, dt=dt)
    abs_curvature_rate = np.abs(curvature_rate)

    kinematic_error = np.abs(yaw_rate - speed * curvature_raw)

    disp_norm = np.sqrt(np.sum(disp_xy ** 2, axis=-1) + eps)
    unit_disp = disp_xy / disp_norm[..., None]
    dot = np.sum(unit_disp[:, 1:] * unit_disp[:, :-1], axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)
    turn_angle_jump = np.zeros_like(speed)
    turn_angle_jump[:, 1:] = angle

    dot_full = np.ones_like(speed)
    dot_full[:, 1:] = dot
    speed_prev = np.concatenate([speed[:, :1], speed[:, :-1]], axis=1)

    return {
        "curvature_raw": curvature_raw,
        "lat_acc": lat_acc,
        "abs_lat_acc": abs_lat_acc,
        "jerk_x": jerk_x,
        "jerk_y": jerk_y,
        "jerk_abs": jerk_abs,
        "curvature_rate": curvature_rate,
        "abs_curvature_rate": abs_curvature_rate,
        "kinematic_error": kinematic_error,
        "turn_angle_jump": turn_angle_jump,
        "turn_dot": dot_full,
        "speed_prev": speed_prev,
    }


def longest_true_run(mask_1d: np.ndarray) -> int:
    """返回布尔序列中 True 的最长连续长度。"""
    best = 0
    cur = 0
    for v in mask_1d.astype(bool):
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def count_longitudinal_sign_changes(local_vx: np.ndarray, speed_th: float = 0.2) -> np.ndarray:
    """统计纵向速度符号变化次数，忽略低速噪声段。"""
    n = local_vx.shape[0]
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        signs = np.sign(local_vx[i])
        valid = np.abs(local_vx[i]) >= speed_th
        signs = signs[valid]
        if signs.shape[0] <= 1:
            continue
        out[i] = int(np.sum(signs[1:] != signs[:-1]))
    return out


def compute_kinematic_profiles(trajs: np.ndarray, dt: float):
    """
    由 body-frame dxdydyaw 计算完整运动学 profile。
    """
    eps = 1e-8
    clips = np.asarray(trajs, dtype=np.float32)
    n, t, _ = clips.shape

    dx = clips[:, :, 0]
    dy = clips[:, :, 1]
    dyaw = clips[:, :, 2]

    yaw = np.cumsum(dyaw, axis=1)
    prev_yaw = np.zeros_like(yaw)
    if t > 1:
        prev_yaw[:, 1:] = yaw[:, :-1]

    cos_y = np.cos(prev_yaw)
    sin_y = np.sin(prev_yaw)
    dx_global = dx * cos_y - dy * sin_y
    dy_global = dx * sin_y + dy * cos_y

    disp_xy = np.stack([dx_global, dy_global], axis=-1)
    xy = np.cumsum(disp_xy, axis=1)

    # 改为与 detect-2.py 一致的中心差分思路：
    # 先对全局位置 xy 做一阶 gradient 得速度，再对速度做 gradient 得加速度。
    # 相比 diff，更不容易在边界和高频抖动处产生尖峰。
    if t > 1:
        vel = np.gradient(xy, axis=1) / (dt + eps)  # [N, T, 2]
        vx = vel[..., 0]
        vy = vel[..., 1]
        acc_vec = np.gradient(vel, axis=1) / (dt + eps)  # [N, T, 2]
        ax = acc_vec[..., 0]
        ay = acc_vec[..., 1]
    else:
        vx = np.zeros((n, t), dtype=clips.dtype)
        vy = np.zeros((n, t), dtype=clips.dtype)
        ax = np.zeros_like(vx)
        ay = np.zeros_like(vy)
    speed = np.sqrt(vx**2 + vy**2 + 1e-6)
    acc = np.sqrt(ax**2 + ay**2 + 1e-6)

    yaw_rate = dyaw / (dt + eps)
    if t > 1:
        yaw_acc = np.gradient(yaw_rate, axis=1) / (dt + eps)
    else:
        yaw_acc = np.zeros_like(yaw_rate)

    curvature = (vx * ay - vy * ax) / (np.power(speed, 3) + eps)
    curvature = np.where(speed < 0.2, 0.0, curvature)
    curvature = np.clip(curvature, -2.0, 2.0)

    local_vx = dx / (dt + eps)
    local_vy = dy / (dt + eps)

    return {
        "xy": xy,
        "disp_xy": disp_xy,
        "yaw": yaw,
        "prev_yaw": prev_yaw,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "ax": ax,
        "ay": ay,
        "acc": acc,
        "yaw_rate": yaw_rate,
        "yaw_acc": yaw_acc,
        "curvature": curvature,
        "local_vx": local_vx,
        "local_vy": local_vy,
    }


def compute_reconstruction_case_metrics(gt_trajs: np.ndarray, pred_trajs: np.ndarray, dt: float) -> dict:
    """
    逐条计算重建质量指标。

    recon_mse 在原始 dxdydyaw 空间计算；ADE/FDE/max_error 先积分到全局 XY，
    与 eval_tokenizer_by_scenario.py 中 worst case 排序使用的轨迹误差口径一致。
    """
    gt_trajs = np.asarray(gt_trajs, dtype=np.float32)
    pred_trajs = np.asarray(pred_trajs, dtype=np.float32)
    if gt_trajs.shape != pred_trajs.shape:
        raise ValueError(f"gt/pred shape mismatch: {gt_trajs.shape} vs {pred_trajs.shape}")
    if gt_trajs.ndim != 3 or gt_trajs.shape[-1] != 3:
        raise ValueError(f"Expected [N, T, 3] trajectories, got {gt_trajs.shape}")

    gt_prof = compute_kinematic_profiles(gt_trajs, dt=dt)
    pred_prof = compute_kinematic_profiles(pred_trajs, dt=dt)

    step_dist_error = np.sqrt(np.sum((pred_prof["xy"] - gt_prof["xy"]) ** 2, axis=-1) + 1e-6)
    recon_mse = np.mean((pred_trajs - gt_trajs) ** 2, axis=(1, 2))

    return {
        "recon_mse": recon_mse.astype(np.float64),
        "ade": step_dist_error.mean(axis=1).astype(np.float64),
        "fde": step_dist_error[:, -1].astype(np.float64),
        "max_error": step_dist_error.max(axis=1).astype(np.float64),
    }


def compute_motion_features(all_dxdydyaw_clips: np.ndarray, fps: float = 5.0, time_duration=None):
    clips = np.asarray(all_dxdydyaw_clips, dtype=np.float32)
    n, t, _ = clips.shape
    duration = time_duration if time_duration is not None else (t / fps)
    duration = max(float(duration), 1e-6)
    dt = max(duration / max(t, 1), 1e-6)

    profiles = compute_kinematic_profiles(clips, dt=dt)

    dyaws = clips[:, :, 2]
    cumulative_yaws = np.cumsum(dyaws, axis=1)
    net_yaw = cumulative_yaws[:, -1] if t > 0 else np.zeros(n, dtype=np.float32)
    gross_yaw = np.sum(np.abs(dyaws), axis=1)

    step_d = np.sqrt(np.sum(profiles["disp_xy"] ** 2, axis=-1) + 1e-6)
    total_dist = np.sum(step_d, axis=1)
    avg_speed = total_dist / duration

    s = np.sign(dyaws)
    s[s == 0] = 1
    sign_changes = np.sum(s[:, 1:] != s[:, :-1], axis=1) if t > 1 else np.zeros(n, dtype=np.int32)

    speed = profiles["speed"]
    acc = profiles["acc"]
    curvature = profiles["curvature"]
    local_vx = profiles["local_vx"]

    reverse_mask = local_vx < -0.2
    reverse_ratio = reverse_mask.mean(axis=1)
    reverse_steps = reverse_mask.sum(axis=1)
    reverse_dist = np.sum(np.abs(local_vx) * dt * reverse_mask, axis=1)
    min_local_vx = np.min(local_vx, axis=1)

    stop_mask = speed < 0.3
    stop_ratio = stop_mask.mean(axis=1)
    stop_steps = stop_mask.sum(axis=1)

    long_vel_sign_changes = count_longitudinal_sign_changes(local_vx, speed_th=0.2)

    has_reverse = (reverse_steps >= 2) | (reverse_dist > 0.5)
    has_stop_or_near_stop = (stop_steps >= 1) | (np.min(speed, axis=1) < 0.3)

    return {
        "net_yaw": net_yaw,
        "gross_yaw": gross_yaw,
        "total_dist": total_dist,
        "avg_speed": avg_speed,
        "sign_changes": sign_changes,
        "max_speed": np.max(speed, axis=1),
        "min_speed": np.min(speed, axis=1),
        "speed_std": np.std(speed, axis=1),
        "mean_abs_acc": np.mean(np.abs(acc), axis=1),
        "max_abs_acc": np.max(np.abs(acc), axis=1),
        "avg_abs_curvature": np.mean(np.abs(curvature), axis=1),
        "max_abs_curvature": np.max(np.abs(curvature), axis=1),
        "reverse_ratio": reverse_ratio,
        "reverse_steps": reverse_steps,
        "reverse_dist": reverse_dist,
        "min_local_vx": min_local_vx,
        "stop_ratio": stop_ratio,
        "stop_steps": stop_steps,
        "long_vel_sign_changes": long_vel_sign_changes,
        "has_reverse": has_reverse,
        "has_stop_or_near_stop": has_stop_or_near_stop,
    }


def build_scenario_masks(all_trajs: np.ndarray, fps: float = 5.0):
    feat = compute_motion_features(all_trajs, fps=fps)

    net_yaw = feat["net_yaw"]
    gross_yaw = feat["gross_yaw"]
    total_dist = feat["total_dist"]
    avg_speed = feat["avg_speed"]
    sign_changes = feat["sign_changes"]
    reverse_ratio = feat["reverse_ratio"]
    reverse_dist = feat["reverse_dist"]
    min_local_vx = feat["min_local_vx"]
    stop_steps = feat["stop_steps"]
    stop_ratio = feat["stop_ratio"]
    long_vel_sign_changes = feat["long_vel_sign_changes"]
    has_reverse = feat["has_reverse"]

    th_dist_static = 1.0
    reverse_dist_th = 0.5
    reverse_ratio_th = 0.08
    th_straight_net = 0.10
    th_straight_gross = 0.20
    v_10 = 10.0 / 3.6
    v_80 = 80.0 / 3.6
    v_120 = 120.0 / 3.6
    th_turn = 0.35
    th_uturn = 2.35
    direct_uturn_min_speed = 2.0

    mask_static = total_dist < th_dist_static
    mask_reverse_base = (
        (~mask_static)
        & ((reverse_dist > reverse_dist_th) | (reverse_ratio > reverse_ratio_th) | (min_local_vx < -0.5))
    )
    mask_three_point_like = (
        (~mask_static)
        & has_reverse
        & ((long_vel_sign_changes >= 1) | (stop_steps >= 1) | (stop_ratio > 0.05))
        & (gross_yaw > 0.6)
    )
    mask_reverse = mask_reverse_base | mask_three_point_like

    mask_direct_uturn = (
        (~mask_static)
        & (~mask_reverse)
        & (np.abs(net_yaw) >= th_uturn)
        & (avg_speed >= direct_uturn_min_speed)
    )

    remaining = (~mask_static) & (~mask_reverse) & (~mask_direct_uturn)
    mask_detour = remaining & (np.abs(net_yaw) < 0.20) & (gross_yaw >= 0.80) & (sign_changes >= 2)

    remaining = remaining & (~mask_detour)
    mask_left = remaining & (net_yaw >= th_turn) & (np.abs(net_yaw) < th_uturn)
    mask_right = remaining & (net_yaw <= -th_turn) & (np.abs(net_yaw) < th_uturn)

    remaining = remaining & (~mask_left) & (~mask_right)
    mask_straight = remaining & (np.abs(net_yaw) < th_straight_net) & (gross_yaw < th_straight_gross)
    mask_low_straight = mask_straight & (avg_speed >= (v_10 - 2 / 3.6)) & (avg_speed <= (v_10 + 2 / 3.6))
    mask_high_straight = mask_straight & (avg_speed >= (v_80 - 10 / 3.6)) & (avg_speed <= (v_80 + 10 / 3.6))
    mask_high_straight_120 = mask_straight & (avg_speed >= (v_120 - 15 / 3.6)) & (avg_speed <= (v_120 + 15 / 3.6))

    categories = {
        "Stationary": mask_static,
        "Reverse": mask_reverse,
        "DirectUTurn": mask_direct_uturn,
        "Detour": mask_detour,
        "LeftTurn": mask_left,
        "RightTurn": mask_right,
        "LowSpeedStraight_10kmh": mask_low_straight,
        "HighSpeedStraight_80kmh": mask_high_straight,
        "HighSpeedStraight_120kmh": mask_high_straight_120,
    }
    return categories, feat


def integrate_to_global(trajs: np.ndarray) -> np.ndarray:
    dx = trajs[:, :, 0]
    dy = trajs[:, :, 1]
    dyaw = trajs[:, :, 2]
    yaw = np.cumsum(dyaw, axis=1)
    prev_yaw = np.zeros_like(yaw)
    prev_yaw[:, 1:] = yaw[:, :-1]

    dx_global = dx * np.cos(prev_yaw) - dy * np.sin(prev_yaw)
    dy_global = dx * np.sin(prev_yaw) + dy * np.cos(prev_yaw)

    x_global = np.cumsum(dx_global, axis=1)
    y_global = np.cumsum(dy_global, axis=1)
    return np.stack([x_global, y_global], axis=-1)


def compute_speed_and_acc(trajs: np.ndarray, dt: float):
    speed = np.sqrt(trajs[:, :, 0] ** 2 + trajs[:, :, 1] ** 2) / dt
    acc = np.diff(speed, axis=1) / dt
    return speed, acc

def preprocess_and_save_norm_params(data_array, save_dir, data_type):
    """
    改进版：使用分位数进行 Robust Scaling，防止离群点压缩有效数据范围。
    """
    print("Pre-processing data (Robust Scaling)...")
    num_steps = data_array.shape[1]
    
    # 1. 计算 Mean/Std (Z-Score)
    mean = np.mean(data_array, axis=(0, 1), keepdims=True)
    std = np.std(data_array, axis=(0, 1), keepdims=True)
    
    # 防止除以 0
    data_z = (data_array - mean) / (std + 1e-8)

    # 2. 关键修改：使用 99.9% 分位数代替 Max
    # 计算绝对值的 99.9 分位数作为边界
    # 这意味着我们将忽略 0.1% 的极端大值，优先保证主体数据的分辨率
    quantile_limit = np.percentile(np.abs(data_z), 99.99, axis=(0, 1), keepdims=True)
    
    # 避免某个维度全是 0 导致 limit 为 0
    quantile_limit = np.maximum(quantile_limit, 1e-6)

    # 3. 截断数据 (Clip)
    # 凡是超过 99.9% 分位数的数据，强制拉回到边界，防止它们撑爆归一化范围
    data_z_clipped = np.clip(data_z, -quantile_limit, quantile_limit)

    # 4. 计算 Scale Factor
    # 现在的边界就是 quantile_limit，我们稍作缩放映射到 [-1, 1]
    scale_factor = quantile_limit * 1.01  # 留 1% 余量

    data_normalized = data_z_clipped / scale_factor

    # 打印统计信息，检查是否撑满了 [-0.99, 0.99]
    print(f"Data Stats (Normalized):")
    for i, name in enumerate(['dx', 'dy', 'dyaw']):
        if i < data_normalized.shape[-1]:
            d_min = data_normalized[..., i].min()
            d_max = data_normalized[..., i].max()
            print(f"  {name} Range: {d_min:.4f} ~ {d_max:.4f}")
            # 如果这里的范围是 -0.99 ~ 0.99，说明归一化非常健康

    # 保存参数
    norm_params = {
        'mean': mean,
        'std': std,
        'scale_factor': scale_factor, # 注意保存的是基于分位数的 scale
        'num_steps': num_steps,
        'clip_limit': quantile_limit # 最好也记录一下截断阈值（推理时也要截断）
    }
    
    with open(os.path.join(save_dir, f"{data_type}_norm_params.pkl"), 'wb') as f:
        pickle.dump(norm_params, f)

    return data_normalized


def acceleration_smoothness_loss(pred_u, gt_u, dt=0.2):
    """
    对加速度做平滑性约束（约束加速度的变化率，即 jerk）
    
    思路：
        - dxdydyaw 可以看作速度（相对于上一帧的增量）
        - 加速度 = dxdydyaw 的一阶差分：acc = dxdydyaw[:, 1:] - dxdydyaw[:, :-1]
        - 加速度的平滑性（jerk）= 加速度的一阶差分：jerk = acc[:, 1:] - acc[:, :-1]
        - 约束 jerk 尽可能小，减少加速度曲线的抖动锯齿
    
    Args:
        pred_u: [B, T, 3] - 预测的 dxdydyaw
        gt_u:   [B, T, 3] - GT 的 dxdydyaw
        dt:     时间步长（秒），默认 0.2 秒（5Hz），用于可选的速度/加速度单位转换
    
    Returns:
        标量 loss：加速度平滑性损失（jerk 的 MSE）
    """
    B, T, C = pred_u.shape
    assert C == 3, "acceleration_smoothness_loss 期望输入为 [B, T, 3] 的 dxdydyaw"
    assert T >= 2, "acceleration_smoothness_loss 需要至少 2 个时间步才能计算加速度"
    
    # 计算加速度：acc = dxdydyaw 的一阶差分
    acc_pred = pred_u[:, 1:, :] - pred_u[:, :-1, :]  # [B, T-1, 3]
    acc_gt = gt_u[:, 1:, :] - gt_u[:, :-1, :]  # [B, T-1, 3]
    
    # 计算加速度的变化率（jerk）：jerk = acc 的一阶差分
    # 需要至少 2 个时间步的加速度才能计算 jerk
    if T < 3:
        # 如果时间步太少，直接对加速度做 MSE 约束
        return F.mse_loss(acc_pred, acc_gt)
    
    jerk_pred = acc_pred[:, 1:, :] - acc_pred[:, :-1, :]  # [B, T-2, 3]
    jerk_gt = acc_gt[:, 1:, :] - acc_gt[:, :-1, :]  # [B, T-2, 3]
    
    # 平滑性损失：约束 jerk（加速度的变化率），减少加速度曲线的抖动
    smoothness_loss = F.mse_loss(jerk_pred, jerk_gt)
    
    return smoothness_loss


def torch_dct_ii(x, n_coeffs: int = None):
    """
    对输入 [B, T, C] 在 T 维度进行 DCT-II 变换 (等价于 scipy.fft.dct, norm='ortho')，
    然后沿着 T 维度保留前 n_coeffs 个系数。

    Args:
        x: [B, T, C] tensor
        n_coeffs: 保留的时间维 DCT 系数个数；如果为 None，则保留全部 T 个系数。
    """
    orig_dtype = x.dtype
    B, T, C = x.shape

    # 统一提升到 float32，避免 torch.fft.rfft 的 dtype 限制
    x = x.to(torch.float32)

    # 1. 搬移到 C 维并做对称延展: [B, C, T] -> [B, C, 2T]
    x = x.transpose(1, 2)
    # 通过镜像延展实现 DCT 的边界条件
    x_padded = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    
    # 2. 运行 FFT（float32）
    fft_res = torch.fft.rfft(x_padded, dim=-1)
    
    # 3. 提取前 T 个系数并应用 DCT 的相位修正
    # 这里的数学推导较复杂，简言之：DCT 是 FFT 的实部加上特定的相位旋转
    n = torch.arange(T, device=x.device, dtype=torch.float32)
    phi = torch.exp(-1j * torch.pi * n / (2 * T))
    
    dct_out = fft_res[..., :T] * phi
    dct_out = dct_out.real
    
    # 归一化 (ortho)
    norm_factor0 = 1.0 / torch.sqrt(torch.tensor(4.0 * T, device=x.device, dtype=torch.float32))
    norm_factor = 1.0 / torch.sqrt(torch.tensor(2.0 * T, device=x.device, dtype=torch.float32))
    dct_out[..., 0] = dct_out[..., 0] * norm_factor0
    dct_out[..., 1:] = dct_out[..., 1:] * norm_factor
    
    dct_out = dct_out.transpose(1, 2)  # [B, T, C]

    # 沿着时间维截取前 n_coeffs 个系数（如果指定）
    if n_coeffs is not None:
        n_coeffs = min(n_coeffs, T)
        dct_out = dct_out[:, :n_coeffs, :]

    return dct_out.to(orig_dtype)

def frequency_smoothness_loss(pred_u, gt_u, keep_ratio=0.3, dt=0.2):
    """
    在 DCT 域内进行约束。
    keep_ratio: 保留多少比例的低频分量。
    """
    # 1. 转换到 DCT 域
    pred_dct = torch_dct_ii(pred_u) # [B, T, 3]
    gt_dct = torch_dct_ii(gt_u)     # [B, T, 3]
    # print(f"pred_dct:{pred_dct.shape}, gt_dct:{gt_dct.shape}")
    # print(f"pred_dct:{pred_dct}, gt_dct:{gt_dct}")
    
    T = pred_u.shape[1]
    cutoff = int(T * keep_ratio)
    
    # 2. 这里的思路是：低频对齐 GT，高频直接强制归零
    low_freq_loss = F.l1_loss(pred_dct[:, :cutoff, :], gt_dct[:, :cutoff, :])
    
    high_freq_loss = torch.mean(torch.abs(pred_dct[:, cutoff:, :]))

    # 增加“变动一致性”：约束预测的 DCT 系数与 GT 的 DCT 系数符号一致
    sign_loss = F.l1_loss(torch.sign(pred_dct[:, :cutoff, :]), torch.sign(gt_dct[:, :cutoff, :]))

    return low_freq_loss + 10.0 * high_freq_loss + 0.1 * sign_loss
