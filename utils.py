import numpy as np
import pickle
import os

from scipy.fft import dct, idct

import torch
import torch.nn.functional as F


def load_sampled_datas():
    # data_path = '/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway_0122/sample_trajectorys_by_scenario.npy'
    data_path = '/share-global/zhe.du/planner/planNN2/tokenizer/0124_json/sample_trajectorys_by_scenario_update0210.npy'
    data = np.load(data_path, allow_pickle=True).item()
    datas = data['trajs']
    return datas


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