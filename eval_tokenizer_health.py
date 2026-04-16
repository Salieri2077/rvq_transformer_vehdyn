import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 导入你原有的模型和工具函数
from rvq_model import ResidualVQ
from utils import load_sampled_datas,load_test_datas

from train_tfm import TrajRVQTransformer

def evaluate_tokenizer_health(
    model: TrajRVQTransformer,
    dataloader: DataLoader,
    device: torch.device,
    noise_std_xy: float = 0.05,  # 给物理空间的 dx, dy 加入 5cm 的扰动
    noise_std_yaw: float = 0.0  # 给物理空间的 dyaw 加入 0.01 弧度的扰动
):
    """
    评估 Tokenizer 的健康度与拓扑稳定性。
    主要指标：
    1. Codebook Utilization (字典利用率)
    2. Overlap Rate (OR，重叠率 / 噪声抗性)
    """
    model.eval()
    
    num_layers = model.num_layers
    vocab_size = model.vocab_size
    
    # 记录每个 RVQ 层激活的 Token 集合
    active_tokens_per_layer = [set() for _ in range(num_layers)]
    
    # 记录 Overlap Rate 的统计变量
    total_samples = 0
    overlap_count_per_layer = torch.zeros(num_layers).to(device)
    exact_match_all_layers = 0  # 记录所有层 Token 都完全一致的极端苛刻情况

    print(f"开始评估 Tokenizer 健康度... (注入物理噪声: XY ±{noise_std_xy}m, Yaw ±{noise_std_yaw}rad)")

    with torch.no_grad():
        for batch in dataloader:
            x_norm = batch[0].to(device)
            B = x_norm.shape[0]
            total_samples += B
            
            # ==========================================
            # 1. 正常推理：获取原始 Token
            # ==========================================
            z = model.encode(x_norm)
            _, _, codes_clean = model.rvq(z)  # codes_clean: [B, num_layers]
            
            # 统计激活的 Token ID (更新 Set)
            codes_cpu = codes_clean.cpu().numpy()
            for i in range(num_layers):
                active_tokens_per_layer[i].update(codes_cpu[:, i].tolist())

            # ==========================================
            # 2. 注入物理微小噪声
            # 为什么要在物理空间加噪声？因为我们在模拟传感器抖动或微小控制误差
            # ==========================================
            # 归一化空间 -> 物理空间
            x_phys = (x_norm * model.norm_scale * model.norm_std) + model.norm_mean
            
            # 构造物理扰动
            noise = torch.randn_like(x_phys)
            noise[..., 0] *= noise_std_xy  # dx noise
            noise[..., 1] *= noise_std_xy  # dy noise
            noise[..., 2] *= noise_std_yaw # dyaw noise
            
            x_phys_noisy = x_phys + noise
            
            # 物理空间 -> 归一化空间
            x_norm_noisy = (x_phys_noisy - model.norm_mean) / (model.norm_std + 1e-8) / model.norm_scale

            # ==========================================
            # 3. 噪声推理：获取加噪后的 Token
            # ==========================================
            z_noisy = model.encode(x_norm_noisy)
            _, _, codes_noisy = model.rvq(z_noisy) # codes_noisy: [B, num_layers]

            # ==========================================
            # 4. 计算 Overlap Rate (重叠率)
            # ==========================================
            # 按层对比：相同的位置为 1，不同为 0
            is_same_token = (codes_clean == codes_noisy).float() # [B, num_layers]
            overlap_count_per_layer += is_same_token.sum(dim=0)
            
            # 苛刻指标：如果一辆车的所有层的 token 都一模一样，才算完美抗噪
            all_layers_same = (is_same_token.sum(dim=1) == num_layers).float() # [B]
            exact_match_all_layers += all_layers_same.sum().item()

    # ==========================================
    # 汇总输出指标
    # ==========================================
    print("\n" + "="*50)
    print("Tokenizer 健康度评估报告")
    print("="*50)
    
    # 1. 输出字典利用率 (Codebook Utilization)
    print("\n[1] 字典利用率 (Codebook Utilization):")
    avg_utilization = 0
    for i in range(num_layers):
        used_count = len(active_tokens_per_layer[i])
        utilization_rate = used_count / vocab_size * 100
        avg_utilization += utilization_rate
        # 只打印前 4 层和最后一层，避免刷屏
        if i < 4 or i == num_layers - 1:
            print(f"  - 第 {i+1:02d} 层: 激活 {used_count:4d} / {vocab_size} ({utilization_rate:5.2f}%)")
        elif i == 4:
            print("  - ...")
    
    print(f"平均字典利用率: {avg_utilization / num_layers:.2f}%")
    if avg_utilization / num_layers < 20:
        print("警告: 字典严重坍缩 (Dead Codes)！大模型将面临词汇量贫乏的问题。")

    # 2. 输出拓扑稳定性 (Overlap Rate)
    print("\n[2] 拓扑稳定性 / 噪声抗性 (Overlap Rate - OR):")
    overlap_rates = (overlap_count_per_layer / total_samples) * 100
    avg_or = overlap_rates.mean().item()
    for i in range(num_layers):
        if i < 4 or i == num_layers - 1:
            print(f"  - 第 {i+1:02d} 层 OR: {overlap_rates[i]:5.2f}%")
        elif i == 4:
            print("  - ...")
            
    print(f"平均 Overlap Rate: {avg_or:.2f}%")
    print(f"全层完美重合率: {exact_match_all_layers / total_samples * 100:.2f}%")
    if avg_or < 50:
        print("警告: Tokenizer 对物理噪音极其敏感！ downstream VLA将难以收敛。")
    print("="*50 + "\n")

    return avg_utilization / num_layers, avg_or


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 1. 配置路径 ===
    save_dir = "./work_dirs/tokenizer/rvq_tfm_kin_0311"
    data_type = "pred"
    model_path = os.path.join(save_dir, f"{data_type}_rvq_taae_model.pth")
    norm_path = os.path.join(save_dir, f"{data_type}_norm_params.pkl")
    
    # === 2. 加载测试数据 ===
    # 用 load_test_datas()，确保和训练集隔离
    print("加载数据中...")
    # sampled_trajs = load_test_datas()
    sampled_trajs = load_sampled_datas()

    if data_type == "history":
        sampled_trajs = sampled_trajs[:, :14, :]
    
    test_size = int(len(sampled_trajs))
    test_trajs = sampled_trajs
    print(f"测试集大小: {test_size} 条轨迹")
    
    # === 3. 加载归一化参数 ===
    with open(norm_path, 'rb') as f:
        norm_params = pickle.load(f)
    mean = torch.tensor(norm_params['mean'], dtype=torch.float32).to(device)
    std = torch.tensor(norm_params['std'], dtype=torch.float32).to(device)
    scale = torch.tensor(norm_params['scale_factor'], dtype=torch.float32).to(device)
    
    # 数据归一化 (一定要和训练时的一致！)
    test_norm = (torch.tensor(test_trajs, dtype=torch.float32).to(device) - mean) / (std + 1e-8) / scale
    
    dataset = TensorDataset(test_norm)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)

    # === 4. 初始化并加载模型权重 ===
    print(f"加载模型权重: {model_path}")
    model = TrajRVQTransformer(
        input_steps=test_trajs.shape[1],
        input_dim=test_trajs.shape[2],
        num_layers=15,   # 和你训练时保持一致
        vocab_size=1024,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
    ).to(device)
    
    model.set_norm_params(mean, std, scale)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    evaluate_tokenizer_health(model, dataloader, device)