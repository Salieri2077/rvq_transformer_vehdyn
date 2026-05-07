import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 导入你原有的模型和工具函数
from rvq_model import ResidualVQ
from utils import load_sampled_datas,load_test_datas

from train_tfm import TrajRVQTransformer


def _integrate_to_global_np(traj: np.ndarray) -> np.ndarray:
    dx = traj[:, 0]
    dy = traj[:, 1]
    dyaw = traj[:, 2]
    yaw = np.cumsum(dyaw, axis=0)
    prev_yaw = np.zeros_like(yaw)
    if len(yaw) > 1:
        prev_yaw[1:] = yaw[:-1]

    dx_global = dx * np.cos(prev_yaw) - dy * np.sin(prev_yaw)
    dy_global = dx * np.sin(prev_yaw) + dy * np.cos(prev_yaw)
    x_global = np.cumsum(dx_global, axis=0)
    y_global = np.cumsum(dy_global, axis=0)
    return np.stack([x_global, y_global], axis=-1)


def _plot_noise_recon_cases(cases, save_path: str):
    if len(cases) == 0:
        return []

    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(save_path)
    stem, ext = os.path.splitext(base)
    if ext == "":
        ext = ".png"

    saved_paths = []
    for i, case in enumerate(cases):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
        gt_xy = _integrate_to_global_np(case["gt_phys"])
        clean_xy = _integrate_to_global_np(case["recon_clean_phys"])
        noisy_xy = _integrate_to_global_np(case["recon_noisy_phys"])

        ax.plot(gt_xy[:, 0], gt_xy[:, 1], label="GT", linewidth=2.2, color="#1f77b4")
        ax.plot(clean_xy[:, 0], clean_xy[:, 1], label="Recon(clean token)", linewidth=2.0, linestyle="--", color="#2ca02c")
        ax.plot(noisy_xy[:, 0], noisy_xy[:, 1], label="Recon(noisy token)", linewidth=2.0, linestyle="-.", color="#d62728")

        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], c="black", s=20)
        # 细长轨迹（x 远大于 y）若强制 equal，会把图压扁到几乎不可读。
        # x_span = float(np.max(gt_xy[:, 0]) - np.min(gt_xy[:, 0]) + 1e-6)
        # y_span = float(np.max(gt_xy[:, 1]) - np.min(gt_xy[:, 1]) + 1e-6)
        # slender_ratio = x_span / y_span
        # if slender_ratio <= 8.0:
        #     ax.set_aspect("equal", adjustable="box")
        # else:
        #     ax.set_aspect("auto")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        clean_tokens = ",".join(str(int(v)) for v in case["tokens_clean"])
        noisy_tokens = ",".join(str(int(v)) for v in case["tokens_noisy"])
        ax.set_title(
            f'sample#{case["sample_idx"]} | same_layers={case["same_layers"]}/{case["num_layers"]}\n'
            f"clean tokens: [{clean_tokens}]\n"
            f"noisy tokens: [{noisy_tokens}]",
            fontsize=9,
        )

        fig.suptitle("Tokenizer Noise Robustness: clean/noisy token recon comparison", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        sample_id = int(case["sample_idx"])
        per_path = os.path.join(save_dir, f"{stem}_sample{sample_id:06d}_{i:02d}{ext}")
        fig.savefig(per_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(per_path)

    return saved_paths


def evaluate_tokenizer_health(
    model: TrajRVQTransformer,
    dataloader: DataLoader,
    device: torch.device,
    noise_std_xy: float = 0.01,  # 给物理空间的 dx, dy 加入 1cm 的扰动
    noise_std_yaw: float = 0.01,  # 给物理空间的 dyaw 加入 0.01 弧度的扰动
    clip_limit: torch.Tensor = None,
    vis_num_cases: int = 6,
    vis_save_path: str = None,
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
    token_counts_per_layer = torch.zeros(num_layers, vocab_size, dtype=torch.long)
    
    # 记录 Overlap Rate 的统计变量
    total_samples = 0
    overlap_count_per_layer = torch.zeros(num_layers).to(device)
    exact_match_all_layers = 0  # 记录所有层 Token 都完全一致的极端苛刻情况
    vis_cases = []

    print(f"开始评估 Tokenizer 健康度... (注入物理噪声: XY ±{noise_std_xy}m, Yaw ±{noise_std_yaw}rad)")
    if vis_save_path is None:
        vis_save_path = os.path.join(os.getcwd(), "tokenizer_health_noise_recon.png")

    with torch.no_grad():
        for batch in dataloader:
            x_norm = batch[0].to(device)
            B = x_norm.shape[0]
            batch_start_idx = total_samples
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
                layer_counts = torch.bincount(codes_clean[:, i].cpu(), minlength=vocab_size)
                token_counts_per_layer[i] += layer_counts

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
            # 保持和训练时一致：z-score -> clip -> scale。
            x_z_noisy = (x_phys_noisy - model.norm_mean) / (model.norm_std + 1e-8)
            if clip_limit is not None:
                x_z_noisy = torch.clamp(x_z_noisy, -clip_limit, clip_limit)
            x_norm_noisy = x_z_noisy / model.norm_scale

            # ==========================================
            # 3. 噪声推理：获取加噪后的 Token
            # ==========================================
            z_noisy = model.encode(x_norm_noisy)
            _, _, codes_noisy = model.rvq(z_noisy) # codes_noisy: [B, num_layers]

            # 可视化样本（尽量少改动主逻辑）：展示 clean/noisy token 对应重建轨迹差异
            need = max(0, int(vis_num_cases) - len(vis_cases))
            if need > 0:
                take = min(need, B)
                x_recon_clean_norm = model.decode_from_codes(codes_clean[:take])
                x_recon_noisy_norm = model.decode_from_codes(codes_noisy[:take])
                x_recon_clean_phys = (x_recon_clean_norm * model.norm_scale * model.norm_std) + model.norm_mean
                x_recon_noisy_phys = (x_recon_noisy_norm * model.norm_scale * model.norm_std) + model.norm_mean

                same_layers = (codes_clean[:take] == codes_noisy[:take]).sum(dim=1).cpu().numpy()
                for j in range(take):
                    vis_cases.append(
                        {
                            "sample_idx": int(batch_start_idx + j),
                            "gt_phys": x_phys[j].cpu().numpy(),
                            "recon_clean_phys": x_recon_clean_phys[j].cpu().numpy(),
                            "recon_noisy_phys": x_recon_noisy_phys[j].cpu().numpy(),
                            "tokens_clean": codes_clean[j].cpu().numpy(),
                            "tokens_noisy": codes_noisy[j].cpu().numpy(),
                            "same_layers": int(same_layers[j]),
                            "num_layers": int(num_layers),
                        }
                    )

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

    # 1.5 输出字典使用分布，避免“用过很多 code，但少数 code 占比极高”的情况被掩盖
    print("\n[1.5] 字典使用分布 (Code Usage Distribution):")
    for i in range(num_layers):
        counts = token_counts_per_layer[i].float()
        probs = counts / counts.sum().clamp_min(1.0)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        top1_rate = sorted_probs[0].item() * 100
        top5_rate = sorted_probs[:5].sum().item() * 100
        entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
        perplexity = torch.exp(entropy).item()

        if i < 4 or i == num_layers - 1:
            top_ids = sorted_ids[:5].tolist()
            top_rates = [round(v * 100, 2) for v in sorted_probs[:5].tolist()]
            print(
                f"  - 第 {i+1:02d} 层: top1={top1_rate:5.2f}% | "
                f"top5={top5_rate:5.2f}% | perplexity={perplexity:7.2f} | "
                f"top_ids={top_ids} | top_rates={top_rates}%"
            )
        elif i == 4:
            print("  - ...")

        if top1_rate > 50:
            print(f"    警告: 第 {i+1:02d} 层单个 token 占比超过 50%，可能存在使用分布过度集中。")

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

    # 画在一个画布上：clean/noisy 重建 + 各自 token 激活值
    saved_paths = _plot_noise_recon_cases(vis_cases, vis_save_path)
    if len(saved_paths) > 0:
        print(f"已保存可视化对比图(每样本一张): {os.path.dirname(saved_paths[0])}")
        print(f"图像数量: {len(saved_paths)}，文件前缀: {os.path.basename(vis_save_path)}")
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
    clip_limit = torch.tensor(norm_params['clip_limit'], dtype=torch.float32).to(device) if 'clip_limit' in norm_params else None
    
    # 数据归一化和训练时保持一致：z-score -> clip -> scale。
    test_norm = (torch.tensor(test_trajs, dtype=torch.float32).to(device) - mean) / (std + 1e-8)
    if clip_limit is not None:
        test_norm = torch.clamp(test_norm, -clip_limit, clip_limit)
    test_norm = test_norm / scale
    
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
    
    evaluate_tokenizer_health(model, dataloader, device, clip_limit=clip_limit)
