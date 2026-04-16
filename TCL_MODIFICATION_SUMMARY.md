# TCL (Temporal Consistency Loss) 集成总结

## 修改概述

参考 ActionCodec 论文提出的微小时间扰动导致Token剧烈跳变的问题，在原始的运动学 RVQ Transformer 中加入了 **TCL (Temporal Consistency Loss)** 约束。

## 核心修改

### 1. 添加 `compute_tcl_loss` 函数

```python
def compute_tcl_loss(codes1: torch.Tensor, codes2: torch.Tensor):
    """
    TCL (Temporal Consistency Loss)：约束微小扰动的codes保持相似。
    
    采用欧式距离度量codes相似度，使得扰动前后的codes非常接近。
    """
    codes1_flat = codes1.view(codes1.shape[0], -1).float()
    codes2_flat = codes2.view(codes2.shape[0], -1).float()
    
    min_batch = min(codes1_flat.shape[0], codes2_flat.shape[0])
    codes1_flat = codes1_flat[:min_batch]
    codes2_flat = codes2_flat[:min_batch]
    
    # MSE distance: 目标是让扰动前后的codes保持一致
    tcl_loss = F.mse_loss(codes1_flat, codes2_flat)
    return tcl_loss
```

### 2. 训练循环中的 TCL Loss 集成

在 `train_rvq_taae` 函数的训练循环中：

```python
# Loss 3: TCL (Temporal Consistency Loss)
tcl_weight = 0.1 if epoch > 30 else 0.0  # 从第30 epoch开始加入TCL

if tcl_weight > 0:
    noise_scale = 0.02  # 微小高斯噪声
    x_noisy = x + torch.randn_like(x) * noise_scale
    
    # 前向传播扰动后的轨迹
    _, _, codes_noisy, _, _ = model(x_noisy)
    
    # 计算TCL loss
    tcl_loss = compute_tcl_loss(codes, codes_noisy)
else:
    tcl_loss = torch.tensor(0.0, device=device)

# 总损失
loss = (
    recon_loss_weight * recon_loss
    + vq_loss_weight * vq_loss
    + kin_smooth_weight * kin_smooth_loss
    + tcl_loss_weight * tcl_loss
)
```

## 损失函数构成

| 损失类型 | 权重 | 启用时机 | 作用 |
|---------|------|---------|------|
| **Reconstruction Loss** | 5.0 | 始终 | 重建轨迹准确性 |
| **VQ Loss** | 0.5 | 始终 | 向量量化损失 |
| **Kinematic Smoothness Loss** | 0.1 | epoch > 30 | 物理约束（速度/曲率平滑） |
| **TCL Loss** | 0.1 | epoch > 30 | 时间一致性约束（新增） |

## TCL Loss 的物理意义

### 问题陈述
原始代码中微小的输入扰动（高斯噪声）可能导致量化的codes发生剧烈跳变，这种**Token不稳定性**在生成任务中会造成严重问题。

### 解决方案
通过 TCL Loss 约束：
1. 对输入轨迹加入微小扰动 $\tilde{x} = x + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, 0.02^2)$
2. 前向传播扰动后的轨迹，得到codes: $codes' = \text{RVQ}(\text{Encoder}(\tilde{x}))$
3. 约束原始codes与扰动codes接近：$\mathcal{L}_{TCL} = ||codes - codes'||^2$

### 预期效果
- ✅ 提高Token的时间稳定性
- ✅ 减少微小输入变化导致的编码剧烈波动
- ✅ 改进后续生成任务的稳定性和连贯性

## 运行验证

### 方式1：直接运行完整训练
```bash
cd /home/an.huang3/VQ-VAE/rvq_transformer_vehdyn
python train_tfm.py
```

输出示例（epoch > 30时会显示TCL loss）：
```
[KinRVQ] Epoch 040 | Recon: 0.03784 | VQ: 0.00235 | KinSmooth: 0.02870 | TCL: 0.04521 | v_mean: 9.11 m/s | v_max: 21.75 | κ_abs: 0.0259
```

## 代码最小化程度

- ✅ 只添加了一个新函数 `compute_tcl_loss` (~20 行)
- ✅ 训练循环中只增加了 TCL 损失计算逻辑 (~10 行)
- ✅ 没有修改模型架构
- ✅ 没有修改 RVQ 或 Encoder/Decoder 结构
- ✅ 保持原有损失函数的权重结构

## 超参数设置

可根据需要调整以下参数：

```python
# TCL 权重启用时机
tcl_weight = 0.1 if epoch > 30 else 0.0

# 扰动噪声幅度
noise_scale = 0.02  # 标准差

# TCL 在总损失中的权重
tcl_loss_weight = 0.1
```

## 后续改进方向

1. 尝试不同的扰动策略（时间偏移、缩放等）
2. 动态调整TCL权重
3. 使用相似度度量替代MSE（余弦相似度、Wasserstein距离）
4. 在多个噪声尺度上应用TCL (多尺度一致性)

