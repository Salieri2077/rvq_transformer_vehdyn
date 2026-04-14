import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EMAVectorQuantizer(nn.Module):
    """
    带有 EMA 更新和死码重置逻辑的矢量量化器。
    相比梯度下降版，EMA 版更稳定，能显著提升 bin 利用 reversed 率。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.decay = decay
        self.epsilon = epsilon

        # 1. 码本：不再通过梯度更新，设为不可求导
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        
        # 2. EMA 统计量：用于追踪每个 bin 的使用频率和特征累加
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embedding.clone())

    def forward(self, inputs):
        # inputs: [B*T, D]
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 3. 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))
            
        # 4. 寻找最近的 code
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 5. 量化
        quantized = F.embedding(encoding_indices, self.embedding)
        
        # 6. EMA 更新逻辑 (仅在训练模式下进行统计)
        if self.training:
            # 计算当前 batch 中每个 bin 的使用量
            current_cluster_size = torch.sum(encodings, dim=0)
            # 计算当前 batch 中分配给每个 bin 的特征总和
            dw = torch.matmul(encodings.t(), flat_input)
            
            # 更新 EMA 统计量
            self.ema_cluster_size.data.mul_(self.decay).add_(current_cluster_size, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # 使用拉普拉斯平滑修正 cluster size
            n = torch.sum(self.ema_cluster_size)
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # 更新码本权重：W = EMA_W / Smoothed_Size
            new_embeddings = self.ema_w / smoothed_cluster_size.unsqueeze(1)
            self.embedding.data.copy_(new_embeddings)

            # --- 附加：死码重置逻辑 ---
            # 如果某个 bin 的 smoothed_cluster_size 太小，说明是死码，用当前 batch 的随机样本替换它
            if torch.min(smoothed_cluster_size) < 1e-2:
                self._revive_dead_codes(flat_input, smoothed_cluster_size)

        # 7. Loss: 仅保留 Commitment Loss (因为码本不通梯度)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # STE 直通估计
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

    def _revive_dead_codes(self, inputs, cluster_size):
        """将不常用的码本重新初始化为输入中的随机向量"""
        dead_indices = (cluster_size < 1e-2).nonzero(as_tuple=True)[0]
        if len(dead_indices) > 0:
            # 随机采样输入作为新的码本中心
            indices = torch.randperm(inputs.shape[0])[:len(dead_indices)]
            random_samples = inputs[indices]
            # 确保写回到 embedding / ema_w 时 dtype 一致（避免 BF16/FP16 混合精度下报错）
            random_samples = random_samples.to(self.embedding.dtype)
            self.embedding.data[dead_indices] = random_samples
            self.ema_w.data[dead_indices] = random_samples
            self.ema_cluster_size.data[dead_indices] = 1.0

    def get_codebook_entry(self, indices):
        return F.embedding(indices, self.embedding)

class ResidualVQ(nn.Module):
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, dropout=0.0, commitment_cost=0.25):
        super().__init__()
        self.layers = nn.ModuleList([
            EMAVectorQuantizer(num_embeddings, embedding_dim, commitment_cost=commitment_cost)
            for _ in range(num_quantizers)
        ])
        self.dropout = dropout

    def _get_dropout_start_idx(self, n_layers):
        """
        根据层段策略决定开始丢弃的索引
        
        Args:
            n_layers: 总层数
            
        Returns:
            dropout_start_idx: 从该索引开始的所有层都会被丢弃
        """
        # 训练时分层 Dropout 机制：根据层段策略决定开始丢弃的索引
        if self.training and self.dropout > 0:
            # 随机选择一个层段策略
            strategy_rand = np.random.random()
            
            if strategy_rand < 0.33:  # 底层策略（1-3层）：几乎不丢弃
                # 90%概率不丢弃（从n_layers开始，即不丢弃），10%概率从第8层开始丢弃
                if np.random.random() < 0.9:
                    dropout_start_idx = n_layers  # 不丢弃任何层
                else:
                    dropout_start_idx = 7  # 从第8层（索引7）开始丢弃
            elif strategy_rand < 0.67:  # 中层策略（4-7层）：适度随机丢弃
                # 50%概率从第7层开始丢弃，50%概率从第5层开始丢弃
                if np.random.random() < 0.5:
                    dropout_start_idx = 6  # 从第7层（索引6）开始丢弃
                else:
                    dropout_start_idx = 4  # 从第5层（索引4）开始丢弃
            else:  # 高层策略（8-10层）：高比例丢弃
                # 70%概率从第6层开始丢弃，30%概率从第4层开始丢弃
                if np.random.random() < 0.7:
                    dropout_start_idx = 5  # 从第6层（索引5）开始丢弃
                else:
                    dropout_start_idx = 3  # 从第4层（索引3）开始丢弃
        else:
            # 推理时或 dropout=0 时，不丢弃任何层
            dropout_start_idx = n_layers
        
        return dropout_start_idx

    def forward(self, x):
        """
        Args:
            x: [B, D] 或 [B, T, D] - 输入特征
        Returns:
            quantized_out: 量化后的输出，维度与输入相同
            all_losses: VQ 损失（标量）
            codes: [B, num_layers] 或 [B, T, num_layers]
        """
        # 检测输入维度
        input_dim = x.dim()
        original_shape = x.shape
        
        # 如果是 3D 输入 [B, T, D]，reshape 成 [B*T, D]
        if input_dim == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
            need_reshape = True
        else:
            need_reshape = False
        
        quantized_out = torch.zeros_like(x)
        residual = x
        all_losses = 0.0
        all_indices = []
        
        n_layers = len(self.layers)
        
        # 根据层段策略决定开始丢弃的索引
        dropout_start_idx = self._get_dropout_start_idx(n_layers)

        # 从 dropout_start_idx 开始的所有层都被丢弃
        for i, layer in enumerate(self.layers):
            if i < dropout_start_idx:
                x_q, loss, indices = layer(residual)
                quantized_out = quantized_out + x_q
                residual = residual - x_q
                all_losses += loss
                all_indices.append(indices)
            else:
                # 被 dropout 的层不参与计算
                break
                
        if len(all_indices) > 0:
            codes = torch.stack(all_indices, dim=1)  # [B*T, num_layers] 或 [B, num_layers]
            # 如果是 3D 输入，reshape codes 回 [B, T, num_layers]
            if need_reshape:
                codes = codes.view(B, T, -1)
        else:
            codes = None
        
        # 如果是 3D 输入，reshape quantized_out 回 [B, T, D]
        if need_reshape:
            quantized_out = quantized_out.view(original_shape)
            
        return quantized_out, all_losses, codes

    def decode_from_indices(self, indices):
        """
        Args:
            indices: [B, num_layers] 或 [B, T, num_layers] - 量化索引
        Returns:
            quantized_out: [B, embedding_dim] 或 [B, T, embedding_dim] - 解码后的特征
        """
        # 检测输入维度
        input_dim = indices.dim()
        original_shape = indices.shape
        
        # 如果是 3D 输入 [B, T, num_layers]，reshape 成 [B*T, num_layers]
        if input_dim == 3:
            B, T, num_layers = indices.shape
            indices = indices.view(B * T, num_layers)
            need_reshape = True
            embedding_dim = self.layers[0].embedding_dim
        else:
            need_reshape = False
            embedding_dim = self.layers[0].embedding_dim
        
        quantized_out = 0.0
        for i, layer in enumerate(self.layers):
            # 兼容推理时可能只输入前k个token的情况
            if i < indices.shape[1]:
                idx = indices[:, i]
                quantized_out += layer.get_codebook_entry(idx)
        
        # 如果是 3D 输入，reshape 回 [B, T, embedding_dim]
        if need_reshape:
            quantized_out = quantized_out.view(B, T, embedding_dim)
        
        return quantized_out
    
    def decode_from_codes(self, codes):
        """
        从 codes 解码，与 decode_from_indices 功能相同（别名）
        Args:
            codes: [B, num_layers] 或 [B, T, num_layers] - 量化索引
        Returns:
            quantized_out: [B, embedding_dim] 或 [B, T, embedding_dim] - 解码后的特征
        """
        return self.decode_from_indices(codes)
