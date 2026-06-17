import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def parse_rvq_scales(rvq_scales):
    """Parse comma-separated RVQ scales into a tuple of positive ints."""
    if rvq_scales is None:
        return None
    if isinstance(rvq_scales, str):
        rvq_scales = rvq_scales.strip()
        if not rvq_scales:
            return None
        values = [int(x.strip()) for x in rvq_scales.split(",") if x.strip()]
    else:
        values = [int(x) for x in rvq_scales]

    if any(v < 1 for v in values):
        raise ValueError(f"rvq_scales should contain positive integers, got {values}")
    return tuple(values)


def _validate_rvq_scales(scales, num_quantizers):
    if scales is None:
        return None
    if len(scales) != num_quantizers:
        raise ValueError(
            f"rvq_scales length ({len(scales)}) should match num_quantizers ({num_quantizers})"
        )
    for prev, cur in zip(scales, scales[1:]):
        if cur < prev:
            raise ValueError(f"rvq_scales should be non-decreasing, got {list(scales)}")
    return tuple(scales)


def make_default_multiscale_scales(max_tokens, num_scales=3):
    """Create increasing temporal scales whose total token count is bounded."""
    max_tokens = max(int(max_tokens), int(num_scales))
    num_scales = max(1, int(num_scales))
    if num_scales == 1:
        return (max_tokens,)

    # Default 3-scale split uses 20% / 33% / remainder. For a 15-token
    # budget this gives [3, 5, 7], keeping total tokens equal to 15.
    if num_scales == 3:
        first = max(1, int(round(max_tokens * 0.20)))
        second = max(first, int(round(max_tokens * 0.33)))
        third = max(second, max_tokens - first - second)
        scales = [first, second, third]
    else:
        weights = np.arange(1, num_scales + 1, dtype=np.float32)
        raw = weights / weights.sum() * float(max_tokens)
        scales = [max(1, int(round(v))) for v in raw]
        delta = max_tokens - sum(scales)
        scales[-1] += delta

    for i in range(1, len(scales)):
        scales[i] = max(scales[i], scales[i - 1])

    while sum(scales) > max_tokens and scales[-1] > scales[-2]:
        scales[-1] -= 1
    return tuple(int(v) for v in scales)


def _resize_temporal(x, target_steps):
    """Resize [B, T, D] along T with linear interpolation."""
    target_steps = int(target_steps)
    if target_steps < 1:
        raise ValueError(f"target_steps should be >= 1, got {target_steps}")
    if x.shape[1] == target_steps:
        return x
    x_t = x.transpose(1, 2)
    original_dtype = x_t.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        x_t = x_t.float()
    resized = F.interpolate(x_t, size=target_steps, mode="linear", align_corners=False)
    resized = resized.to(original_dtype)
    return resized.transpose(1, 2)


def normalize_rvq_type(rvq_type):
    return (rvq_type or "residual").strip().lower().replace("-", "_")


def is_multi_scale_rvq(rvq_type):
    return normalize_rvq_type(rvq_type) in {
        "multi_scale",
        "multiscale",
        "mutil_scale",
        "mutilscale",
    }


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
            # 当 dead code 数量超过 batch 样本数时，允许有放回采样，避免 shape mismatch
            n_dead = len(dead_indices)
            n_inputs = int(inputs.shape[0])
            if n_dead <= n_inputs:
                indices = torch.randperm(n_inputs, device=inputs.device)[:n_dead]
            else:
                indices = torch.randint(0, n_inputs, (n_dead,), device=inputs.device)
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

    def forward_with_distances(self, x):
        """
        与 forward 使用相同的 residual quantization 顺序，但额外返回每层距离矩阵。

        Args:
            x: [B, D] 或 [B, T, D] - 输入特征
        Returns:
            quantized_out: 量化后的输出，维度与输入相同
            all_losses: VQ 损失（标量；这里只复用 commitment 形式，不更新 EMA）
            codes: [B, num_layers] 或 [B, T, num_layers]
            all_dists: list，每层为 [B, vocab_size] 或 [B, T, vocab_size]
        """
        input_dim = x.dim()
        original_shape = x.shape

        if input_dim == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
            need_reshape = True
        else:
            need_reshape = False

        quantized_out = torch.zeros_like(x)
        residual = x
        all_losses = x.new_zeros(())
        all_indices = []
        all_dists = []

        n_layers = len(self.layers)
        dropout_start_idx = self._get_dropout_start_idx(n_layers)

        for i, layer in enumerate(self.layers):
            if i >= dropout_start_idx:
                break

            flat_input = residual.view(-1, layer.embedding_dim)
            codebook = layer.embedding.to(flat_input.dtype)
            distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(codebook ** 2, dim=1)
                - 2 * torch.matmul(flat_input, codebook.t())
            )
            indices = torch.argmin(distances, dim=1)
            quantized = F.embedding(indices, codebook)

            e_latent_loss = F.mse_loss(quantized.detach(), residual)
            loss = layer.commitment_cost * e_latent_loss
            x_q = residual + (quantized - residual).detach()

            quantized_out = quantized_out + x_q
            residual = residual - x_q
            all_losses = all_losses + loss
            all_indices.append(indices)
            all_dists.append(distances)

        if len(all_indices) > 0:
            codes = torch.stack(all_indices, dim=1)
            if need_reshape:
                codes = codes.view(B, T, -1)
                all_dists = [dist.view(B, T, -1) for dist in all_dists]
        else:
            codes = None

        if need_reshape:
            quantized_out = quantized_out.view(original_shape)

        return quantized_out, all_losses, codes, all_dists

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


class MultiScaleResidualVQ(nn.Module):
    """
    SnapMoGen/MoMask++ 风格的 multi-scale residual VQ。

    核心差异：
      - 所有 residual quantization 层共享同一个 EMA codebook。
      - 对 [B, T, D] latent，每一层先把 residual 插值到指定 temporal scale h_v，
        量化后再插值回 full scale T，逐层相加并更新 residual。
      - 对当前 train_tfm/train_tfm_bicycle 的 [B, D] trajectory latent，T 自动视为 1，
        因此保持与现有 RVQ 相同的 forward/decode 表面。
    """

    def __init__(
        self,
        num_quantizers,
        num_embeddings,
        embedding_dim,
        dropout=0.0,
        commitment_cost=0.25,
        scales=None,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()
        self.num_quantizers = int(num_quantizers)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.scales = _validate_rvq_scales(parse_rvq_scales(scales), self.num_quantizers)
        self.dropout = float(dropout)

        self.quantizer = EMAVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
        )
        # Convenience surface for code that inspects model.rvq.layers. The
        # repeated references are intentional: this RVQ has one shared codebook.
        self.layers = [self.quantizer for _ in range(self.num_quantizers)]

    def _resolve_scales(self, full_steps):
        full_steps = int(full_steps)
        if full_steps < 1:
            raise ValueError(f"full_steps should be >= 1, got {full_steps}")

        if self.scales is not None:
            return [min(int(s), full_steps) for s in self.scales]

        if full_steps == 1:
            return [1 for _ in range(self.num_quantizers)]

        scales = []
        for i in range(self.num_quantizers):
            power = self.num_quantizers - 1 - i
            scale = max(1, int(round(full_steps / (2 ** power))))
            if scales:
                scale = max(scale, scales[-1])
            scales.append(min(scale, full_steps))
        scales[-1] = full_steps
        return scales

    def _get_dropout_start_idx(self, n_layers):
        # Match ResidualVQ's layer-segment dropout policy so switching rvq_type
        # changes the quantizer design, not the training schedule.
        if self.training and self.dropout > 0:
            strategy_rand = np.random.random()

            if strategy_rand < 0.33:
                if np.random.random() < 0.9:
                    return n_layers
                return min(7, n_layers)
            if strategy_rand < 0.67:
                if np.random.random() < 0.5:
                    return min(6, n_layers)
                return min(4, n_layers)
            if np.random.random() < 0.7:
                return min(5, n_layers)
            return min(3, n_layers)

        return n_layers

    def _quantize_without_ema(self, residual_down):
        flat_input = residual_down.reshape(-1, self.embedding_dim)
        codebook = self.quantizer.embedding.to(flat_input.dtype)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(codebook ** 2, dim=1)
            - 2 * torch.matmul(flat_input, codebook.t())
        )
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, codebook)
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        loss = self.quantizer.commitment_cost * e_latent_loss
        quantized = flat_input + (quantized - flat_input).detach()
        return quantized, loss, indices, distances

    def forward(self, x):
        """
        Args:
            x: [B, D] 或 [B, T, D]
        Returns:
            quantized_out: 与 x 同形状
            all_losses: VQ commitment loss 标量
            codes: [B, num_layers] for [B,D], or [B, sum(h_v)] for [B,T,D]
        """
        input_dim = x.dim()
        if input_dim == 2:
            x_seq = x.unsqueeze(1)
            squeeze_time = True
        elif input_dim == 3:
            x_seq = x
            squeeze_time = False
        else:
            raise ValueError(f"MultiScaleResidualVQ expects [B,D] or [B,T,D], got {tuple(x.shape)}")

        batch_size, full_steps, embedding_dim = x_seq.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embedding_dim}")

        scales = self._resolve_scales(full_steps)
        dropout_start_idx = self._get_dropout_start_idx(self.num_quantizers)

        quantized_out = torch.zeros_like(x_seq)
        residual = x_seq
        all_losses = x_seq.new_zeros(())
        all_indices = []

        for i in range(dropout_start_idx):
            scale = scales[i]
            residual_down = _resize_temporal(residual, scale)
            flat_down = residual_down.reshape(batch_size * scale, embedding_dim)
            quantized_flat, loss, indices = self.quantizer(flat_down)
            quantized_down = quantized_flat.view(batch_size, scale, embedding_dim)
            quantized_full = _resize_temporal(quantized_down, full_steps)

            quantized_out = quantized_out + quantized_full
            residual = residual - quantized_full
            all_losses = all_losses + loss
            all_indices.append(indices.view(batch_size, scale))

        if not all_indices:
            codes = None
        elif squeeze_time:
            codes = torch.cat(all_indices, dim=1)
        else:
            codes = torch.cat(all_indices, dim=1)

        if squeeze_time:
            quantized_out = quantized_out.squeeze(1)

        return quantized_out, all_losses, codes

    def forward_with_distances(self, x):
        input_dim = x.dim()
        if input_dim == 2:
            x_seq = x.unsqueeze(1)
            squeeze_time = True
        elif input_dim == 3:
            x_seq = x
            squeeze_time = False
        else:
            raise ValueError(f"MultiScaleResidualVQ expects [B,D] or [B,T,D], got {tuple(x.shape)}")

        batch_size, full_steps, embedding_dim = x_seq.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embedding_dim}")

        scales = self._resolve_scales(full_steps)
        dropout_start_idx = self._get_dropout_start_idx(self.num_quantizers)
        quantized_out = torch.zeros_like(x_seq)
        residual = x_seq
        all_losses = x_seq.new_zeros(())
        all_indices = []
        all_dists = []

        for i in range(dropout_start_idx):
            scale = scales[i]
            residual_down = _resize_temporal(residual, scale)
            quantized_flat, loss, indices, distances = self._quantize_without_ema(residual_down)
            quantized_down = quantized_flat.view(batch_size, scale, embedding_dim)
            quantized_full = _resize_temporal(quantized_down, full_steps)

            quantized_out = quantized_out + quantized_full
            residual = residual - quantized_full
            all_losses = all_losses + loss
            all_indices.append(indices.view(batch_size, scale))
            all_dists.append(distances.view(batch_size, scale, self.num_embeddings))

        codes = torch.cat(all_indices, dim=1) if all_indices else None
        if squeeze_time:
            quantized_out = quantized_out.squeeze(1)
            all_dists = [d.squeeze(1) for d in all_dists]

        return quantized_out, all_losses, codes, all_dists

    def _split_codes_for_decode(self, indices, output_length):
        if isinstance(indices, (list, tuple)):
            return [idx.long() for idx in indices]

        if indices.dim() == 3:
            # Compatibility path for [B, T, L] full-scale codes.
            return [indices[:, :, i].long() for i in range(indices.shape[-1])]

        if indices.dim() != 2:
            raise ValueError(f"indices should be [B,L], [B,T,L], or a list, got {tuple(indices.shape)}")

        total_tokens = indices.shape[1]
        if output_length is None:
            if total_tokens > self.num_quantizers:
                raise ValueError(
                    "output_length is required when decoding concatenated multi-scale codes"
                )
            return [indices[:, i : i + 1].long() for i in range(total_tokens)]

        scales = self._resolve_scales(output_length)
        split_codes = []
        offset = 0
        for scale in scales:
            if offset >= total_tokens:
                break
            next_offset = min(offset + scale, total_tokens)
            split_codes.append(indices[:, offset:next_offset].long())
            offset = next_offset
        return split_codes

    def decode_from_indices(self, indices, output_length=None):
        """
        Args:
            indices: [B,L] concatenated scale tokens, [B,T,L] full-scale tokens,
                or a list of [B,h_v] tensors.
            output_length: required for concatenated temporal multi-scale tokens.
        Returns:
            [B,D] when output_length is None, otherwise [B, output_length, D].
        """
        split_codes = self._split_codes_for_decode(indices, output_length)
        if not split_codes:
            raise ValueError("No codes provided for decoding")

        batch_size = split_codes[0].shape[0]
        full_steps = int(output_length) if output_length is not None else 1
        quantized_out = self.quantizer.embedding.new_zeros(
            batch_size,
            full_steps,
            self.embedding_dim,
        )

        for idx in split_codes:
            if idx.dim() != 2:
                raise ValueError(f"Each scale code tensor should be [B,h], got {tuple(idx.shape)}")
            if idx.shape[0] != batch_size:
                raise ValueError("All scale code tensors should have the same batch size")
            scale = idx.shape[1]
            quantized_down = self.quantizer.get_codebook_entry(idx.reshape(-1))
            quantized_down = quantized_down.view(batch_size, scale, self.embedding_dim)
            quantized_out = quantized_out + _resize_temporal(quantized_down, full_steps)

        if output_length is None:
            return quantized_out.squeeze(1)
        return quantized_out

    def decode_from_codes(self, codes, output_length=None):
        return self.decode_from_indices(codes, output_length=output_length)


# Keep the misspelled name requested by the user as an alias.
MutilScaleResidualVQ = MultiScaleResidualVQ


def build_residual_vq(
    rvq_type,
    num_quantizers,
    num_embeddings,
    embedding_dim,
    dropout=0.0,
    commitment_cost=0.25,
    rvq_scales=None,
):
    rvq_type = normalize_rvq_type(rvq_type)
    if rvq_type in {"residual", "rvq"}:
        return ResidualVQ(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            dropout=dropout,
            commitment_cost=commitment_cost,
        )
    if rvq_type in {"multi_scale", "multiscale", "mutil_scale", "mutilscale"}:
        return MultiScaleResidualVQ(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            scales=rvq_scales,
            dropout=dropout,
            commitment_cost=commitment_cost,
        )
    raise ValueError(f"Unknown rvq_type={rvq_type!r}; expected 'residual' or 'multi_scale'")
