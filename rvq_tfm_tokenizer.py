import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import torch.nn.functional as F

from src.utils.geometry import integrate_trajectory_keyframe


# ==========================================
# 1. 模型结构定义 (必须与训练时完全一致)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, D, T]
        return x + self.norm(self.relu(self.conv(x)))
class VectorQuantizer(nn.Module):
    """
    推理用的 VectorQuantizer，与训练时的 EMAVectorQuantizer 兼容。
    使用 register_buffer 存储 embedding，以匹配训练时保存的权重格式。
    同时注册 EMA 相关的 buffer（虽然推理时不用，但需要匹配权重格式）。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 使用 register_buffer 而不是 nn.Embedding，以匹配 EMAVectorQuantizer 的权重格式
        # 训练时 EMAVectorQuantizer 使用 register_buffer('embedding', ...)
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        
        # 注册 EMA 相关的 buffer，以匹配 EMAVectorQuantizer 的权重格式
        # 这些 buffer 在推理时不会被使用，但需要存在以正确加载权重
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embedding.clone())

    def forward(self, inputs):
        # 推理时只需要查表
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return None, None, encoding_indices.squeeze(1) # 只返回 indices

    def get_codebook_entry(self, indices):
        # 使用 F.embedding 以匹配 EMAVectorQuantizer 的实现
        return F.embedding(indices, self.embedding)

class ResidualVQ(nn.Module):
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, dropout=0.0, commitment_cost=0.25):
        super().__init__()
        # 注意：commitment_cost 参数需要与训练时一致（train_taae.py 中使用的是 0.25）
        # 虽然推理时不会用到 commitment_cost，但为了保持结构一致，还是传递这个参数
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost=commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(self, x, n_layers=None):
        """
        Args:
            x: [B, embedding_dim] 或 [B, T, embedding_dim] - 输入 latent
            n_layers: 使用前几层进行编码，如果为 None，使用所有层
        Returns:
            codes: [B, n_layers] 或 [B, T, n_layers]
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
        
        residual = x
        all_indices = []
        
        # 确定使用多少层
        if n_layers is None:
            layers_to_use = self.layers
        else:
            if n_layers > len(self.layers):
                raise ValueError(f"n_layers ({n_layers}) > total layers ({len(self.layers)})")
            if n_layers < 1:
                raise ValueError("n_layers must be >= 1")
            layers_to_use = self.layers[:n_layers]
        
        # 推理模式：逐层量化，计算残差
        for layer in layers_to_use:
            # 这里我们复用 forward 计算距离找 index
            # 注意：训练代码里 forward 返回 quantized, loss, indices
            # 我们只需要 indices，然后查表得到 quantized 用于算残差
            _, _, indices = layer(residual)
            all_indices.append(indices)
            
            # 查表得到当前层的量化值
            x_q = layer.get_codebook_entry(indices)
            residual = residual - x_q
            
        codes = torch.stack(all_indices, dim=1)  # [B*T, n_layers] 或 [B, n_layers]
        
        # 如果是 3D 输入，reshape codes 回 [B, T, n_layers]
        if need_reshape:
            codes = codes.view(B, T, -1)
        
        return None, None, codes

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
        
        # 使用 embedding 的 dtype（通常是 float32）
        # embedding 现在是 buffer，不是 nn.Embedding，所以直接访问 .dtype
        embedding_dtype = self.layers[0].embedding.dtype
        quantized_out = torch.zeros(indices.shape[0], embedding_dim, device=indices.device, dtype=embedding_dtype)
        for i, layer in enumerate(self.layers):
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

class TrajRVQTransformer(nn.Module):
    """
    运动学 RVQ Transformer（推理版）。
    与 train_tfm.py 中的结构完全一致：
    - Encoder: [B, T, C] → flatten → TransformerEncoder → latent [B, D]
    - RVQ: latent → token codes [B, num_layers]
    - Decoder: codes → TransformerDecoder → 运动学 heads (v/κ/dy) → 公式 rollout → [B, T, C]
    """

    def __init__(
        self,
        input_steps=25,
        input_dim=3,
        num_layers=10,
        vocab_size=512,
        d_model=256,
        nhead=8,
        num_transformer_layers=4,
        dt=0.2,
    ):
        super().__init__()
        self.input_steps = input_steps
        self.input_dim = input_dim
        self.input_flat_dim = input_steps * input_dim
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dt = dt

        # --- Encoder ---
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

        # --- RVQ（推理版，只做查表）---
        self.rvq = ResidualVQ(
            num_quantizers=num_layers,
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            dropout=0.0,
            commitment_cost=0.25,
        )

        # --- Decoder: Transformer + 运动学 heads ---
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

        # 运动学参数 heads（输出物理空间的 v / κ / dy）
        self.v_head = nn.Linear(d_model, input_steps)
        self.kappa_head = nn.Linear(d_model, input_steps)
        self.dy_head = nn.Linear(d_model, input_steps)

        nn.init.constant_(self.v_head.bias, 10.0)

        # 归一化参数 buffers（从训练时的 state_dict 加载）
        self.register_buffer('norm_mean', torch.zeros(1, 1, 3))
        self.register_buffer('norm_std', torch.ones(1, 1, 3))
        self.register_buffer('norm_scale', torch.ones(1, 1, 3))

    def _kinematic_decode(self, h_dec):
        """运动学解码：v/κ/dy → dx=v*dt, dyaw=v*κ*dt → 归一化空间"""
        v = F.softplus(self.v_head(h_dec))                      # [B, T]
        kappa = torch.tanh(self.kappa_head(h_dec)) * 0.5        # [B, T]
        dy_phys = self.dy_head(h_dec) * 0.01                    # [B, T]

        dx_phys = v * self.dt
        dyaw_phys = v * kappa * self.dt

        x_phys = torch.stack([dx_phys, dy_phys, dyaw_phys], dim=-1)  # [B, T, 3]
        x_norm = (x_phys - self.norm_mean) / (self.norm_std + 1e-8) / self.norm_scale
        return x_norm

    def encode(self, x: torch.Tensor, n_layers=None):
        """x: [B, T, C] → z: [B, D]"""
        B, T, C = x.shape
        assert T == self.input_steps and C == self.input_dim

        x_flat = x.view(B, self.input_flat_dim)
        h = self.input_proj(x_flat)
        h = h.unsqueeze(1)
        h = h + self.pos_embed
        h = self.transformer_encoder(h)
        h = h.squeeze(1)
        z = self.to_latent(h)
        return z

    def encode_to_codes(self, x, n_layers=None):
        """x: [B, T, C] → codes: [B, n_layers]"""
        z = self.encode(x)
        _, _, codes = self.rvq(z, n_layers=n_layers)
        return codes

    def decode_from_codes(self, codes: torch.Tensor):
        """codes: [B, num_layers] → x_recon: [B, T, 3] (归一化空间)"""
        z_q = self.rvq.decode_from_codes(codes)

        h_dec = z_q.unsqueeze(1)
        h_dec = h_dec + self.decoder_pos_embed
        h_dec = self.transformer_decoder(h_dec)
        h_dec = h_dec.squeeze(1)

        x_recon = self._kinematic_decode(h_dec)
        return x_recon

# ==========================================
# 2. Tokenizer 类实现
# ==========================================
class RVQTFMTokenizer:

    def __init__(self, work_dir="./work_dirs", data_type="pred", input_steps=25, device=None, n_layers=None,
                 enable_post_smoothing: bool = False):
        """
        Args:
            work_dir:   包含 *_rvq_taae_model.pth 和 *_norm_params.pkl 的目录
            data_type:  'pred' 或 'history' 等，用于区分不同模型 / 归一化参数
            input_steps: 轨迹步数 (pred: 25, history: 14)
            n_layers:   指定使用的 RVQ 层数，用于统计计算量（默认 None，使用所有层）
        """
        # 是否在 decode 阶段做简单的时序后处理平滑
        self.enable_post_smoothing = enable_post_smoothing

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        model_path = os.path.join(work_dir, f"{data_type}_rvq_taae_model.pth")
        norm_path = os.path.join(work_dir, f"{data_type}_norm_params.pkl")

        # 1. 加载归一化参数
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Norm params not found at {norm_path}")

        with open(norm_path, "rb") as f:
            norm_params = pickle.load(f)

        # 归一化参数保存时是 [1, 1, 3] 形状（keepdims=True）
        # 直接转换为 Tensor，PyTorch 会自动处理广播
        # 与 rvq_tokenizer.py 保持一致的处理方式
        self.mean = torch.tensor(norm_params["mean"], device=self.device, dtype=torch.float32)
        self.std = torch.tensor(norm_params["std"], device=self.device, dtype=torch.float32)
        self.scale_factor = torch.tensor(
            norm_params["scale_factor"], device=self.device, dtype=torch.float32
        )
        if "clip_limit" in norm_params:
            self.clip_limit = torch.tensor(
                norm_params["clip_limit"], device=self.device, dtype=torch.float32
            )
        else:
            self.clip_limit = None

        self.model = TrajRVQTransformer(
            input_steps=input_steps,
            input_dim=3,
            num_layers=15,
            vocab_size=1024,
            d_model=128,
            nhead=4,
            num_transformer_layers=2,
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print(f"RVQTAAETokenizer loaded from {work_dir}, data_type={data_type}")
        print(f"  Normalization params shape: mean={self.mean.shape}, std={self.std.shape}, scale_factor={self.scale_factor.shape}")
        print(f"  Normalization params values: mean={self.mean.squeeze().cpu().numpy()}, std={self.std.squeeze().cpu().numpy()}, scale_factor={self.scale_factor.squeeze().cpu().numpy()}")
        
        # 打印模型规模与计算量信息（可以指定使用的层数）
        self._print_model_profile(n_layers=n_layers)

    def _print_model_profile(self, n_layers=None):
        """
        打印模型参数量、显存占用以及大致的计算量估算（以单条轨迹前向为单位）。
        可以指定使用的 RVQ 层数，用于评估不同层数配置下的开销。
        
        Args:
            n_layers: 使用的 RVQ 层数，如果为 None，使用所有层
        """
        model = self.model
        
        # 确定实际使用的层数
        if n_layers is None:
            actual_layers = model.num_layers
        else:
            actual_layers = min(n_layers, model.num_layers)
        
        # 1. 参数量与显存占用
        # 注意：参数量是固定的（所有层都加载了），但计算量会根据 n_layers 变化
        total_params = sum(p.numel() for p in model.parameters())
        total_params_m = total_params / 1e6
        size_mb_fp32 = total_params * 4 / (1024 ** 2)
        size_mb_fp16 = total_params * 2 / (1024 ** 2)
        
        # 2. 计算量估算（MACs - Multiply-Accumulate Operations）
        # 输入: [1, T, 3] = [1, input_steps, 3]，展平后 [1, T*3]
        T = model.input_steps
        input_flat_dim = model.input_flat_dim  # T * input_dim
        d_model = model.d_model
        
        total_macs = 0
        
        # 2.1 Input 投影 + Transformer Encoder（纯 Transformer，无 Conv）
        # input_proj: Linear(input_flat_dim, d_model)
        input_proj_macs = input_flat_dim * d_model
        total_macs += input_proj_macs
        
        # 2.2 Transformer Encoder（序列长度=1，因为展平后当作单 token）
        # seq_len = 1，所以 Attention 的 T*T 项退化为 1
        seq_len = 1
        attention_macs_per_layer = (
            3 * d_model * d_model * seq_len +  # Q/K/V 投影
            2 * seq_len * seq_len * d_model +  # Attention 计算 (1x1)
            d_model * d_model * seq_len  # Output 投影
        )
        ffn_macs_per_layer = 2 * d_model * 4 * d_model * seq_len  # FFN
        transformer_encoder_macs = model.num_transformer_layers * (attention_macs_per_layer + ffn_macs_per_layer)
        total_macs += transformer_encoder_macs
        
        # 2.3 to_latent: LayerNorm + ReLU + Linear(d_model, d_model)
        # LayerNorm/ReLU 忽略，主要 Linear: d_model * d_model
        to_latent_macs = d_model * d_model
        total_macs += to_latent_macs
        
        # 2.4 RVQ 量化（根据实际使用的层数）
        # 每一层 VectorQuantizer 的距离计算: [1, D] x [D, K] 的 matmul
        # 估算为 2 * D * K MACs（乘加各一次）
        rvq_macs = 0
        if hasattr(model, "rvq") and len(model.rvq.layers) > 0:
            vq_layer = model.rvq.layers[0]  # 所有层参数相同
            # embedding 现在是 buffer (tensor)，属性在 VectorQuantizer 类中
            num_embeddings = vq_layer.num_embeddings
            embedding_dim = vq_layer.embedding_dim
            # 每层的距离计算：计算到所有 codebook entries 的距离
            rvq_macs_per_layer = 2 * num_embeddings * embedding_dim
            rvq_macs = rvq_macs_per_layer * actual_layers
        
        total_macs += rvq_macs
        
        # 2.5 Transformer Decoder（序列长度=1）
        transformer_decoder_macs = model.num_transformer_layers * (attention_macs_per_layer + ffn_macs_per_layer)
        total_macs += transformer_decoder_macs
        
        # 2.6 运动学 Heads: 3 × Linear(d_model, input_steps)
        kin_heads_macs = 3 * d_model * T
        total_macs += kin_heads_macs
        
        total_macs_m = total_macs / 1e6
        
        print("\n[RVQTFMTokenizer Model Profile (Kinematic Decoder)]")
        print(f"  参数量: {total_params_m:.3f} M params (所有层)")
        print(f"  模型大小(FP32): {size_mb_fp32:.2f} MB")
        print(f"  模型大小(FP16): {size_mb_fp16:.2f} MB")
        print(f"  使用的 RVQ 层数: {actual_layers}/{model.num_layers}")
        print(f"  单条轨迹前向估算 MACs: ~{total_macs_m:.3f} M (使用 {actual_layers} 层 RVQ)")
        print(f"     - Input Projection: {input_proj_macs / 1e6:.3f} M")
        print(f"     - Transformer Encoder: {transformer_encoder_macs / 1e6:.3f} M")
        print(f"     - to_latent: {to_latent_macs / 1e6:.3f} M")
        print(f"     - RVQ ({actual_layers} layers): {rvq_macs / 1e6:.3f} M")
        print(f"     - Transformer Decoder: {transformer_decoder_macs / 1e6:.3f} M")
        print(f"     - Kinematic Heads (v/κ/dy): {kin_heads_macs / 1e6:.3f} M")

    def __call__(self, traj_data, n_layers=None):
        """
        Encode: [..., T, 3] -> [..., n_layers] Tokens
        
        Args:
            traj_data: 输入轨迹数据 [..., T, 3]
            n_layers: 使用前几层进行编码，如果为 None，使用所有层（默认10层）
                     例如：n_layers=5 会输出5个token
        
        Returns:
            codes: [..., n_layers] int64 tensor，每条轨迹对应 n_layers 个 token
        """
        # 数据转 Tensor
        if isinstance(traj_data, np.ndarray):
            x = torch.from_numpy(traj_data).float().to(self.device)
        else:
            x = traj_data.float().to(self.device)

        original_shape = x.shape  # [..., T, 3]

        # Flatten batch 维
        if len(original_shape) > 3:
            x = x.view(-1, self.model.input_steps, self.model.input_dim)
        elif len(original_shape) == 2:  # [T, 3]
            x = x.unsqueeze(0)

        # 1. 归一化 (与 train_taae.py 中的 preprocess_and_save_norm_params 保持一致)
        # 训练时：data_z = (data_array - mean) / (std + 1e-8)
        #          data_z_clipped = np.clip(data_z, -quantile_limit, quantile_limit)  # 如果使用 clip
        #          data_normalized = data_z_clipped / scale_factor
        # 推理时编码：x_norm = (x - mean) / (std + 1e-8)
        #            x_norm_clipped = clip(x_norm, -clip_limit, clip_limit)  # 如果训练时使用了 clip
        #            x_scaled = x_norm_clipped / scale_factor
        x_norm = (x - self.mean) / (self.std + 1e-8)
        # 如果训练时使用了 clip，推理时也需要 clip
        if self.clip_limit is not None:
            x_norm = torch.clamp(x_norm, -self.clip_limit, self.clip_limit)
        x_scaled = x_norm / self.scale_factor

        # 2. Encode（可以指定使用前几层）
        with torch.no_grad():
            codes = self.model.encode_to_codes(x_scaled, n_layers=n_layers)  # [B, n_layers]

        # 恢复 Batch 维
        if len(original_shape) == 2:
            return codes.squeeze(0)  # [n_layers]
        elif len(original_shape) > 3:
            # 将 [B, n_layers] reshape 回 [..., n_layers]
            return codes.view(*original_shape[:-2], codes.shape[-1])

        return codes  # [B, n_layers]

    def decode(self, tokens, n_layers=10):
        """
        Decode: Tokens -> Trajectory
        Args:
            tokens: [..., num_layers] int tensor
                    - 轨迹级 token，每条轨迹用 num_layers 个 token 表示
            n_layers: 使用前几层 token 进行解码（<= num_layers）
        Returns:
            traj_recon: [..., T, 3] numpy array
        """
        # 1. 转 Tensor
        if isinstance(tokens, np.ndarray):
            codes = torch.from_numpy(tokens).long().to(self.device)
        else:
            codes = tokens.long().to(self.device)

        original_shape = codes.shape  # e.g. [B, num_layers] or [num_layers] or [..., num_layers]

        # 2. 处理维度：统一转换为 [B, num_layers] 的轨迹级 token
        # 记录原始 batch 维度（去掉最后一个 num_layers 维），用于后续恢复
        batch_dims = original_shape[:-1]

        if codes.dim() == 1:
            # [num_layers] -> [1, num_layers]
            codes = codes.unsqueeze(0)
        elif codes.dim() > 2:
            # [..., num_layers] -> [B, num_layers]
            codes = codes.view(-1, codes.shape[-1])

        # 3. 选择使用的层数
        total_layers = codes.shape[-1]
        if n_layers is not None:
            if n_layers > total_layers:
                print(
                    f"Warning: n_layers ({n_layers}) > input tokens ({total_layers}). Using all."
                )
                n_layers = total_layers
            elif n_layers < 1:
                raise ValueError("n_layers must be >= 1")
            codes = codes[..., :n_layers]

        # 4. Decode（轨迹级解码）
        with torch.no_grad():
            x_recon_scaled = self.model.decode_from_codes(codes)  # [B, T, 3]
        
        # 调试信息：检查模型输出
        if x_recon_scaled.shape[0] == 1:
            print(f"  Debug: x_recon_scaled shape={x_recon_scaled.shape}, first few values={x_recon_scaled[0, :3, :].cpu().numpy()}")

        # 4. 反归一化 (与训练时的归一化过程完全对应)
        # 训练时归一化：data_normalized = (data_array - mean) / (std + 1e-8) / scale_factor
        # 推理时反归一化：x_recon = x_recon_scaled * scale_factor * (std + 1e-8) + mean
        # 注意：顺序必须与归一化时完全相反
        x_recon_norm = x_recon_scaled * self.scale_factor
        x_recon = x_recon_norm * (self.std + 1e-8) + self.mean

        # 4.5 可选：对时间维做轻量后处理平滑（减少小锯齿），默认关闭
        # 使用简单的 3 点滑动平均滤波：y_t = (x_{t-1} + x_t + x_{t+1}) / 3
        if self.enable_post_smoothing:
            # x_recon: [B, T, 3]
            B_dec, T_dec, C_dec = x_recon.shape
            if T_dec >= 5:
                # 在时间维 pad 一下，方便卷积
                # 采用 edge padding，避免边界明显缩小
                pad = 2
                x_pad = torch.zeros(B_dec, T_dec + 2 * pad, C_dec, device=x_recon.device, dtype=x_recon.dtype)
                x_pad[:, pad:-pad, :] = x_recon
                x_pad[:, :pad, :] = x_recon[:, :1, :].expand(-1, pad, -1)
                x_pad[:, -pad:, :] = x_recon[:, -1:, :].expand(-1, pad, -1)
                # 简单 5 点均值滤波核 [1,1,1,1,1] / 5
                kernel = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=x_recon.device, dtype=x_recon.dtype) / 5.0
                # 手工卷积：对每个通道做一维卷积
                smoothed = []
                for c in range(C_dec):
                    # [B, T+2*pad]
                    xc = x_pad[:, :, c]
                    # 滑动窗口求平均
                    xc_s = (
                        xc[:, 0:-4] * kernel[0]
                        + xc[:, 1:-3] * kernel[1]
                        + xc[:, 2:-2] * kernel[2]
                        + xc[:, 3:-1] * kernel[3]
                        + xc[:, 4:] * kernel[4]
                    )
                    smoothed.append(xc_s.unsqueeze(-1))
                x_recon = torch.cat(smoothed, dim=-1)

        # 5. 恢复形状
        # x_recon 现在是 [B, T, 3]，需要恢复到原始 batch 维度
        if len(batch_dims) == 0:
            # 输入是 [num_layers] 或 [B, num_layers]，输出分别是 [T,3] 或 [B,T,3]
            if len(original_shape) == 1:
                out_tensor = x_recon.squeeze(0)  # [T,3]
            else:
                out_tensor = x_recon  # [B,T,3]
        else:
            # 输入是 [..., num_layers]，输出应该是 [..., T, 3]
            target_shape = list(batch_dims) + [self.model.input_steps, self.model.input_dim]
            out_tensor = x_recon.view(*target_shape)

        return out_tensor.cpu().numpy()

if __name__ == "__main__":
    # data_info = np.load('/data-algorithm-hl/zhe.du/planner/plannn2/data_v3/tokenizer/pca_tokenizer_urban_highway_uturn_aeb_0116/sample_trajectorys_by_scenario.npy', allow_pickle=True).item()
    # datas = data_info['trajs']
    data_paths = [
        "/home/zhe.du/code/planner/nio_planner/v3/dct_acc/work_dirs/dxdydyaw/all_datas.npy",
        "/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_10000_fulldata.npy",
        "/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_300000_fulldata.npy",
        "/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_5000_highwaydata.npy",
        # '/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_3000_highwaydata.npy',
        # '/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_6000_highwaydata.npy',
        "/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_5000_uturn.npy",
        # '/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_10000_uturn.npy',
        "/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_2000_aeb.npy",
        # '/home/zhe.du/code/planner/nio_planner/nn2_tools/work_dirs/pca_tokenizer_dxdydyaw_highway/trajs_10000_aeb.npy',
    ]

    all_datas = []
    for path in data_paths:
        data = np.load(path, allow_pickle=True).item()
        all_datas.append(data["trajs"])
        print(f'{path} 数据量: {data["trajs"].shape[0]}')
    datas = np.concatenate(all_datas, axis=0)
    print("拼接后数据 datas.shape:", datas.shape)
    
    type = 'pred'
    num_steps = 25
    num_layers = 15
    if type == 'history':
        datas = datas[:, :14, :]
        num_steps = 14

    # sample_data = datas[1757878]
    sample_data = datas[2]

    tokenizer = RVQTFMTokenizer(work_dir="./work_dirs/tokenizer/rvq_tfm_kin_0311", data_type=type, input_steps=num_steps, n_layers=num_layers, enable_post_smoothing=False)
    tokens = tokenizer(sample_data, n_layers=num_layers)
    print(f"\nTokens: {tokens.cpu().numpy()}")

    recon_data = tokenizer.decode(tokens, n_layers=num_layers)
    
    # 4. 误差分析
    # A. 控制量误差
    dx_error = np.mean((sample_data[:,0] - recon_data[:,0])**2)
    dy_error = np.mean((sample_data[:,1] - recon_data[:,1])**2)
    dyaw_error = np.mean((sample_data[:,2] - recon_data[:,2])**2)
    
    # B. 轨迹位置误差 (积分后)
    gt_traj = integrate_trajectory_keyframe(sample_data[None, ...])
    recon_traj = integrate_trajectory_keyframe(recon_data[None, ...])
    
    pos_err = np.linalg.norm(gt_traj[0, :, :2] - recon_traj[0, :, :2], axis=1)
    max_pos_err = np.max(pos_err)
    final_pos_err = pos_err[-1]

    yaw_error = gt_traj[0, :, 2] - recon_traj[0, :, 2]

    np.set_printoptions(precision=5, suppress=True)


    print(f"sample_data: {sample_data}")
    print(f"recon_data: {recon_data}")

    print("-" * 30)
    print(f"控制量 dx MSE    : {dx_error:.6f}")
    print(f"控制量 dy MSE    : {dy_error:.6f}")
    print(f"控制量 dyaw MSE: {dyaw_error:.6f}")

    print("-" * 30)
    print(f"dx Peak (GT vs Recon): {sample_data[:,0].min():.3f} vs {recon_data[:,0].min():.3f}")
    print(f"dy Peak (GT vs Recon): {sample_data[:,1].min():.3f} vs {recon_data[:,1].min():.3f}")
    print(f"dyaw Peak (GT vs Recon): {sample_data[:,2].min():.3f} vs {recon_data[:,2].min():.3f}")
    print("-" * 30)
    print(f"gt_traj: {gt_traj[0]}")
    print(f"recon_traj: {recon_traj[0]}")
    print(f"pos_err: {pos_err}")
    print(f"最大位置误差 (Max Drift)  : {max_pos_err:.4f} m")
    print(f"终点位置误差 (Final Drift): {final_pos_err:.4f} m")
    
