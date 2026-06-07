import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Rotates query and key vectors based on their positions to inject
    relative position information into attention mechanisms.

    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块

        Args:
            theta: float - Θ 值，控制旋转频率
            d_k: int - Query 和 Key 向量的维度
            max_seq_len: int - 最大序列长度
            device: torch.device | None - 存储 buffer 的设备
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率: θ_k = theta^{-2(k-1)/d_k} for k = 1,..., d_k/2
        # 更常用的形式: inv_freq = 1 / (theta^(2i/d_k))
        i = torch.arange(0, d_k, 2, device=device).float()
        inv_freq = 1.0 / (theta ** (i / d_k))

        # 预计算所有位置的频率
        # positions: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device).float()

        # 角度: positions[:, None] * inv_freq[None, :]
        # shape: (max_seq_len, d_k/2)
        angles = torch.outer(positions, inv_freq)

        # 预计算 sin 和 cos
        # 需要将每个角度重复两次，因为每个 2D 对需要两个值
        # 方法：先计算 sin/cos，然后 repeat_interleave
        cos = angles.cos()  # (max_seq_len, d_k/2)
        sin = angles.sin()

        # 重复每个值两次，得到 (max_seq_len, d_k)
        # 这样 cos[0] 和 cos[1] 相同，对应同一个 2D 对的两个元素
        cos = cos.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        sin = sin.repeat_interleave(2, dim=-1)

        # 注册为 buffer (不参与训练，但会保存到模型)
        # persistent=False 表示不会出现在 state_dict 中
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用 RoPE 到输入张量

        Args:
            x: torch.Tensor - 输入张量，形状 (..., seq_len, d_k)
            token_positions: torch.Tensor - token 位置，形状 (..., seq_len)

        Returns:
            torch.Tensor - 旋转后的张量，形状与 x 相同
        """
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        cos = cos[..., 0::2]
        sin = sin[..., 0::2]
        x1=x[...,0::2]
        x2=x[...,1::2]
        out1=x1*cos-x2*sin
        out2=x1*sin+x2*cos
        out=torch.stack([out1, out2], dim=-1)
        return out.flatten(-2)


