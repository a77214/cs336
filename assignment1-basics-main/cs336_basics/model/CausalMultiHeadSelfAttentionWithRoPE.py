import torch
import torch.nn as nn

from cs336_basics.model.Linear import Linear
from cs336_basics.model.RoPE import RoPE
from cs336_basics.model.scaled_dot_product_attention import scaled_dot_product_attention

class CausalMultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Causal multi-head self-attention with RoPE applied to Q and K (not V).

    This version uses a fused QKV projection:
        qkv = W_qkv x
        q, k, v = split(qkv)
    """

    def __init__(
            self, d_model: int, num_heads: int, theta: float, max_seq_len: int,
            device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads

        # Separate projections (matches reference state_dict keys).
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        # RoPE operates on per-head dimension
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a (seq_len, seq_len) causal mask where True means 'allowed'."""
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_model)
            token_positions: Tensor of shape (..., seq_len)

        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.size(-1)}")

        seq_len = x.size(-2)
        device = x.device

        # Project to Q, K, V: (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads: (..., seq_len, num_heads, head_dim)
        # Then transpose to (..., num_heads, seq_len, head_dim)
        new_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = q.view(new_shape).transpose(-3, -2)
        k = k.view(new_shape).transpose(-3, -2)
        v = v.view(new_shape).transpose(-3, -2)

        # Apply RoPE to Q and K for each head (heads are treated as batch-like dims)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Causal mask shared across heads and batches
        mask = self._causal_mask(seq_len, device=device)

        # Attention: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads back: (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        return self.output_proj(out)
