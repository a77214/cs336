import torch
from torch import nn

from cs336_basics.model.Linear import Linear
from cs336_basics.model.scaled_dot_product_attention import scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention (no RoPE).

    This module computes:
        Q = W_Q x, K = W_K x, V = W_V x
        heads = SDPA(Q_heads, K_heads, V_heads, causal_mask)
        out = W_O concat(heads)

    Shapes:
        x:   (..., seq_len, d_model)
        QKV: (..., seq_len, d_model)
        heads view: (..., num_heads, seq_len, head_dim)
        output: (..., seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.head_dim = self.d_model // self.num_heads  # d_k = d_v = d_model / h

        # Separate projections (one matmul each). Combining into one is an optional optimization
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build a (seq_len, seq_len) causal mask where True means "allowed"
        """
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_model)

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
        # Then move heads into a batch-like dimension: (..., num_heads, seq_len, head_dim)
        new_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = q.view(new_shape).transpose(-3, -2)
        k = k.view(new_shape).transpose(-3, -2)
        v = v.view(new_shape).transpose(-3, -2)

        # Causal mask shared across heads and batches
        mask = self._causal_mask(seq_len, device=device)

        # SDPA: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads: (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        # Output projection: (..., seq_len, d_model)
        return self.o_proj(out)


