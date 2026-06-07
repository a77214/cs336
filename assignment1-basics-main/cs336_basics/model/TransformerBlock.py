import torch
from torch import nn

from cs336_basics.model.CausalMultiHeadSelfAttentionWithRoPE import CausalMultiHeadSelfAttentionWithRoPE
from cs336_basics.model.RMSNorm import RMSNorm
from cs336_basics.model.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, *,
        max_seq_len: int, theta: float, eps: float = 1e-5,
        device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn=CausalMultiHeadSelfAttentionWithRoPE(
            d_model=self.d_model,
            num_heads=self.num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        self.ln1=RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.ln2=RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.ffn=SwiGLU(d_model,d_ff)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        layer1=x+self.attn(self.ln1(x),token_positions)
        layer2=layer1+self.ffn(self.ln2(layer1))
        return layer2