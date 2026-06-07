import torch
from torch import nn

from cs336_basics.model.Embedding import Embedding
from cs336_basics.model.Linear import Linear
from cs336_basics.model.RMSNorm import RMSNorm
from cs336_basics.model.TransformerBlock import TransformerBlock


class TransformerLM(nn.Module):
    """
    A Transformer language model composed of:
      token embedding -> N pre-norm Transformer blocks -> final RMSNorm -> LM head.

    This implementation uses RoPE inside each TransformerBlock's attention module.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        *,
        rope_theta: float,
        max_seq_len: int | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len if max_seq_len is not None else context_length)

        # Token embedding table: (vocab_size, d_model)
        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device=device, dtype=dtype)
        self.layers= nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=self.max_seq_len,
                    theta=rope_theta,
                    eps=eps,
                    device=device,
                    dtype=dtype
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head=Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        if in_indices.dim() != 2:
            raise ValueError(f"in_indices must have shape (batch, seq_len), got {tuple(in_indices.shape)}")

        batch, seq_len = in_indices.shape
        if seq_len > self.context_length:
            raise ValueError(f"seq_len={seq_len} exceeds context_length={self.context_length}")

        # Token positions for RoPE: (batch, seq_len)
        token_positions = torch.arange(seq_len, device=in_indices.device, dtype=torch.long).view(1, seq_len)
        token_positions = token_positions.expand(batch, seq_len)

        x=self.token_embeddings(in_indices)
        for block in self.layers:
            x = block(x, token_positions)
        x=self.ln_final(x)
        logits = self.lm_head(x)
        return logits
