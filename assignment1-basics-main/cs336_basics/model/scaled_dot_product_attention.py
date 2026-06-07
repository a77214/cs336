import math
import torch

from cs336_basics.model.softmax import softmax


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError("query/key/value must have shape (..., seq_len, d_*)")

    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError("batch dimensions of query, key, value must match")

    d_k = query.shape[-1]
    if d_k != key.shape[-1]:
        raise ValueError("query and key must have the same d_k")

    # Compute attention logits in float32 for stability
    original_dtype = query.dtype
    Q = query.to(torch.float32)
    K = key.to(torch.float32)
    V = value.to(torch.float32)
    score=torch.matmul(Q, K.transpose(-2, -1))
    score=score/math.sqrt(d_k)
    if mask is not None:
        score=score.masked_fill(mask==False,-float('inf'))
    atten=softmax(score,dim=-1)
    output=torch.matmul(atten,V)
    return   output.to(original_dtype)



