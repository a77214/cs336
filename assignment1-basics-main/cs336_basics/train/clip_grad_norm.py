import math

import torch


def clip_grad_norm(
        parameters,
        max_norm,
        eps: float = 1e-6)-> float:

    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")
    grads=[]
    for p in parameters:
        grad=p.grad
        if grad is None:
            continue
        if grad.is_sparse:
            raise ValueError("grad is sparse")
        grads.append(grad)
    total_sq=0.0
    for g in grads:
        total_sq+=float(g.detach().float().pow(2).sum().item())
    total_norm=math.sqrt(total_sq)
    clip_norm=max_norm/(total_norm+eps)
    if clip_norm<1:
        for g in grads:
            g.mul_(clip_norm)
    return total_norm





