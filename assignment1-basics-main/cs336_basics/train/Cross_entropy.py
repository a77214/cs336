import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension [..., vocab_size].")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"targets shape {targets.shape} must match logits batch shape {logits.shape[:-1]}.")
    if targets.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise TypeError("targets must be an integer tensor of class indices.")
    logits_max=logits.max(dim=-1, keepdim=True)[0]
    logits_stable=logits-logits_max
    log_sum_exp=torch.logsumexp(logits_stable,dim=-1)
    logits_target=torch.gather(logits_stable,-1,targets.unsqueeze(-1)).squeeze(-1)
    loss=log_sum_exp-logits_target
    return loss.mean()





