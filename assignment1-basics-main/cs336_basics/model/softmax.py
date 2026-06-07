import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max=torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    sum_exp=torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp
