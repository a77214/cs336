import os
import torch
from typing import BinaryIO, IO, Union

PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        filepath: PathOrFile
) -> None:
    obj = {
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(obj, filepath)


def load_checkpoint(
        src: PathOrFile,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    ckpt = torch.load(src,map_location='cpu')
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")

    if "state_dict" not in ckpt or "optimizer" not in ckpt or "iteration" not in ckpt:
        raise KeyError("Checkpoint dict missing required keys: 'state_dict', 'optimizer', 'iteration'")

    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['iteration']