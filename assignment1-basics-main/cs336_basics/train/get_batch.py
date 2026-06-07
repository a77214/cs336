import torch
import numpy as np
import numpy.typing as npt

def get_batch(
        dataset:npt.NDArray,
        batch_size:int,
        context_length:int,
        device:str,
        )-> tuple[torch.Tensor, torch.Tensor]:
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    n = int(dataset.shape[0])
    if n < context_length + 1:
        raise ValueError(
            f"dataset too small: need at least context_length+1 tokens, got n={n}, context_length={context_length}")
    max_start=n-context_length-1
    starts=np.random.randint(0,max_start+1,batch_size,dtype=np.int64)

    offsets=np.arange(context_length+1,dtype=np.int64)
    indices=starts[:, None] + offsets[None, :]
    block=dataset[indices]

    input=block[:,:-1]
    target=block[:,1:]


    inputs = torch.from_numpy(input).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(target).to(device=device, dtype=torch.long)

    return inputs, targets

