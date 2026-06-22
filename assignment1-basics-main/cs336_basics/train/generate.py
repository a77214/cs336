import torch
from typing import Optional

from cs336_basics.model.Tokenizer import Tokenizer
from cs336_basics.model.TransformerLM import TransformerLM
from cs336_basics.train.AdamW import AdamW
from cs336_basics.train.config import get_default_config


def top_p_sampling(
    probs: torch.Tensor,
    top_p: float,
)->torch.Tensor:
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be between 0 and 1, but got {top_p}")
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum=torch.cumsum(sorted_probs,-1)
    keep=cum<=top_p
    keep[...,0]=True
    filter_sorted_prob=sorted_probs * keep.to(sorted_probs.dtype)
    filter_sorted_prob=filter_sorted_prob/filter_sorted_prob.sum(-1,keepdim=True)

    out=torch.zeros_like(probs)
    out.scatter_(-1,sorted_indices,filter_sorted_prob)
    return out

@torch.no_grad()
def generate(
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        *,
        end_token_id:int,
        max_new_tokens:int,
        temperature:float=1.0,
        top_p:float=1.0,
) -> torch.Tensor:
    if prompt_ids.dim() != 1:
        raise ValueError(f"prompt_ids must be 1D (t,), got shape {tuple(prompt_ids.shape)}")
    if prompt_ids.dtype != torch.long:
        prompt_ids = prompt_ids.to(torch.long)

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    model_was_training= model.training
    model.eval()

    device = next(model.parameters()).device
    out=prompt_ids.to(device)

    context_length: Optional[int] = getattr(model, "context_length", None)
    for _ in range(max_new_tokens):
        if context_length is not None and out.numel() > context_length:
            inp = out[-context_length:]
        else:
            inp = out
        logits = model(inp)
        next_logits=logits[0,-1,:]

        if temperature == 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            if temperature < 0.0:
                raise ValueError(f"temperature must be >= 0, got {temperature}")
            scaled=next_logits/float(temperature)
            probs=scaled.softmax(dim=-1)
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)

            next_id = int(torch.multinomial(probs, num_samples=1).item())

        out = torch.cat([out, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)

        if next_id == int(end_token_id):
            break
    if model_was_training:
        model.train()

    return out

if __name__ == "__main__":
    EOT = "<|endoftext|>"

    cfg=get_default_config()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.from_files(
        "workspace/tinystories_bpe_vocab.pkl",
        "workspace/tinystories_bpe_merges.pkl",
        special_tokens=[EOT]
    )
    end_token_id=tok.special_id[EOT]
    model_dtype = cfg.model.torch_dtype.lower()
    dtype_map = {"float32": torch.float32, "fp32": torch.float32,
                 "float16": torch.float16, "fp16": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    dtype = dtype_map[model_dtype]

    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=d_ff,
        rope_theta=cfg.model.rope_theta,
        max_seq_len=cfg.model.max_seq_len,
        eps=cfg.model.rmsnorm_eps,
        device=device,
        dtype=dtype,
    ).to(device)
    optimizer=AdamW(model.parameters())
    prompt = "Once upon a time"
    prompt_ids = torch.tensor(tok.encode(prompt), dtype=torch.long)

    out_ids = generate(
        model,
        prompt_ids,
        end_token_id=end_token_id,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.9,
    )
    print (tok.decode(out_ids.tolist()))