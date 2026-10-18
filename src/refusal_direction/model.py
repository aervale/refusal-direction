"""Model loading, prompt formatting, batched forward passes and generation.

Everything here is plumbing (provided). The interesting parts live in directions.py,
interventions.py, selection.py and orthogonalize.py.
"""

from contextlib import nullcontext

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

# Qwen2.5 chat format, written out explicitly so you can see exactly what the model sees.
# The paper calls the tokens *after* the instruction the "post-instruction" (eoi) tokens:
#   <|im_end|> \n <|im_start|> assistant \n
# Candidate refusal directions are extracted at each of these positions.
USER_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
SYSTEM_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n"
EOI_STRING = "<|im_end|>\n<|im_start|>assistant\n"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str = "auto") -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(cfg) -> HookedTransformer:
    """Load the chat model into TransformerLens.

    We use `from_pretrained_no_processing` so the weights are *exactly* the HF weights
    (no LayerNorm folding / centering). That keeps the residual stream in the same basis
    as the original model, which matters when we later edit weights (orthogonalize.py).
    """
    torch.set_grad_enabled(False)  # we never train anything
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.model_name,
        device=pick_device(),
        dtype=pick_dtype(cfg.dtype),
        default_padding_side="left",
    )
    model.tokenizer.padding_side = "left"  # left-pad so position -1 is always the last real token
    model.eval()
    return model


def format_prompt(instruction: str, system_prompt: str | None = None) -> str:
    prefix = SYSTEM_TEMPLATE.format(system=system_prompt) if system_prompt else ""
    return prefix + USER_TEMPLATE.format(instruction=instruction)


def tokenize_instructions(model, instructions, system_prompt=None, suffix: str = ""):
    """Return (tokens, attention_mask), both [batch, pos], left-padded.

    `suffix` is appended to each instruction *inside* the user turn (used for adversarial
    suffixes in the mechanistic experiments).
    """
    prompts = [format_prompt(instr + suffix, system_prompt) for instr in instructions]
    enc = model.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    return enc.input_ids.to(model.cfg.device), enc.attention_mask.to(model.cfg.device)


def eoi_positions(model) -> list[int]:
    """Negative indices of the post-instruction tokens, e.g. [-5, -4, -3, -2, -1]."""
    n = len(model.tokenizer.encode(EOI_STRING, add_special_tokens=False))
    return list(range(-n, 0))


def batches(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield slice(i, min(i + batch_size, n))


@torch.no_grad()
def last_token_logits(model, tokens, attention_mask, fwd_hooks=(), batch_size=16) -> torch.Tensor:
    """Logits at the final prompt position (= the model's first response token). [batch, d_vocab], float32."""
    outs = []
    ctx = model.hooks(fwd_hooks=list(fwd_hooks)) if fwd_hooks else nullcontext()
    with ctx:
        for b in batches(tokens.shape[0], batch_size):
            logits = model(tokens[b], attention_mask=attention_mask[b])
            outs.append(logits[:, -1, :].float())
    return torch.cat(outs)


@torch.no_grad()
def generate(model, tokens, attention_mask, fwd_hooks=(), max_new_tokens=64, batch_size=16, show_progress=True):
    """Greedy decoding with optional hooks active. Returns a list of decoded completions.

    Deliberately simple (no KV cache): every step re-runs the whole sequence, so hooks
    see every position on every step. Slower, but there's no cache subtlety to reason about.
    """
    eos_ids = {model.tokenizer.eos_token_id, model.tokenizer.convert_tokens_to_ids("<|im_end|>")}
    pad_id = model.tokenizer.pad_token_id
    completions = []
    ctx = model.hooks(fwd_hooks=list(fwd_hooks)) if fwd_hooks else nullcontext()
    it = list(batches(tokens.shape[0], batch_size))
    with ctx:
        for b in tqdm(it, desc="generating", disable=not show_progress):
            toks, mask = tokens[b], attention_mask[b]
            done = torch.zeros(toks.shape[0], dtype=torch.bool, device=toks.device)
            new = []
            for _ in range(max_new_tokens):
                next_tok = model(toks, attention_mask=mask)[:, -1, :].argmax(-1)
                next_tok = torch.where(done, torch.full_like(next_tok, pad_id), next_tok)
                new.append(next_tok)
                done |= torch.isin(next_tok, torch.tensor(list(eos_ids), device=toks.device))
                toks = torch.cat([toks, next_tok[:, None]], dim=1)
                mask = torch.cat([mask, torch.ones_like(next_tok[:, None])], dim=1)
                if done.all():
                    break
            new = torch.stack(new, dim=1)
            completions += model.tokenizer.batch_decode(new, skip_special_tokens=True)
    return completions
