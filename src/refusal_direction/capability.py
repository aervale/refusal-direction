"""Did we break the model? (paper, Section 4.2). Provided.

The paper runs MMLU, ARC, GSM8K, TruthfulQA, WinoGrande, TinyBench plus CE loss on the
Pile / Alpaca. We do a lightweight version that fits in a Colab session:
  - cross-entropy on WikiText-2 (general language modelling)
  - cross-entropy on harmless chat completions written by the base model (on-distribution)
  - MMLU accuracy on a random subset, scored by the logit of the answer letter
"""

import random

import torch
import torch.nn.functional as F
from datasets import load_dataset

from .model import batches, format_prompt, last_token_logits

LETTERS = "ABCD"


def load_wikitext(n: int = 200, min_chars: int = 400, seed: int = 0) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) >= min_chars and not t.strip().startswith("=")]
    return random.Random(seed).sample(texts, min(n, len(texts)))


@torch.no_grad()
def mean_cross_entropy(model, texts: list[str], max_len: int = 256, batch_size: int = 8) -> float:
    """Average next-token CE (nats) over all real (non-pad) tokens."""
    enc = model.tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                          return_tensors="pt", add_special_tokens=False)
    toks, mask = enc.input_ids.to(model.cfg.device), enc.attention_mask.to(model.cfg.device)
    total, count = 0.0, 0
    for b in batches(toks.shape[0], batch_size):
        logits = model(toks[b], attention_mask=mask[b]).float()
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), toks[b][:, 1:].flatten(), reduction="none")
        m = (mask[b][:, :-1] * mask[b][:, 1:]).flatten().float()
        total += (loss * m).sum().item()
        count += m.sum().item()
    return total / count


def load_mmlu(n: int = 200, seed: int = 0) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [ds[i] for i in idx]


def mmlu_prompt(row) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return f"{row['question']}\n\n{choices}\n\nAnswer with only the letter (A, B, C, or D)."


@torch.no_grad()
def mmlu_accuracy(model, rows: list[dict], system_prompt=None, fwd_hooks=(), batch_size: int = 8) -> float:
    prompts = [format_prompt(mmlu_prompt(r), system_prompt) for r in rows]
    enc = model.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    toks, mask = enc.input_ids.to(model.cfg.device), enc.attention_mask.to(model.cfg.device)
    letter_ids = [model.tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    logits = last_token_logits(model, toks, mask, fwd_hooks=fwd_hooks, batch_size=batch_size)
    preds = logits[:, letter_ids].argmax(-1).tolist()
    return sum(p == r["answer"] for p, r in zip(preds, rows)) / len(rows)
