"""Choosing one direction out of n_pos x n_layers candidates (Section 2.3 + Appendix C).

For each candidate r^{(l)}_i we compute, on the *validation* sets:
  bypass_score : mean refusal_score on harmful prompts with r ABLATED everywhere.
                 Lower = ablation removes refusal better.
  induce_score : mean refusal_score on harmless prompts with r ADDED at layer l.
                 Higher = adding r makes the model refuse innocuous requests.
  kl_score     : KL(base || ablated) of the next-token distribution on harmless prompts.
                 Lower = ablation barely changes normal behavior.
Then filter and pick (Exercise 4).
"""

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .interventions import get_ablation_hooks, get_addition_hooks
from .model import last_token_logits
from .refusal import refusal_score


@torch.no_grad()
def filter_by_refusal(model, instructions, tokenize_fn, refusal_toks, want_refusal: bool, n: int, batch_size=16):
    """Keep prompts the base model handles 'as labelled' (paper, App. A). (provided)

    Some 'harmful' prompts don't actually get refused and some 'harmless' ones do; they add
    noise to the means, so the paper drops them.
    """
    toks, mask = tokenize_fn(instructions)
    scores = refusal_score(last_token_logits(model, toks, mask, batch_size=batch_size), refusal_toks)
    keep = (scores > 0) if want_refusal else (scores < 0)
    kept = [x for x, k in zip(instructions, keep.tolist()) if k]
    if len(kept) < n:
        print(f"warning: only {len(kept)} prompts survived filtering (wanted {n})")
    return kept[:n]


@torch.no_grad()
def score_candidates(model, candidates, eoi_positions, harmful_val, harmless_val, refusal_toks, batch_size=16):
    """candidates: [n_pos, n_layers, d_model]; *_val: (tokens, mask). Returns a DataFrame. (provided)"""
    base_harmless_logits = last_token_logits(model, *harmless_val, batch_size=batch_size)
    base_logprobs = base_harmless_logits.log_softmax(-1)

    rows = []
    n_pos, n_layers, _ = candidates.shape
    for pi in tqdm(range(n_pos), desc="positions"):
        for layer in tqdm(range(n_layers), desc="layers", leave=False):
            r = candidates[pi, layer]

            ablate = get_ablation_hooks(model, r)
            harmful_ablated = last_token_logits(model, *harmful_val, fwd_hooks=ablate, batch_size=batch_size)
            harmless_ablated = last_token_logits(model, *harmless_val, fwd_hooks=ablate, batch_size=batch_size)

            add = get_addition_hooks(model, r, layer)
            harmless_added = last_token_logits(model, *harmless_val, fwd_hooks=add, batch_size=batch_size)

            kl = F.kl_div(harmless_ablated.log_softmax(-1), base_logprobs, log_target=True, reduction="batchmean")
            rows.append(
                dict(
                    pos=eoi_positions[pi],
                    pos_idx=pi,
                    layer=layer,
                    bypass_score=refusal_score(harmful_ablated, refusal_toks).mean().item(),
                    induce_score=refusal_score(harmless_added, refusal_toks).mean().item(),
                    kl_score=kl.item(),
                )
            )
    return pd.DataFrame(rows)


def select_direction(scores: pd.DataFrame, n_layers: int, cfg) -> pd.Series:
    """Exercise 4. Pick the best candidate (paper, Appendix C). Return its row of `scores`.

    Discard candidates where any of these hold:
      - induce_score <= cfg.induce_threshold   (adding it doesn't induce refusal)
      - kl_score     >= cfg.kl_threshold       (ablating it changes harmless behavior too much)
      - layer        >= cfg.prune_layer_fraction * n_layers   (too close to the output; such
                       directions tend to just suppress the literal tokens "I"/"As")
    Of the survivors, return the row with the LOWEST bypass_score.
    Raise a ValueError with a helpful message if nothing survives.

    Then think: which of these filters do you expect to matter most for a 1.5B model?
    Look at the plot in notebook 01 and check.
    """
    ok = (
        (scores.induce_score > cfg.induce_threshold)
        & (scores.kl_score < cfg.kl_threshold)
        & (scores.layer < cfg.prune_layer_fraction * n_layers)
    )
    survivors = scores[ok]
    if survivors.empty:
        raise ValueError("No candidate passed the filters; inspect `scores` and consider loosening thresholds.")
    return survivors.loc[survivors.bypass_score.idxmin()]
