"""Exercise 3: extract candidate refusal directions by difference-in-means (Section 2.3, Eq. 1).

For every layer l and every post-instruction position i:
    mu^{(l)}_i = mean over harmful prompts of  x^{(l)}_i(t)
    nu^{(l)}_i = mean over harmless prompts of x^{(l)}_i(t)
    r^{(l)}_i  = mu^{(l)}_i - nu^{(l)}_i
where x^{(l)}_i is the residual stream *at the input of layer l* (TransformerLens:
"blocks.{l}.hook_resid_pre") at position i.

That gives |positions| x n_layers candidate directions; selection.py picks one.
"""

import torch

from .model import batches


@torch.no_grad()
def get_mean_activations(model, tokens, attention_mask, positions: list[int], batch_size: int = 16) -> torch.Tensor:
    """Exercise 3a. Mean residual-stream activation at each (position, layer).

    tokens, attention_mask: [n_prompts, seq]  (left-padded, so negative positions line up
                            across prompts: -1 is always the last token)
    positions:              negative indices, e.g. [-5, ..., -1]
    returns:                [len(positions), n_layers, d_model], float32

    Suggested approach: loop over batches; for each, run the model with hooks on every
    blocks.{l}.hook_resid_pre that add activation[:, positions, :] (summed over the batch,
    cast to float32) into a running-sum tensor. Divide by n_prompts at the end.
    Alternative: model.run_with_cache(..., names_filter=...) -- simpler, more memory.
    Either way, pass attention_mask to the model.
    Accumulate in float32: summing hundreds of fp16 activations loses precision.
    """
    raise NotImplementedError("Exercise 3a: compute mean resid_pre activations at each (position, layer)")


def diff_in_means(mean_harmful: torch.Tensor, mean_harmless: torch.Tensor) -> torch.Tensor:
    """Exercise 3b. One line. Why harmful - harmless and not the other way round?"""
    raise NotImplementedError("Exercise 3b")


def get_candidate_directions(model, harmful, harmless, positions, batch_size=16) -> torch.Tensor:
    """harmful/harmless: (tokens, attention_mask) tuples. Returns [n_pos, n_layers, d_model]. (provided)"""
    mu = get_mean_activations(model, *harmful, positions, batch_size)
    nu = get_mean_activations(model, *harmless, positions, batch_size)
    return diff_in_means(mu, nu)
