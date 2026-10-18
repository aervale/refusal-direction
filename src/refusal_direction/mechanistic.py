"""Exercise 6: mechanistic extras (Section 5 of the paper + some sanity probes).

Questions these let you ask:
  - Where in depth does the refusal direction "light up" for harmful vs harmless prompts?
    (projection_by_layer)
  - Which attention heads write the refusal direction? (head_contributions)
  - When an adversarial suffix jailbreaks the model, does the projection onto r drop, and do
    those heads shift their attention from the instruction to the suffix? (Section 5)
  - What does r "mean" if you read it straight through the unembedding? (logit_lens_direction)
"""

import torch

from .model import batches


@torch.no_grad()
def projection_by_layer(model, tokens, attention_mask, direction, position: int = -1, batch_size: int = 16) -> torch.Tensor:
    """Exercise 6a. Scalar projection of resid_pre onto r_hat at one position, for every layer.

    returns [n_prompts, n_layers], float32.   projection = x . r_hat
    (Hook every blocks.{l}.hook_resid_pre; same pattern as directions.get_mean_activations,
    but keep per-prompt values instead of summing.)
    """
    raise NotImplementedError("Exercise 6a")


@torch.no_grad()
def head_contributions(model, tokens, attention_mask, direction, position: int = -1, batch_size: int = 16) -> torch.Tensor:
    """Exercise 6b. How much each attention head writes along r_hat at `position`.

    returns [n_prompts, n_layers, n_heads], float32.

    Head h's contribution to the residual stream is z_h @ W_O[h], where z is
    blocks.{l}.attn.hook_z with shape [batch, pos, n_heads, d_head] and
    W_O is model.blocks[l].attn.W_O with shape [n_heads, d_head, d_model].
    Project that onto r_hat. (The attention bias b_O is shared, not per-head: leave it out.)

    Sanity check (the test does this): summing over heads and adding b_O . r_hat should
    equal hook_attn_out . r_hat.
    Efficiency hint: you never need the d_model-sized head outputs. W_O[h] @ r_hat is a
    d_head vector, so precompute those once per layer.
    """
    raise NotImplementedError("Exercise 6b")


@torch.no_grad()
def attention_from_position(model, tokens, attention_mask, layer_heads, position: int = -1) -> torch.Tensor:
    """Attention pattern from query `position` for selected heads. (provided)

    layer_heads: list of (layer, head) pairs
    returns [batch, len(layer_heads), key_pos]
    """
    layers = sorted({l for l, _ in layer_heads})
    _, cache = model.run_with_cache(
        tokens,
        attention_mask=attention_mask,
        names_filter=lambda name: any(name == f"blocks.{l}.attn.hook_pattern" for l in layers),
        stop_at_layer=max(layers) + 1,
    )
    return torch.stack([cache["pattern", l][:, h, position, :] for l, h in layer_heads], dim=1).float()


@torch.no_grad()
def logit_lens_direction(model, direction, k: int = 20):
    """Tokens whose unembedding vectors align most / least with r_hat. (provided)

    Caveat: this skips the final RMSNorm's per-dimension scaling (ln_final.w). Try folding it
    in (multiply r_hat elementwise by model.ln_final.w) and see if the story changes.
    """
    r_hat = (direction / direction.norm()).to(model.W_U.dtype).to(model.cfg.device)
    logits = (r_hat @ model.W_U).float()
    top, bottom = logits.topk(k), (-logits).topk(k)
    decode = lambda ids: [model.tokenizer.decode([i]) for i in ids.tolist()]
    return {"top": list(zip(decode(top.indices), top.values.tolist())),
            "bottom": list(zip(decode(bottom.indices), (-bottom.values).tolist()))}
