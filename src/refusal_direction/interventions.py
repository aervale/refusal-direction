"""Exercises 1-2: the two interventions from the paper (Section 2.4).

Notation: r is a candidate refusal direction (a d_model vector), r_hat = r / ||r||.

Directional ablation (Eq. 4), applied at EVERY layer and EVERY token position:
    x' = x - r_hat r_hat^T x
  i.e. remove the component of the activation along r_hat. If refusal really is mediated
  by this one direction, ablating it should stop the model from refusing harmful prompts.

Activation addition (Eq. 3), applied at a SINGLE layer l (the one r was extracted from),
at every token position:
    x' = x + r
  Note: r, not r_hat. The un-normalized difference-in-means vector already has the
  "right" magnitude: it moves a harmless mean activation onto the harmful mean.

TransformerLens hooks have the signature  hook_fn(activation, hook) -> new_activation.
A list of (hook_name, hook_fn) pairs can be passed to model.run_with_hooks(...) or
model.hooks(fwd_hooks=...). Hook names look like "blocks.3.hook_resid_pre".
"""

import torch


def project_out(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Exercise 1a. Remove the component of x along `direction`.

    x:         [..., d_model]   (any number of leading dims)
    direction: [d_model]        (NOT necessarily unit norm -- normalize it yourself)
    returns    [..., d_model], same dtype as x

    Gotchas: activations are often fp16/bf16 while the direction is float32, and they may
    live on different devices. Do the math in x's dtype/device.
    """
    r_hat = direction.to(device=x.device, dtype=x.dtype)
    r_hat = r_hat / r_hat.norm()
    return x - (x @ r_hat)[..., None] * r_hat


def get_ablation_hooks(model, direction: torch.Tensor) -> list[tuple[str, callable]]:
    """Exercise 1b. Hooks that ablate `direction` everywhere.

    Following the paper's code, ablate at three places in every block l:
      blocks.{l}.hook_resid_pre   (what the block reads)
      blocks.{l}.hook_attn_out    (what attention writes)
      blocks.{l}.hook_mlp_out     (what the MLP writes)
    Question to think about: why isn't hooking resid_pre alone enough to guarantee the
    final residual stream (what the unembedding reads) has no r_hat component?
    """
    def hook_fn(activation, hook):
        return project_out(activation, direction)

    hooks = []
    for layer in range(model.cfg.n_layers):
        for name in ("hook_resid_pre", "hook_attn_out", "hook_mlp_out"):
            hooks.append((f"blocks.{layer}.{name}", hook_fn))
    return hooks


def get_addition_hooks(model, direction: torch.Tensor, layer: int, coeff: float = 1.0) -> list[tuple[str, callable]]:
    """Exercise 2. Hook that adds coeff * direction to blocks.{layer}.hook_resid_pre at every position.

    Use the direction as given (do not normalize). coeff=-1 subtracts it, which is the
    paper's other way to bypass refusal -- try it later.
    """
    def hook_fn(activation, hook):
        return activation + coeff * direction.to(device=activation.device, dtype=activation.dtype)

    return [(f"blocks.{layer}.hook_resid_pre", hook_fn)]
