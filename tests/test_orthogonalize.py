"""Exercise 5.   pytest tests/test_orthogonalize.py -v

The key claim of Section 4: editing the weights is EXACTLY equivalent to ablating with hooks.
"""

import copy

import torch

from refusal_direction.interventions import get_ablation_hooks
from refusal_direction.orthogonalize import orthogonalize_weights


def test_orthogonalized_model_matches_hook_ablation(model, batch, direction):
    tokens, mask = batch
    hooked = model.run_with_hooks(tokens, attention_mask=mask, fwd_hooks=get_ablation_hooks(model, direction))

    edited = copy.deepcopy(model)
    orthogonalize_weights(edited, direction)
    out = edited(tokens, attention_mask=mask)

    real = mask.bool()  # compare only non-pad positions
    assert torch.allclose(out[real], hooked[real], atol=1e-4), (
        "edited model != hook-ablated model. Did you miss a component that writes to the residual stream?"
    )


def test_original_model_untouched_by_copy(model, batch, direction):
    tokens, mask = batch
    before = model(tokens, attention_mask=mask)
    orthogonalize_weights(copy.deepcopy(model), direction)
    assert torch.allclose(model(tokens, attention_mask=mask), before)


def test_orthogonalize_is_idempotent(model, direction):
    orthogonalize_weights(model, direction)
    w = model.blocks[0].mlp.W_out.clone()
    orthogonalize_weights(model, direction)
    assert torch.allclose(model.blocks[0].mlp.W_out, w, atol=1e-6)
