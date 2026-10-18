"""Exercise 6.   pytest tests/test_mechanistic.py -v"""

import torch

from refusal_direction.mechanistic import head_contributions, projection_by_layer


def test_projection_by_layer_matches_cache(model, batch, direction):
    tokens, mask = batch
    out = projection_by_layer(model, tokens, mask, direction, position=-1, batch_size=4)
    assert out.shape == (tokens.shape[0], model.cfg.n_layers)
    _, cache = model.run_with_cache(tokens, attention_mask=mask)
    r_hat = direction / direction.norm()
    ref = torch.stack([cache["resid_pre", l][:, -1] @ r_hat for l in range(model.cfg.n_layers)], dim=1)
    assert torch.allclose(out, ref, atol=1e-4)


def test_head_contributions_sum_to_attn_out(model, batch, direction):
    tokens, mask = batch
    out = head_contributions(model, tokens, mask, direction, position=-1, batch_size=4)
    assert out.shape == (tokens.shape[0], model.cfg.n_layers, model.cfg.n_heads)
    _, cache = model.run_with_cache(tokens, attention_mask=mask)
    r_hat = direction / direction.norm()
    for l in range(model.cfg.n_layers):
        attn_proj = cache["attn_out", l][:, -1] @ r_hat
        bias_proj = model.blocks[l].attn.b_O @ r_hat
        assert torch.allclose(out[:, l].sum(-1) + bias_proj, attn_proj, atol=1e-4), f"layer {l} mismatch"
