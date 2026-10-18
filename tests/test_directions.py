"""Exercise 3.   pytest tests/test_directions.py -v"""

import torch

from refusal_direction.directions import diff_in_means, get_mean_activations

POSITIONS = [-3, -2, -1]


def reference_means(model, tokens, mask):
    _, cache = model.run_with_cache(tokens, attention_mask=mask)
    return torch.stack(
        [torch.stack([cache["resid_pre", l][:, p].mean(0) for l in range(model.cfg.n_layers)]) for p in POSITIONS]
    )


def test_mean_activations_shape_and_dtype(model, batch):
    out = get_mean_activations(model, *batch, POSITIONS, batch_size=4)
    assert out.shape == (len(POSITIONS), model.cfg.n_layers, model.cfg.d_model)
    assert out.dtype == torch.float32


def test_mean_activations_match_cache(model, batch):
    out = get_mean_activations(model, *batch, POSITIONS, batch_size=4)
    assert torch.allclose(out, reference_means(model, *batch), atol=1e-5)


def test_mean_activations_independent_of_batch_size(model, batch):
    a = get_mean_activations(model, *batch, POSITIONS, batch_size=1)
    b = get_mean_activations(model, *batch, POSITIONS, batch_size=6)
    assert torch.allclose(a, b, atol=1e-5), "uneven final batch? divide by the number of prompts, not batches"


def test_diff_in_means():
    mu, nu = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    assert torch.allclose(diff_in_means(mu, nu), mu - nu)
