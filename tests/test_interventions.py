"""Exercises 1-2.   pytest tests/test_interventions.py -v"""

import torch

from refusal_direction.interventions import get_ablation_hooks, get_addition_hooks, project_out


def unit(v):
    return v / v.norm()


# ---- Exercise 1a: project_out ----

def test_project_out_removes_component(direction):
    x = torch.randn(4, 7, 32)
    out = project_out(x, direction)
    assert out.shape == x.shape
    assert torch.allclose(out @ unit(direction), torch.zeros(4, 7), atol=1e-5)


def test_project_out_keeps_orthogonal_part(direction):
    x = torch.randn(5, 32)
    x_perp = x - (x @ unit(direction))[:, None] * unit(direction)
    assert torch.allclose(project_out(x, direction), x_perp, atol=1e-5)


def test_project_out_is_idempotent_and_scale_invariant(direction):
    x = torch.randn(3, 32)
    once = project_out(x, direction)
    assert torch.allclose(project_out(once, direction), once, atol=1e-5)
    assert torch.allclose(project_out(x, 100 * direction), once, atol=1e-5)


def test_project_out_preserves_dtype(direction):
    x = torch.randn(3, 32, dtype=torch.float16)
    assert project_out(x, direction).dtype == torch.float16


# ---- Exercise 1b: ablation hooks ----

def test_ablation_hook_names_are_valid(model, direction):
    hooks = get_ablation_hooks(model, direction)
    assert len(hooks) == 3 * model.cfg.n_layers
    for name, fn in hooks:
        assert name in model.hook_dict, f"{name} is not a hook point"
        assert callable(fn)


def test_ablation_removes_direction_from_final_residual(model, batch, direction):
    tokens, mask = batch
    seen = {}
    last = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    capture = (last, lambda act, hook: seen.setdefault("x", act.clone()))
    model.run_with_hooks(tokens, attention_mask=mask, fwd_hooks=get_ablation_hooks(model, direction) + [capture])
    proj = seen["x"] @ unit(direction)
    assert proj.abs().max() < 1e-4, "final residual stream still has a component along the direction"


def test_ablation_changes_output(model, batch, direction):
    tokens, mask = batch
    base = model(tokens, attention_mask=mask)
    ablated = model.run_with_hooks(tokens, attention_mask=mask, fwd_hooks=get_ablation_hooks(model, direction))
    assert not torch.allclose(base, ablated), "hooks had no effect -- are they returning the new activation?"


# ---- Exercise 2: activation addition ----

def test_addition_adds_unnormalized_direction(model, batch, direction):
    tokens, mask = batch
    layer = 1
    name = f"blocks.{layer}.hook_resid_pre"
    store = {}
    cap = lambda key: (name, lambda act, hook: store.setdefault(key, act.clone()))
    model.run_with_hooks(tokens, attention_mask=mask, fwd_hooks=[cap("base")])
    hooks = get_addition_hooks(model, direction, layer, coeff=2.0)
    assert [n for n, _ in hooks] == [name]
    model.run_with_hooks(tokens, attention_mask=mask, fwd_hooks=hooks + [cap("added")])
    assert torch.allclose(store["added"] - store["base"], (2.0 * direction).expand_as(store["base"]), atol=1e-5)
