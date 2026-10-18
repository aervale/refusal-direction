"""A tiny, randomly-initialized model with the same architecture family as Qwen2.5
(RMSNorm, rotary, gated SiLU MLP, grouped-query attention). Tests run on CPU in seconds,
with no downloads -- so you can check your exercises locally in VS Code before Colab.
"""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

PAD = 0


@pytest.fixture(scope="session")
def tiny_cfg():
    return HookedTransformerConfig(
        n_layers=3,
        d_model=32,
        n_ctx=64,
        d_head=8,
        n_heads=4,
        n_key_value_heads=2,
        d_mlp=64,
        d_vocab=50,
        act_fn="silu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_dim=8,
        gated_mlp=True,
        final_rms=True,
        seed=0,
    )


@pytest.fixture
def model(tiny_cfg):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    m = HookedTransformer(tiny_cfg)
    m.eval()
    # Real Qwen has zero attention-output bias, but nonzero biases make the tests stricter.
    for block in m.blocks:
        block.attn.b_O.data.normal_()
        block.mlp.b_out.data.normal_()
    return m


@pytest.fixture
def batch():
    """Left-padded batch: (tokens, attention_mask), 6 prompts of varying length."""
    g = torch.Generator().manual_seed(1)
    lengths = [10, 7, 12, 9, 12, 5]
    seq = max(lengths)
    tokens = torch.full((len(lengths), seq), PAD)
    mask = torch.zeros_like(tokens)
    for i, n in enumerate(lengths):
        tokens[i, seq - n :] = torch.randint(1, 50, (n,), generator=g)
        mask[i, seq - n :] = 1
    return tokens, mask


@pytest.fixture
def direction():
    return torch.randn(32, generator=torch.Generator().manual_seed(2)) * 3  # deliberately not unit norm
