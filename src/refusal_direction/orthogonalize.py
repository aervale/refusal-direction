"""Exercise 5: weight orthogonalization -- the "jailbreak by editing weights" result (Section 4, Eq. 5).

Directional ablation needs hooks at inference time. But notice: if NOTHING ever writes
along r_hat into the residual stream, the residual stream never has an r_hat component,
so ablating it is a no-op. We can enforce that by editing every matrix that writes to the
residual stream:
    W_out <- W_out - r_hat r_hat^T W_out
(Equivalently, in TransformerLens's [.., d_model]-last convention: project each output row.)

Which components write to the residual stream in Qwen2.5? (Check model.named_parameters().)
  - the token embedding          model.W_E                     [d_vocab, d_model]
  - each attention output        model.blocks[l].attn.W_O      [n_heads, d_head, d_model]
                         + bias  model.blocks[l].attn.b_O      [d_model]
  - each MLP output              model.blocks[l].mlp.W_out     [d_mlp, d_model]
                         + bias  model.blocks[l].mlp.b_out     [d_model]
(No learned positional embedding: Qwen uses rotary, which lives inside attention.)

If you get this right, the edited model's logits should match the hook-ablated model's
logits to numerical precision -- that's exactly what tests/test_orthogonalize.py checks.
"""

import torch

from .interventions import project_out


@torch.no_grad()
def orthogonalize_weights(model, direction: torch.Tensor) -> None:
    """Exercise 5. Edit `model` IN PLACE so no component writes along `direction`.

    Hint: project_out(W, r) already does the right thing for any tensor whose last dim is
    d_model -- including a bias vector. Use param.data.copy_(...) to write in place.
    Afterwards, the model needs no hooks at all.
    """
    params = [model.W_E]
    for block in model.blocks:
        params += [block.attn.W_O, block.attn.b_O, block.mlp.W_out, block.mlp.b_out]
    for p in params:
        p.data.copy_(project_out(p.data, direction))
