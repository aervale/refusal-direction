"""End-to-end smoke test of the provided plumbing on the tiny model.
Passes once Exercises 1-4 are done (it calls them). Not a correctness test of the science --
just "do all the pieces fit together".
"""

import torch

from refusal_direction.config import Config
from refusal_direction.directions import get_candidate_directions
from refusal_direction.model import generate, last_token_logits
from refusal_direction.selection import score_candidates, select_direction


class FakeTokenizer:
    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def convert_tokens_to_ids(self, tok):
        return 2

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(i) for i in row.tolist() if i > 2) for row in ids]


def test_selection_pipeline_runs(model, batch):
    tokens, mask = batch
    harmful, harmless = (tokens[:3], mask[:3]), (tokens[3:], mask[3:])
    positions = [-2, -1]
    cands = get_candidate_directions(model, harmful, harmless, positions, batch_size=2)
    scores = score_candidates(model, cands, positions, harmful, harmless, refusal_toks=[5, 6], batch_size=2)
    assert len(scores) == len(positions) * model.cfg.n_layers
    assert set(scores.columns) >= {"pos", "layer", "bypass_score", "induce_score", "kl_score"}
    # Ablating r^(l)_i should usually lower the toy "refusal score" relative to baseline on the
    # harmful set; on a random model this isn't guaranteed, so we only check it runs and picks.
    cfg = Config(kl_threshold=float("inf"), induce_threshold=-float("inf"), prune_layer_fraction=1.0)
    best = select_direction(scores, model.cfg.n_layers, cfg)
    assert best.bypass_score == scores.bypass_score.min()


def test_generate_and_logits(model, batch):
    model.tokenizer = FakeTokenizer()
    tokens, mask = batch
    logits = last_token_logits(model, tokens, mask, batch_size=4)
    assert logits.shape == (tokens.shape[0], model.cfg.d_vocab)
    outs = generate(model, tokens, mask, max_new_tokens=5, batch_size=4, show_progress=False)
    assert len(outs) == tokens.shape[0] and all(isinstance(o, str) for o in outs)
    # left padding must not change a prompt's prediction: compare row 1 alone vs in the batch
    real = mask[1].bool()
    solo = model(tokens[1][real][None])[:, -1]
    assert torch.allclose(solo[0], logits[1], atol=1e-4), "left-padding leaks into predictions"
