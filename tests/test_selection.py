"""Exercise 4.   pytest tests/test_selection.py -v"""

import pandas as pd
import pytest

from refusal_direction.config import Config
from refusal_direction.selection import select_direction

N_LAYERS = 10


def make_scores():
    return pd.DataFrame(
        [
            # best bypass overall, but too late in the network (8 >= 0.8 * 10)
            dict(pos=-1, pos_idx=4, layer=8, bypass_score=-9.0, induce_score=5.0, kl_score=0.01),
            # great bypass but breaks harmless behaviour
            dict(pos=-1, pos_idx=4, layer=5, bypass_score=-8.0, induce_score=5.0, kl_score=0.5),
            # great bypass but doesn't induce refusal
            dict(pos=-2, pos_idx=3, layer=4, bypass_score=-7.0, induce_score=-1.0, kl_score=0.01),
            # the right answer
            dict(pos=-2, pos_idx=3, layer=6, bypass_score=-5.0, induce_score=3.0, kl_score=0.05),
            dict(pos=-5, pos_idx=0, layer=3, bypass_score=-2.0, induce_score=4.0, kl_score=0.02),
        ]
    )


def test_select_direction_applies_all_filters():
    best = select_direction(make_scores(), N_LAYERS, Config())
    assert (best.layer, best.pos) == (6, -2)


def test_select_direction_raises_when_nothing_survives():
    scores = make_scores()
    scores["kl_score"] = 1.0
    with pytest.raises(ValueError):
        select_direction(scores, N_LAYERS, Config())
