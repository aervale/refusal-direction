"""Datasets. We reuse the exact splits from the paper's repo so numbers are comparable.

harmful_*  : AdvBench + MaliciousInstruct + TDC2023 + HarmBench (train/val); JailbreakBench (test)
harmless_* : Alpaca instructions
"""

import json
import random
import urllib.request
from pathlib import Path

SPLITS_URL = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/{name}.json"
SPLIT_NAMES = [f"{kind}_{split}" for kind in ("harmful", "harmless") for split in ("train", "val", "test")]


def _download(name: str, data_dir: Path) -> Path:
    path = data_dir / f"{name}.json"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SPLITS_URL.format(name=name), path)
    return path


def load_split(name: str, data_dir: str = "data") -> list[str]:
    """Return the list of instruction strings for one split, e.g. 'harmful_train'."""
    assert name in SPLIT_NAMES, f"unknown split {name!r}; choose from {SPLIT_NAMES}"
    with open(_download(name, Path(data_dir))) as f:
        return [row["instruction"] for row in json.load(f)]


def load_datasets(cfg) -> dict[str, list[str]]:
    """Sample train/val/test sets of the sizes in cfg (seeded, so reruns match)."""
    rng = random.Random(cfg.seed)
    sizes = {"train": cfg.n_train, "val": cfg.n_val}
    out = {}
    for kind in ("harmful", "harmless"):
        for split, n in sizes.items():
            data = load_split(f"{kind}_{split}", cfg.data_dir)
            # Over-sample 2x so there is slack after filtering (see model.filter_by_refusal);
            # the filter trims back down to n.
            k = min(len(data), n * 2)
            out[f"{kind}_{split}"] = rng.sample(data, k)
    out["harmful_test"] = load_split("harmful_test", cfg.data_dir)[: cfg.n_test]
    out["harmless_test"] = rng.sample(load_split("harmless_test", cfg.data_dir), cfg.n_test)
    return out
