"""Save/load results between notebooks (so notebook 02 doesn't redo notebook 01's work). Provided."""

import json
from pathlib import Path

import torch


def _dir(cfg) -> Path:
    d = Path(cfg.artifact_dir) / cfg.model_name.split("/")[-1]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_direction(cfg, direction, meta: dict) -> Path:
    d = _dir(cfg)
    torch.save(direction.detach().float().cpu(), d / "direction.pt")
    (d / "direction_meta.json").write_text(json.dumps(meta, indent=2))
    return d


def load_direction(cfg):
    d = _dir(cfg)
    return torch.load(d / "direction.pt"), json.loads((d / "direction_meta.json").read_text())


def save_json(cfg, name: str, obj) -> Path:
    path = _dir(cfg) / name
    path.write_text(json.dumps(obj, indent=2))
    return path


def path(cfg, name: str) -> Path:
    return _dir(cfg) / name
