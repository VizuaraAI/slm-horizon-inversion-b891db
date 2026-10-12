"""Config loading, seeding and small helpers shared by the entry points."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "model": "simulated:0.92:0.0",
    "family": "synthetic",
    "amplifier": "greedy",
    "horizons": [2, 4, 8, 12, 16],
    "per_horizon": 25,
    "limit": 200,
    "verifier": None,
    "device": "cpu",
    "out_dir": "results",
    "seed": 0,
}


def load_config(path: Optional[str], args=None) -> Dict[str, Any]:
    """Merge defaults, the YAML file, and any command line overrides."""
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    if args is not None:
        for key in DEFAULTS:
            value = getattr(args, key, None)
            if value is not None:
                cfg[key] = value
        horizons = getattr(args, "horizons", None)
        if isinstance(horizons, str):
            cfg["horizons"] = [int(part) for part in horizons.split(",") if part.strip()]
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_name(cfg: Dict[str, Any]) -> str:
    model = str(cfg["model"]).replace("/", "-").replace(":", "-")
    return f"{model}__{cfg['family']}__{cfg['amplifier']}__seed{cfg['seed']}"


def progress(items: Iterable, desc: str = ""):
    """tqdm when it is installed, a plain iterator otherwise."""
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        return items


def fmt_pct(value: float) -> str:
    return "n/a" if value != value else f"{100.0 * value:.1f}"
