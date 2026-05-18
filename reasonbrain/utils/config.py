"""Configuration loader for ReasonBrain (YAML-only, dict-output)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return a plain dict.

    We deliberately *do not* use OmegaConf / Hydra here so the config object is
    trivially serialisable / picklable across worker processes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must parse to a dict, got {type(cfg)}.")
    return cfg
