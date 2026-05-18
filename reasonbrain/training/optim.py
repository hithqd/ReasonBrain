"""Optimizer and LR-schedule helpers."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: dict) -> AdamW:
    """Construct an AdamW optimizer following ``cfg`` (see ``configs/default.yaml``)."""
    return AdamW(
        params,
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"],
    )


# ---------------------------------------------------------------------------
def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> LambdaLR:
    """Cosine schedule with linear warmup."""
    warmup = cfg.get("warmup_steps", 0)
    total = cfg["max_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
def trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Iterate over parameters that have ``requires_grad`` set."""
    return [p for p in model.parameters() if p.requires_grad]
