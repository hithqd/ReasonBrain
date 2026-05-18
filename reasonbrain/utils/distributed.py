"""Distributed-training helpers (thin wrappers around accelerate)."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def is_main_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(os.environ.get("RANK", "0")) == 0
    return torch.distributed.get_rank() == 0


@contextmanager
def main_process_first():
    """Yield first on the main rank; other ranks wait on a barrier."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        yield
        return
    if torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
    yield
    if torch.distributed.get_rank() == 0:
        torch.distributed.barrier()
