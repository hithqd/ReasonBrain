"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_FMT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


def get_logger(name: str = "reasonbrain", level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(handler)
    logger.setLevel(level or logging.INFO)
    logger.propagate = False
    return logger
