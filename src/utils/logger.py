"""Shared logger factory. Errors must be raised/logged, never silently swallowed
(project rule, Section 35)."""
from __future__ import annotations

import logging
import sys

from src.utils.config import get_config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    try:
        level_name = get_config().get("logging", {}).get("level", "INFO")
    except Exception:
        level_name = "INFO"

    level = getattr(logging, str(level_name).upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger