"""
Central configuration loader.

Every module in this project reads settings through `load_config()` /
`get_config()` rather than hard-coding values, per project rule
(Sections 22, 32, 40).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_lock = threading.Lock()
_cached_config: dict[str, Any] | None = None
_cached_path: Path | None = None


class ConfigError(Exception):
    """Raised when configuration is missing or malformed."""


def load_config(path: str | os.PathLike | None = None, force_reload: bool = False) -> dict:
    """Load config.yaml (or a given path) once and cache it."""
    global _cached_config, _cached_path

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    with _lock:
        if _cached_config is not None and not force_reload and config_path == _cached_path:
            return _cached_config

        if not config_path.exists():
            raise ConfigError(
                f"Configuration file not found at '{config_path}'. "
                "Copy config.yaml to the project root or pass an explicit path."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ConfigError(f"Configuration file '{config_path}' did not parse to a mapping.")

        _cached_config = data
        _cached_path = config_path
        return _cached_config


def get_config() -> dict:
    """Return the cached config, loading the default file if needed."""
    if _cached_config is None:
        return load_config()
    return _cached_config


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-declared relative path against the project root."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def project_root() -> Path:
    return _PROJECT_ROOT
