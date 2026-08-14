"""FIXED config loader."""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]


def load(path: str | Path = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)  # safe_load: never execute arbitrary tags from an untrusted file


def load_task(path: str | Path = "configs/task.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
