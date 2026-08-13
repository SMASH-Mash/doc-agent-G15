"""Stage 9 — confidence calibration (calibrated-confidence NFR)"""

from __future__ import annotations

from ..contracts import *  # noqa


def temperature_scale(logits: list[float], labels: list[int]) -> float:
    """Fit temperature on val; return scaler. IMPLEMENT."""
    raise NotImplementedError("Calibration: temperature scaling")


def ece(confidences: list[float], correct: list[bool]) -> float:
    raise NotImplementedError("Calibration: ECE")
