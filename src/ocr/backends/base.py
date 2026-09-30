"""OCR backend protocol: recognize(crop) -> (text, mean_or_min_conf, per-char conf)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class BackendUnavailable(RuntimeError):
    """Optional OCR engine is not installed or has no weights."""


@runtime_checkable
class OcrBackend(Protocol):
    name: str

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """Return raw text and an aggregate OCR confidence in [0, 1]."""
        ...

    def recognize_detailed(self, crop: np.ndarray) -> tuple[str, list[float]]:
        """Return raw text and per-character confidences (may be a single value)."""
        ...
