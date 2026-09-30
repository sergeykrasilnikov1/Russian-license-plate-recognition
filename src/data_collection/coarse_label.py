"""Coarse plate-type classification without a neural network.

Stage 2 has no fine-tuned detector yet (that arrives in Stage 4 on the GPU
server), so plate type is guessed from two cheap physical cues that follow
directly from GOST geometry: background color and width/height ratio.

Thresholds are derived from the PLATE_TYPES registry instead of being
hardcoded, so registering a new plate type is enough to support it here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..utils.geometry import PLATE_TYPES


@dataclass(frozen=True)
class CoarseGuess:
    plate_type: str
    confidence: float
    aspect: float
    yellow_ratio: float
    background: str

    def as_dict(self) -> dict[str, str]:
        return {
            "plate_type": self.plate_type,
            "confidence": f"{self.confidence:.3f}",
            "aspect": f"{self.aspect:.3f}",
            "yellow_ratio": f"{self.yellow_ratio:.3f}",
            "background": self.background,
        }


# OpenCV HSV: hue 0..180
YELLOW_LOWER = np.array([15, 70, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)


def yellow_ratio(crop_bgr: np.ndarray) -> float:
    """Share of pixels inside the yellow HSV range (type1b background)."""
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    return float(mask.mean() / 255.0)


class CoarseTypeClassifier:
    """Pick the registered plate type whose geometry best fits the crop."""

    def __init__(self, yellow_threshold: float = 0.30, candidates: tuple[str, ...] = ("type1", "type1a", "type1b")) -> None:
        self.yellow_threshold = yellow_threshold
        self.candidates = candidates

    def classify(self, crop_bgr: np.ndarray, aspect: float | None = None) -> CoarseGuess:
        h, w = crop_bgr.shape[:2]
        if aspect is None:
            aspect = (w / h) if h else 0.0

        ratio = yellow_ratio(crop_bgr)
        background = "yellow" if ratio >= self.yellow_threshold else "white"

        pool = [t for t in self.candidates if PLATE_TYPES[t].background == background]
        if not pool:
            pool = list(self.candidates)

        def geom_aspect(name: str) -> float:
            g = PLATE_TYPES[name]
            return g.width_mm / g.height_mm

        scored = sorted(pool, key=lambda t: abs(math.log((aspect or 1e-6) / geom_aspect(t))))
        best = scored[0]
        best_err = abs(math.log((aspect or 1e-6) / geom_aspect(best)))

        if len(scored) > 1:
            runner_err = abs(math.log((aspect or 1e-6) / geom_aspect(scored[1])))
            margin = runner_err - best_err
            confidence = 1.0 - math.exp(-2.0 * margin)
        else:
            confidence = math.exp(-best_err)

        color_bonus = 0.15 if background == "yellow" and ratio >= 0.5 else 0.0
        confidence = float(min(0.95, max(0.05, confidence + color_bonus)))

        return CoarseGuess(
            plate_type=best,
            confidence=confidence,
            aspect=float(aspect),
            yellow_ratio=ratio,
            background=background,
        )
