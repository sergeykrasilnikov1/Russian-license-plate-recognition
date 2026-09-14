"""Classical-CV heuristic: is the plate mounted on a vehicle?"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VehicleFilterConfig:
    context_pad: float = 0.5
    min_context_score: float = 0.3
    # Plate width relative to image: typical bumper shot, not a full-screen poster.
    min_rel_width: float = 0.04
    max_rel_width: float = 0.55
    # Prefer plates in the lower portion of the frame.
    max_center_y: float = 0.92
    min_center_y: float = 0.15


def _clip_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> tuple[int, int, int, int]:
    return (
        int(max(0, min(w - 1, x0))),
        int(max(0, min(h - 1, y0))),
        int(max(0, min(w, x1))),
        int(max(0, min(h, y1))),
    )


def vehicle_context_score(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    cfg: VehicleFilterConfig | None = None,
) -> float:
    """Score in [0, 1]. Higher → more vehicle-like context around the plate.

    Signals (no neural net):
    - relative plate size in a bumper-camera range;
    - vertical position (plates are rarely at the very top of a car photo);
    - darker strip below the plate (bumper / underbody) vs bright uniform field
      (shop window / screen / billboard).
    """
    cfg = cfg or VehicleFilterConfig()
    h, w = image_bgr.shape[:2]
    if w < 8 or h < 8:
        return 0.0
    x0, y0, x1, y1 = bbox_xyxy
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    rel_w = bw / w
    cy = (y0 + y1) / 2 / h

    size_score = 1.0
    if rel_w < cfg.min_rel_width:
        size_score = rel_w / cfg.min_rel_width
    elif rel_w > cfg.max_rel_width:
        size_score = max(0.0, 1.0 - (rel_w - cfg.max_rel_width) / cfg.max_rel_width)

    pos_score = 1.0
    if cy < cfg.min_center_y:
        pos_score = cy / max(cfg.min_center_y, 1e-6)
    elif cy > cfg.max_center_y:
        pos_score = max(0.0, 1.0 - (cy - cfg.max_center_y) / (1.0 - cfg.max_center_y))

    # Context band below the plate.
    pad = cfg.context_pad
    cx0, cy0, cx1, cy1 = _clip_box(
        x0 - pad * bw,
        y1,
        x1 + pad * bw,
        y1 + max(bh * 1.2, h * 0.08),
        w,
        h,
    )
    band = image_bgr[cy0:cy1, cx0:cx1]
    if band.size == 0:
        below_score = 0.4
    else:
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        # Bumper/road: mid-dark + some texture. Screen/paper: bright + flat.
        darkness = 1.0 - min(1.0, mean / 220.0)
        texture = min(1.0, std / 40.0)
        below_score = 0.55 * darkness + 0.45 * texture

    score = 0.35 * size_score + 0.25 * pos_score + 0.40 * below_score
    return float(np.clip(score, 0.0, 1.0))


def is_on_vehicle(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    cfg: VehicleFilterConfig | None = None,
) -> tuple[bool, float]:
    cfg = cfg or VehicleFilterConfig()
    score = vehicle_context_score(image_bgr, bbox_xyxy, cfg)
    return score >= cfg.min_context_score, score
