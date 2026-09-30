"""Perspective warp of a plate quad into a GOST-sized rectangle.

Type1a two-line plates support two layouts:
- flatten: warp then stitch top/bottom halves into one row (CRNN-friendly)
- stacked: warp to a 2-row rectangle (keeps original layout)
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from src.utils.geometry import PLATE_TYPES, warp_size

Type1aMode = Literal["flatten", "stacked"]
DEFAULT_TYPE1A_MODE: Type1aMode = "flatten"


class DegenerateQuadError(ValueError):
    """Quad has ~zero area or collinear corners."""


def parse_quad(raw: str | np.ndarray | list) -> np.ndarray:
    """Parse `x0,y0,...,x3,y3` or an (4,2) array."""
    if isinstance(raw, np.ndarray):
        pts = np.asarray(raw, dtype=np.float32).reshape(4, 2)
        return pts
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=np.float32)
        return arr.reshape(4, 2)
    parts = [float(x) for x in str(raw).replace(" ", "").split(",") if x]
    if len(parts) != 8:
        raise ValueError(f"quad must have 8 numbers, got {len(parts)}")
    return np.asarray(parts, dtype=np.float32).reshape(4, 2)


def bbox_to_quad(xywh: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = xywh
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def xyxy_to_quad(xyxy: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return corners TL, TR, BR, BL.

    Sorts by angle around the centroid (counter-clockwise), then rotates the
    ring so the first point is the visual top-left (min x+y). Clockwise input
    is reversed to CCW.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        raise DegenerateQuadError("non-finite quad")

    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(angles)]
    area = _polygon_area(ordered)
    if area < 4.0 or not cv2.isContourConvex(ordered):
        raise DegenerateQuadError(f"degenerate or non-convex quad area={area:.3f}")

    # Signed area of the sorted ring: positive = CCW in image coords? y-down
    # makes CCW appear clockwise; we only need a consistent winding.
    signed = 0.0
    for i in range(4):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % 4]
        signed += (x2 - x1) * (y2 + y1)
    if signed > 0:
        ordered = ordered[::-1]

    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def _dst_rect(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def warp_quad(
    image_bgr: np.ndarray,
    quad: np.ndarray | str,
    out_wh: tuple[int, int],
) -> np.ndarray:
    src = order_quad(parse_quad(quad))
    w, h = int(out_wh[0]), int(out_wh[1])
    if w < 8 or h < 8:
        raise ValueError(f"warp size too small: {out_wh}")
    matrix = cv2.getPerspectiveTransform(src, _dst_rect(w, h))
    return cv2.warpPerspective(image_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def type1a_row_quads(ordered_tl_tr_br_bl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an ordered plate quad into top-row and bottom-row sub-quads."""
    tl, tr, br, bl = np.asarray(ordered_tl_tr_br_bl, dtype=np.float32)
    # GOST type1a: upper line ~ top 48%, lower line ~ bottom 48%, thin gap.
    top = np.stack(
        [_lerp(tl, bl, 0.00), _lerp(tr, br, 0.00), _lerp(tr, br, 0.48), _lerp(tl, bl, 0.48)],
        axis=0,
    )
    bot = np.stack(
        [_lerp(tl, bl, 0.52), _lerp(tr, br, 0.52), _lerp(tr, br, 1.00), _lerp(tl, bl, 1.00)],
        axis=0,
    )
    return top.astype(np.float32), bot.astype(np.float32)


def _flatten_two_line_from_subquads(image_bgr: np.ndarray, ordered: np.ndarray, out_w: int) -> np.ndarray:
    """Warp each type1a row separately, then stitch left|right (CRNN one-line)."""
    top_q, bot_q = type1a_row_quads(ordered)
    row_h = max(32, int(out_w * 0.22))
    top = warp_quad(image_bgr, top_q, (out_w, row_h))
    bot = warp_quad(image_bgr, bot_q, (out_w, row_h))
    return np.concatenate([top, bot], axis=1)


def warp_plate(
    image_bgr: np.ndarray,
    quad: np.ndarray | str,
    plate_type: str,
    *,
    type1a_mode: Type1aMode = DEFAULT_TYPE1A_MODE,
    scale: float = 0.8,
) -> np.ndarray:
    """Warp a plate into the rectangle implied by PLATE_TYPES."""
    if plate_type not in PLATE_TYPES:
        plate_type = "type1"
    w, h = warp_size(plate_type, scale=scale)
    geom = PLATE_TYPES[plate_type]
    if geom.rows == 2 and type1a_mode == "flatten":
        ordered = order_quad(parse_quad(quad))
        return _flatten_two_line_from_subquads(image_bgr, ordered, w)
    return warp_quad(image_bgr, quad, (w, h))


def horizontal_ink_score(gray_or_bgr: np.ndarray) -> float:
    """Higher when ink forms a compact horizontal band (good for single-line OCR)."""
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    # Dark glyphs on light plate
    ink = 255 - gray
    col = ink.mean(axis=0)
    row = ink.mean(axis=1)
    if row.sum() <= 1e-6:
        return 0.0
    # Peakiness of the vertical profile: one (flatten) vs two (stacked) modes
    row_n = row / (row.max() + 1e-6)
    # Gini-like concentration
    conc = float((row_n**2).sum() / (row_n.sum() + 1e-6))
    col_energy = float(col.std())
    return conc * math.log1p(col_energy)


def choose_type1a_mode(samples: list[tuple[np.ndarray, np.ndarray | str]]) -> Type1aMode:
    """Pick flatten vs stacked on synthetic crops by ink-band score."""
    if not samples:
        return DEFAULT_TYPE1A_MODE
    scores = {"flatten": [], "stacked": []}
    for image, quad in samples:
        for mode in ("flatten", "stacked"):
            warped = warp_plate(image, quad, "type1a", type1a_mode=mode)
            scores[mode].append(horizontal_ink_score(warped))
    mean_f = float(np.mean(scores["flatten"]))
    mean_s = float(np.mean(scores["stacked"]))
    return "flatten" if mean_f >= mean_s else "stacked"
