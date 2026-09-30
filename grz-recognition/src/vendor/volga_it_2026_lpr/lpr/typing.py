"""Plate-type decision that combines detector class, plate geometry, colour and OCR grammar.

The detector alone confuses white foreign plates with type1 (they differ only in layout), so the final
type is decided here:
  * two-row geometry (quad aspect < 2.4) + rows that read as  L DDD / LL RR  -> type1a
  * yellow field                                                            -> type1b (if text fits one of the 1B masks) else other
  * single row white/other                                                  -> type1 iff text matches the standard mask, else other
The detector class is used as a prior: it wins when the OCR evidence is weak.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .plates import RE_STD, RE_TAXI, UNK

RE_TOP_1A = re.compile(r"^[ABEKMHOPCTYX#][0-9#]{3}$")
RE_BOT_1A = re.compile(r"^[ABEKMHOPCTYX#]{2}([0-9#]{2}|[127#][0-9#]{2})$")


def yellowness(rect_bgr: np.ndarray) -> float:
    """Fraction of plate pixels whose hue/saturation say 'yellow' (works for GOST 1B yellow and taxi yellow)."""
    if rect_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 12) & (h <= 40) & (s >= 70) & (v >= 90)
    return float(mask.mean())


def whiteness(rect_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    return float(((s < 60) & (v > 120)).mean())


def is_bluish(rect_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 95) & (h <= 130) & (s >= 90) & (v >= 60)).mean()) > 0.45


def is_reddish(rect_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h <= 8) | (h >= 170)) & (s >= 110) & (v >= 60)).mean()) > 0.45


def is_dark(rect_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    return float(rect_bgr.mean()) < 60 and float((hsv[..., 2] > 150).mean()) < 0.2


def std_score(text: str) -> float:
    """How well a string fits the standard mask: 1 = perfect, partial credit for '#' positions."""
    if not text or not RE_STD.match(text):
        return 0.0
    return 1.0 - 0.5 * text.count(UNK) / len(text)


def taxi_score(text: str) -> float:
    if not text or not RE_TAXI.match(text):
        return 0.0
    return 1.0 - 0.5 * text.count(UNK) / len(text)


def decide_type(det_cls: str, det_score: float, quad: np.ndarray, rect_1row: np.ndarray, text: str, ocr_score: float,
                rows_text: Optional[Tuple[str, str]] = None) -> Tuple[str, float]:
    """Return (plate_type, type_confidence)."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    aspect = w / max(h, 1e-3)
    yel = yellowness(rect_1row)
    fit1 = max(std_score(text), taxi_score(text))
    two_row = aspect < 2.4 and fit1 < 0.8          # a confident single-row reading overrides the geometry
    # --- two-row plates -------------------------------------------------------------------------
    if two_row:
        if rows_text and RE_TOP_1A.match(rows_text[0]) and RE_BOT_1A.match(rows_text[1]) and yel < 0.25:
            return "type1a", max(0.6, ocr_score)
        if det_cls == "type1a" and det_score >= 0.5 and yel < 0.25:
            return "type1a", det_score * 0.8
        return "other", max(0.5, det_score)
    # --- yellow plates (checked before the dark test: a night-time yellow plate is dark too) ----
    if yel >= 0.25:
        fit = max(std_score(text), taxi_score(text))
        if fit > 0 or det_cls == "type1b":
            return "type1b", max(0.55, fit if fit > 0 else det_score * 0.8)
        return "other", max(0.5, det_score)
    # --- coloured plates (blue police, red diplomatic, black military) --------------------------
    if is_bluish(rect_1row) or is_reddish(rect_1row) or (is_dark(rect_1row) and fit1 < 0.8):
        return "other", max(0.6, det_score)
    # --- white single row -----------------------------------------------------------------------
    fit = std_score(text)
    if det_cls == "other":
        # the detector says "foreign/other": override only on a clean, fully readable Russian mask
        if fit >= 1.0 and ocr_score >= 0.6:
            return "type1", 0.85 * max(fit, ocr_score)
        return "other", max(0.5, det_score)
    if fit >= 0.7 and ocr_score >= 0.5:
        return "type1", max(fit, ocr_score)
    if det_cls in ("type1", "type1b") and fit > 0:
        return "type1", fit * 0.9
    if det_cls == "type1" and det_score >= 0.6 and text.count(UNK) >= len(text) // 2:
        return "type1", det_score * 0.7        # unreadable but the detector is sure it is a Russian plate
    if det_cls == "type1b" and (taxi_score(text) > 0 or det_score >= 0.6):
        return "type1b", max(taxi_score(text), det_score * 0.7)
    if det_cls == "type1a" and det_score >= 0.6:
        return "type1", det_score * 0.6
    return "other", max(0.5, det_score if det_cls == "other" else 0.5)
