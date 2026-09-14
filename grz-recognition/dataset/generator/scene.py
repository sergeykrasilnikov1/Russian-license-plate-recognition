"""Procedural scenes: background, perspective placement, photometric conditions."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.geometry import PLATE_TYPES


def _asphalt(rng: np.random.Generator, width: int, height: int, night: bool) -> np.ndarray:
    base = 38 if night else 92
    img = np.full((height, width, 3), base, dtype=np.uint8)
    noise = rng.integers(-18, 19, size=(height, width, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # faint road grain
    if width > 80:
        for _ in range(int(3 + rng.integers(0, 4))):
            y = int(rng.integers(0, height))
            color = int(base + rng.integers(-10, 25))
            cv2.line(img, (0, y), (width, y), (color, color, color), 1)
    return img


def _bumper(rng: np.random.Generator, img: np.ndarray, night: bool) -> np.ndarray:
    """Paint a simple vehicle rear so the plate sits on something car-like."""
    h, w = img.shape[:2]
    hue = int(rng.integers(0, 180))
    sat = int(rng.integers(40, 180))
    val = int(rng.integers(40, 90) if night else rng.integers(80, 180))
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :] = (hue, sat, val)
    body = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    y0 = int(h * (0.35 + 0.15 * rng.random()))
    img[y0:, :] = cv2.addWeighted(img[y0:, :], 0.25, body[y0:, :], 0.75, 0)
    bumper_y = int(h * (0.62 + 0.12 * rng.random()))
    bumper_h = int(h * 0.12)
    bumper_color = tuple(int(c) for c in (20, 20, 20) if True)
    shade = int(30 + rng.integers(0, 40))
    cv2.rectangle(img, (0, bumper_y), (w, min(h, bumper_y + bumper_h)), (shade, shade, shade), -1)
    return img


def build_background(rng: np.random.Generator, width: int, height: int, night: bool) -> np.ndarray:
    img = _asphalt(rng, width, height, night)
    if rng.random() < 0.85:
        img = _bumper(rng, img, night)
    if night:
        img = np.clip(img.astype(np.int16) - int(rng.integers(20, 50)), 0, 255).astype(np.uint8)
    return img


def random_destination_quad(
    rng: np.random.Generator,
    image_wh: tuple[int, int],
    plate_wh: tuple[int, int],
    plate_type: str,
) -> np.ndarray:
    """Four clockwise corners (float32) of where the plate will sit."""
    iw, ih = image_wh
    aspect = PLATE_TYPES[plate_type].width_mm / PLATE_TYPES[plate_type].height_mm
    target_w = int(iw * float(rng.uniform(0.18, 0.42)))
    target_h = max(16, int(target_w / aspect))
    if target_h > ih * 0.45:
        target_h = int(ih * 0.45)
        target_w = int(target_h * aspect)

    margin = 8
    max_x = max(margin, iw - target_w - margin)
    max_y = max(margin, ih - target_h - margin)
    x0 = int(rng.integers(margin, max(margin + 1, max_x)))
    y0 = int(rng.integers(int(ih * 0.35), max(int(ih * 0.35) + 1, max_y)))

    # Perspective jitter: up to ~18% of each side
    jx, jy = target_w * 0.18, target_h * 0.18

    def j(axis_span: float) -> float:
        return float(rng.uniform(-axis_span, axis_span))

    tl = (x0 + j(jx), y0 + j(jy))
    tr = (x0 + target_w + j(jx), y0 + j(jy))
    br = (x0 + target_w + j(jx), y0 + target_h + j(jy))
    bl = (x0 + j(jx), y0 + target_h + j(jy))
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    # Keep inside the image
    quad[:, 0] = np.clip(quad[:, 0], 0, iw - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, ih - 1)
    return quad


def paste_plate(scene: np.ndarray, plate_bgra: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    ph, pw = plate_bgra.shape[:2]
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))
    warped = cv2.warpPerspective(plate_bgra, matrix, (scene.shape[1], scene.shape[0]), flags=cv2.INTER_LINEAR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    rgb = warped[:, :, :3].astype(np.float32)
    out = scene.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def quad_bbox(quad: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def apply_conditions(rng: np.random.Generator, image: np.ndarray, night: bool) -> tuple[np.ndarray, list[str]]:
    """Photometric augmentations. Geometry (bbox/quad) is already baked in."""
    conditions: list[str] = ["night" if night else "day"]
    out = image

    if rng.random() < 0.12:
        out = _rain(rng, out)
        conditions.append("rain")
    if rng.random() < 0.10:
        out = _snow(rng, out)
        conditions.append("snow")
    if rng.random() < 0.28:
        out = _dirt(rng, out)
        conditions.append("dirt")
    if rng.random() < 0.20:
        out = _glare(rng, out)
        conditions.append("glare")
    if rng.random() < 0.18:
        k = int(rng.choice([5, 7, 9]))
        out = cv2.GaussianBlur(out, (k, 1), 0)
        conditions.append("motion_blur")
    # Strong perspective was applied at placement time
    if rng.random() < 0.55:
        conditions.append("angle")
    return out, conditions


def _rain(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    n = int(280 * (w * h) / (1280 * 720))
    for _ in range(max(40, n)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(8, 22))
        cv2.line(overlay, (x, y), (x + int(rng.integers(-2, 3)), y + length), (210, 210, 210), 1)
    return cv2.addWeighted(img, 0.78, overlay, 0.22, 0)


def _snow(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    n = int(180 * (w * h) / (1280 * 720))
    for _ in range(max(30, n)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        cv2.circle(overlay, (x, y), r, (255, 255, 255), -1)
    return cv2.addWeighted(img, 0.82, overlay, 0.18, 0)


def _dirt(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    for _ in range(int(rng.integers(4, 12))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        axes = (int(rng.integers(8, 40)), int(rng.integers(4, 18)))
        color = tuple(int(c) for c in rng.integers(20, 70, size=3))
        cv2.ellipse(overlay, (x, y), axes, float(rng.integers(0, 180)), 0, 360, color, -1)
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)


def _glare(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    axes = (int(w * rng.uniform(0.08, 0.22)), int(h * rng.uniform(0.04, 0.12)))
    cv2.ellipse(overlay, (cx, cy), axes, float(rng.integers(0, 180)), 0, 360, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 12)
    return np.clip(img.astype(np.int16) + overlay.astype(np.int16) * 0.55, 0, 255).astype(np.uint8)
