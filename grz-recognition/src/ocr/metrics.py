"""Character and plate-level OCR scores vs plate_mask ground truth."""

from __future__ import annotations

from src.utils.plate_mask import normalize_plate


def char_accuracy(pred: str, gt: str) -> float:
    """Position-wise match after normalize; length mismatch counts as errors."""
    p = normalize_plate(pred)
    g = normalize_plate(gt)
    n = max(len(p), len(g), 1)
    p = p.ljust(n, "\0")
    g = g.ljust(n, "\0")
    return sum(a == b for a, b in zip(p, g)) / n


def full_plate_match(pred: str, gt: str) -> bool:
    return normalize_plate(pred) == normalize_plate(gt)


def gt_is_readable(gt: str) -> bool:
    g = normalize_plate(gt)
    return bool(g) and "#" not in g


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (p, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)
