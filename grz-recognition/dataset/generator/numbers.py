"""Valid GOST plate strings and a small set of real region codes."""

from __future__ import annotations

import numpy as np

from src.utils.plate_mask import ALLOWED_LETTERS, validate_plate

# Two-digit and three-digit codes that actually exist (sample, not exhaustive).
# Three-digit codes start with 1, 2 or 7 — the mask already enforces that.
REGION_CODES = (
    "01", "02", "05", "16", "23", "24", "25", "26", "27", "29",
    "31", "34", "35", "36", "38", "39", "42", "45", "47", "50",
    "52", "54", "55", "59", "61", "63", "66", "72", "74", "76",
    "77", "78", "86", "89", "92", "95",
    "102", "116", "123", "124", "125", "134", "136", "142", "150",
    "152", "154", "159", "161", "163", "174", "177", "178", "186",
    "190", "196", "197", "199", "702", "716", "750", "761", "763",
    "777", "797", "799",
)

_LETTERS = tuple(sorted(ALLOWED_LETTERS))


def random_region(rng: np.random.Generator) -> str:
    return str(REGION_CODES[int(rng.integers(0, len(REGION_CODES)))])


def random_plate_text(rng: np.random.Generator) -> str:
    """One letter, three digits, two letters, 2- or 3-digit region."""
    letter = _LETTERS[int(rng.integers(0, len(_LETTERS)))]
    digits = f"{int(rng.integers(0, 1000)):03d}"
    series = "".join(_LETTERS[int(rng.integers(0, len(_LETTERS)))] for _ in range(2))
    plate = f"{letter}{digits}{series}{random_region(rng)}"
    ok, normalized = validate_plate(plate, allow_hash=False)
    if not ok:
        raise RuntimeError(f"generator produced invalid plate {plate!r}: {normalized}")
    return normalized


def split_plate(plate: str) -> tuple[str, str, str, str]:
    """(first letter, 3 digits, 2 series letters, region)."""
    return plate[0], plate[1:4], plate[4:6], plate[6:]
