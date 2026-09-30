"""Russian GRZ plate number mask validation (GOST types 1 / 1A / 1B)."""

from __future__ import annotations

import re
from typing import Tuple

# Letters that share glyph shape in Cyrillic and Latin
ALLOWED_LETTERS = frozenset("ABEKMHOPCTYX")
ALLOWED_DIGITS = frozenset("0123456789")

# Region: 2 digits, or 3 digits starting with 1, 2, or 7
REGION_PATTERN = re.compile(r"^(\d{2}|[127]\d{2})$")

# Full plate with optional # for unreadable positions
_PLATE_RE = re.compile(
    r"^([ABEKMHOPCTYX#])([0-9#]{3})([ABEKMHOPCTYX#]{2})([0-9#]{2}|[127#][0-9#]{2})$"
)

# Cyrillic → Latin lookalikes used in scraped/OCR text
_CYR_TO_LAT = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)


def normalize_plate(raw: str) -> str:
    """Uppercase, strip spaces/dashes, map Cyrillic lookalikes to Latin."""
    s = raw.upper().replace(" ", "").replace("-", "").translate(_CYR_TO_LAT)
    return s


def validate_plate(plate: str, allow_hash: bool = True) -> Tuple[bool, str]:
    """
    Validate plate against GOST mask.

    Returns (ok, normalized_or_reason).
    Unreadable positions may be '#" when allow_hash=True.
    """
    plate = normalize_plate(plate)
    if not allow_hash and "#" in plate:
        return False, "hash_not_allowed"
    if not _PLATE_RE.match(plate):
        return False, "mask_mismatch"
    # Extra check for concrete (non-#) region when fully numeric
    region = plate[6:]
    if "#" not in region and not REGION_PATTERN.match(region):
        return False, "invalid_region"
    return True, plate


# Plates outside the three target types: diplomatic (002CD178), military and
# similar series start with digits instead of a letter. They must be reported
# as `other`, never as a target type with an invented number.
_SPECIAL_RE = re.compile(r"^\d{3}[A-Z]{1,2}\d{2,3}$")


def looks_like_special_plate(text: str) -> bool:
    """True for non-target Russian series (diplomatic, military, transit)."""
    return bool(_SPECIAL_RE.match(normalize_plate(text)))


def mask_invalid_chars(plate: str) -> str:
    """Replace characters outside alphabet with '#' keeping length."""
    plate = normalize_plate(plate)
    out = []
    for i, ch in enumerate(plate):
        if i == 0 or i in (4, 5):
            out.append(ch if ch in ALLOWED_LETTERS or ch == "#" else "#")
        elif i < 6:
            out.append(ch if ch in ALLOWED_DIGITS or ch == "#" else "#")
        else:
            out.append(ch if ch in ALLOWED_DIGITS or ch == "#" else "#")
    return "".join(out)
