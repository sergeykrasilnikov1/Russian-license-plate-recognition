"""Shared utilities: plate mask, warp, metrics, geometry templates."""

from .plate_mask import (
    ALLOWED_LETTERS,
    REGION_PATTERN,
    looks_like_special_plate,
    normalize_plate,
    validate_plate,
)

__all__ = [
    "ALLOWED_LETTERS",
    "REGION_PATTERN",
    "looks_like_special_plate",
    "normalize_plate",
    "validate_plate",
]
