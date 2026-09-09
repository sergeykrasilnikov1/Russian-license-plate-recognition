"""GOST R 50577-2018 geometry templates (parametrized plate types)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PlateGeometry:
    """Physical plate size in mm and layout hints for synthetic rendering."""

    name: str
    width_mm: float
    height_mm: float
    rows: int  # 1 = single line, 2 = two-line (type1a)
    background: str  # "white" | "yellow"
    foreground: str  # "black"


# Extensible registry — add new types/countries here without touching pipeline core
PLATE_TYPES: Dict[str, PlateGeometry] = {
    "type1": PlateGeometry("type1", 520.0, 112.0, 1, "white", "black"),
    "type1a": PlateGeometry("type1a", 290.0, 170.0, 2, "white", "black"),
    "type1b": PlateGeometry("type1b", 520.0, 112.0, 1, "yellow", "black"),
    "other": PlateGeometry("other", 520.0, 112.0, 1, "white", "black"),
}

CLASS_ID_TO_NAME = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}


def aspect_ratio(plate_type: str) -> float:
    g = PLATE_TYPES[plate_type]
    return g.width_mm / g.height_mm


def warp_size(plate_type: str, scale: float = 0.8) -> Tuple[int, int]:
    """Pixel size for perspective-warp output (~mm * scale)."""
    g = PLATE_TYPES[plate_type]
    return int(g.width_mm * scale), int(g.height_mm * scale)
