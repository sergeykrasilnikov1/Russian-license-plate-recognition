"""Automatic collection of real GRZ images from open, licensed sources.

CPU-only: no torch/paddle dependency. Heavy auto-labeling with the trained
detector happens in Stage 4; this package aggregates, filters and coarsely
classifies candidate images.
"""

from .licensing import LicenseInfo, LicensePolicy, normalize_license
from .meta_store import MetaRow, MetaStore

__all__ = [
    "LicenseInfo",
    "LicensePolicy",
    "normalize_license",
    "MetaRow",
    "MetaStore",
]
