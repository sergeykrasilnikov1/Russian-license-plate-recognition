"""License normalization and admission policy for collected images.

The submitted dataset is published under CC BY 4.0 and every image must allow
non-commercial reuse. Crops and augmentations are derivative works, so
NoDerivatives licenses are rejected outright, and anything unrecognized is
rejected rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class LicenseInfo:
    raw: str
    spdx: str
    allows_noncommercial: bool
    allows_derivatives: bool
    share_alike: bool
    requires_attribution: bool

    @property
    def is_public_domain(self) -> bool:
        return self.spdx in {"CC0-1.0", "PD"}


class LicensePolicy(str, Enum):
    """Which licenses may enter the dataset."""

    # Only licenses that can be redistributed under CC BY 4.0 without friction
    STRICT = "strict"
    # Everything that permits non-commercial reuse and derivative works
    NONCOMMERCIAL = "noncommercial"


_STRICT_ALLOWED = {"CC0-1.0", "PD", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-2.0", "Apache-2.0", "MIT"}

_UNKNOWN = LicenseInfo(
    raw="",
    spdx="UNKNOWN",
    allows_noncommercial=False,
    allows_derivatives=False,
    share_alike=False,
    requires_attribution=True,
)


def normalize_license(raw: str | None) -> LicenseInfo:
    """Map a free-form license string (Commons/Openverse/HF) to LicenseInfo."""
    if not raw:
        return _UNKNOWN
    text = raw.strip()
    key = re.sub(r"[\s_]+", "-", text.lower())

    if any(t in key for t in ("cc0", "zero", "publicdomain", "public-domain", "pdm")):
        return LicenseInfo(text, "CC0-1.0", True, True, False, False)
    if key in {"pd", "pd-self", "pd-old", "pd-us"} or key.startswith("pd-"):
        return LicenseInfo(text, "PD", True, True, False, False)
    if key in {"apache-2.0", "apache-license-2.0", "apache2", "apache-2"}:
        return LicenseInfo(text, "Apache-2.0", True, True, False, True)
    if key == "mit":
        return LicenseInfo(text, "MIT", True, True, False, True)

    cc = re.search(r"cc-?by([a-z\-]*?)-?(\d\.\d)?$", key)
    if cc or key.startswith("cc-by") or key.startswith("by-") or key == "by":
        version = (cc.group(2) if cc and cc.group(2) else "4.0")
        nc = "nc" in key.split("-")
        nd = "nd" in key.split("-")
        sa = "sa" in key.split("-")
        parts = ["CC-BY"]
        if nc:
            parts.append("NC")
        if nd:
            parts.append("ND")
        if sa:
            parts.append("SA")
        spdx = "-".join(parts) + f"-{version}"
        return LicenseInfo(text, spdx, True, not nd, sa, True)

    return LicenseInfo(text, "UNKNOWN", False, False, False, True)


def is_acceptable(info: LicenseInfo, policy: LicensePolicy) -> tuple[bool, str]:
    """Return (accepted, reason)."""
    if info.spdx == "UNKNOWN":
        return False, "unknown_license"
    if not info.allows_derivatives:
        return False, "no_derivatives"
    if not info.allows_noncommercial:
        return False, "commercial_only"
    if policy is LicensePolicy.STRICT and info.spdx not in _STRICT_ALLOWED:
        return False, f"not_strict_compatible:{info.spdx}"
    return True, "ok"
