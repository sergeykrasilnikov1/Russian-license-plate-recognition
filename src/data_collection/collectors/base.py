"""Collector contract shared by every data source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

USER_AGENT = "grz-dataset-collector/0.1 (AIS Gorod semifinal; non-commercial research)"


class CollectorUnavailable(RuntimeError):
    """Raised when a source cannot be used (no SDK, no credentials, no access)."""


class RateLimited(RuntimeError):
    """The source throttled us. Never worked around, only waited out or reported."""


def request_with_retry(session, url: str, *, params=None, timeout: float = 30.0, attempts: int = 4, base_delay: float = 2.0):
    """GET with exponential backoff that honours Retry-After.

    A persistent 401/403/429 is surfaced as RateLimited so the collector can
    stop and report it, rather than hammering the service or trying to look
    like a browser.
    """
    import time

    import requests

    last_status = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
            continue

        if response.status_code == 200:
            return response

        last_status = response.status_code
        if response.status_code in (401, 403, 429):
            retry_after = response.headers.get("Retry-After")
            if attempt == attempts - 1:
                raise RateLimited(f"http_{response.status_code}")
            wait = float(retry_after) if (retry_after or "").isdigit() else base_delay * (2**attempt)
            time.sleep(min(wait, 30.0))
            continue

        response.raise_for_status()

    raise RateLimited(f"http_{last_status}")


@dataclass(frozen=True)
class YoloBox:
    """Normalized bbox supplied by the source dataset itself."""

    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_pixels(self, image_w: int, image_h: int) -> tuple[float, float, float, float]:
        w = self.w * image_w
        h = self.h * image_h
        return self.xc * image_w - w / 2, self.yc * image_h - h / 2, w, h


@dataclass
class RawItem:
    """One downloaded candidate image plus its provenance."""

    source: str
    source_url: str
    license_raw: str
    filename: str
    path: Path | None = None
    data: bytes | None = None
    query: str = ""
    suggested_type: str | None = None
    #: True when the source dataset documents its plate type, so the coarse
    #: geometric classifier must not override it.
    type_is_authoritative: bool = False
    boxes: list[YoloBox] = field(default_factory=list)
    attribution: str = ""
    #: Plate text supplied by the source (e.g. encoded in the filename),
    #: always re-validated against the GOST mask before use.
    plate_text: str | None = None
    plate_text_source: str = ""

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is not None:
            return self.path.read_bytes()
        raise ValueError(f"RawItem {self.filename!r} carries no image payload")


class Collector(ABC):
    """Base class for a single open data source."""

    name: str = "base"
    #: Human-readable note on how the source is accessed, used in the datasheet
    access_note: str = ""

    def __init__(self, cache_dir: Path, timeout: float = 30.0) -> None:
        self.cache_dir = Path(cache_dir) / self.name
        self.timeout = timeout

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        """Return (usable, reason) without downloading anything."""

    @abstractmethod
    def collect(self, limit: int) -> Iterator[RawItem]:
        """Yield at most `limit` candidate images."""

    def _session(self):
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        return session
