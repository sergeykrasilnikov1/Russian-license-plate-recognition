"""Kaggle collector (official API, requires credentials).

Target: MADE CV 2021 Contest 02 — real and synthetic type1 plates plus
negative examples. Kaggle needs `~/.kaggle/kaggle.json` or
KAGGLE_USERNAME/KAGGLE_KEY; without them the source is skipped and the
reason is recorded in the Stage 2 report.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .base import Collector, RawItem

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class KaggleSource:
    slug: str
    kind: str  # "competition" | "dataset"
    license_raw: str
    plate_type: str = "type1"

    @property
    def landing_url(self) -> str:
        base = "competitions" if self.kind == "competition" else "datasets"
        return f"https://www.kaggle.com/{base}/{self.slug}"


DEFAULT_SOURCES: tuple[KaggleSource, ...] = (
    KaggleSource("made-cv-2021-contest-02-license-plate-recognition", "competition", "Competition rules: non-commercial research use"),
)


class KaggleCollector(Collector):
    name = "kaggle"
    access_note = (
        "Official Kaggle API. Requires ~/.kaggle/kaggle.json or "
        "KAGGLE_USERNAME/KAGGLE_KEY; skipped when credentials are missing."
    )

    def __init__(self, cache_dir, timeout: float = 120.0, sources: tuple[KaggleSource, ...] = DEFAULT_SOURCES) -> None:
        super().__init__(cache_dir, timeout)
        self.sources = sources

    @staticmethod
    def _has_credentials() -> bool:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        return (Path.home() / ".kaggle" / "kaggle.json").is_file()

    def check_available(self) -> tuple[bool, str]:
        if not self._has_credentials():
            return False, "no_credentials:~/.kaggle/kaggle.json"
        try:
            import kaggle  # noqa: F401
        except Exception as exc:
            return False, f"sdk_unavailable:{type(exc).__name__}"
        return True, "ok"

    def _download(self, source: KaggleSource) -> Path | None:
        target = self.cache_dir / source.slug
        target.mkdir(parents=True, exist_ok=True)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            if source.kind == "competition":
                api.competition_download_files(source.slug, path=str(target), quiet=True)
            else:
                api.dataset_download_files(source.slug, path=str(target), quiet=True, unzip=True)
        except Exception as exc:
            log.warning("kaggle download failed %s: %s", source.slug, exc)
            return None
        self._unzip_all(target)
        return target

    @staticmethod
    def _unzip_all(target: Path) -> None:
        import zipfile

        for archive in target.rglob("*.zip"):
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(archive.with_suffix(""))
            except Exception as exc:
                log.warning("kaggle unzip failed %s: %s", archive, exc)

    def collect(self, limit: int) -> Iterator[RawItem]:
        remaining = limit
        for source in self.sources:
            if remaining <= 0:
                return
            root = self._download(source)
            if root is None:
                continue
            for image_path in sorted(root.rglob("*")):
                if remaining <= 0:
                    return
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                yield RawItem(
                    source=f"{self.name}:{source.slug}",
                    source_url=source.landing_url,
                    license_raw=source.license_raw,
                    filename=image_path.name,
                    path=image_path,
                    query=source.slug,
                    suggested_type=source.plate_type,
                )
                remaining -= 1
