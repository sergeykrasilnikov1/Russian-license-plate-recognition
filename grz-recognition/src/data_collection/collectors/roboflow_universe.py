"""Roboflow Universe collector (official SDK + Universe REST API).

Projects are verified through `GET /universe/search` and `GET /{ws}/{project}`,
so the license and the latest version number come from the API rather than
from hardcoded guesses. Requires `ROBOFLOW_API_KEY`; without it the collector
reports itself unavailable instead of falling back to scraping.

Several of these datasets encode the plate number in the image filename. That
is source-provided metadata, not a model prediction, so it is used as plate
text — but only after it passes the GOST mask validator, and only when the
image contains exactly one annotated plate.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ...utils.plate_mask import looks_like_special_plate, normalize_plate, validate_plate
from .base import Collector, RawItem, YoloBox

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
API_BASE = "https://api.roboflow.com"

# Roboflow appends "_<ext>.rf.<hash>" to every exported filename
_RF_SUFFIX_RE = re.compile(r"_(png|jpg|jpeg|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)


@dataclass(frozen=True)
class RoboflowProject:
    workspace: str
    project: str
    #: None → resolve the latest version through the API
    version: int | None = None
    plate_type: str = "type1"
    homogeneous_type: bool = True
    #: Filenames carry the plate number (verified by inspecting the export)
    filename_has_plate: bool = False

    @property
    def landing_url(self) -> str:
        return f"https://universe.roboflow.com/{self.workspace}/{self.project}"


# Verified via the Universe API: Russian-plate object detection projects with
# a redistributable license. None of them distinguishes type1a/type1b, which
# matches the premise of the task.
DEFAULT_PROJECTS: tuple[RoboflowProject, ...] = (
    RoboflowProject("tatmantech", "russian-license-plates-ec7zg", filename_has_plate=True),
)


class RoboflowCollector(Collector):
    name = "roboflow_universe"
    access_note = (
        "Official Roboflow SDK download (YOLOv8 export) with license and "
        "version resolved through the Universe REST API. Requires "
        "ROBOFLOW_API_KEY; skipped entirely when the key is absent."
    )

    def __init__(self, cache_dir, timeout: float = 60.0, projects: tuple[RoboflowProject, ...] = DEFAULT_PROJECTS, api_key: str | None = None) -> None:
        super().__init__(cache_dir, timeout)
        self.projects = projects
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self._meta_cache: dict[str, dict] = {}

    def check_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "no_api_key:ROBOFLOW_API_KEY"
        try:
            import roboflow  # noqa: F401
        except Exception as exc:
            return False, f"sdk_unavailable:{type(exc).__name__}"
        try:
            session = self._session()
            r = session.get(f"{API_BASE}/", params={"api_key": self.api_key}, timeout=self.timeout)
            if r.status_code != 200:
                return False, f"http_{r.status_code}"
            return True, f"ok:workspace={r.json().get('workspace', '?')}"
        except Exception as exc:
            return False, f"unreachable:{type(exc).__name__}"

    def _project_meta(self, project: RoboflowProject) -> dict:
        key = f"{project.workspace}/{project.project}"
        if key in self._meta_cache:
            return self._meta_cache[key]
        session = self._session()
        r = session.get(f"{API_BASE}/{key}", params={"api_key": self.api_key}, timeout=self.timeout)
        r.raise_for_status()
        self._meta_cache[key] = r.json()
        return self._meta_cache[key]

    def _resolve(self, project: RoboflowProject) -> tuple[int, str]:
        """Return (version, license) from the API."""
        meta = self._project_meta(project)
        license_raw = (meta.get("project") or {}).get("license") or ""
        if project.version is not None:
            return project.version, license_raw

        versions = meta.get("versions") or []
        numbers: list[int] = []
        for v in versions:
            tail = str(v.get("id", "")).rsplit("/", 1)[-1]
            if tail.isdigit():
                numbers.append(int(tail))
        if not numbers:
            raise ValueError(f"{project.project}: no downloadable versions")
        return max(numbers), license_raw

    def _download(self, project: RoboflowProject, version: int) -> Path | None:
        target = self.cache_dir / f"{project.project}_v{version}"
        if (target / "data.yaml").is_file():
            return target
        try:
            from roboflow import Roboflow

            # The SDK writes tqdm bars straight to stdout; keep logs readable.
            with contextlib.redirect_stdout(io.StringIO()):
                rf = Roboflow(api_key=self.api_key)
                dataset = (
                    rf.workspace(project.workspace)
                    .project(project.project)
                    .version(version)
                    .download("yolov8", location=str(target))
                )
            return Path(dataset.location)
        except Exception as exc:
            log.warning("roboflow download failed %s v%s: %s", project.project, version, exc)
            return None

    @staticmethod
    def _label_for(image_path: Path) -> Path:
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

    @staticmethod
    def _parse_boxes(label_path: Path) -> list[YoloBox]:
        boxes: list[YoloBox] = []
        if not label_path.is_file():
            return boxes
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                boxes.append(YoloBox(cls, xc, yc, w, h))
        return boxes

    @classmethod
    def _plate_from_filename(cls, stem: str) -> tuple[str | None, str | None]:
        """Return (plate_text, plate_type_override) from an export filename."""
        base = _RF_SUFFIX_RE.sub("", stem)
        candidate = normalize_plate(base)
        ok, normalized = validate_plate(candidate, allow_hash=False)
        if ok:
            return normalized, None
        if looks_like_special_plate(candidate):
            return None, "other"
        return None, None

    def collect(self, limit: int) -> Iterator[RawItem]:
        remaining = limit
        for project in self.projects:
            if remaining <= 0:
                return
            try:
                version, license_raw = self._resolve(project)
            except Exception as exc:
                log.warning("roboflow metadata failed %s: %s", project.project, exc)
                continue

            root = self._download(project, version)
            if root is None:
                continue

            for image_path in sorted(root.rglob("*")):
                if remaining <= 0:
                    return
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                boxes = self._parse_boxes(self._label_for(image_path))
                if not boxes:
                    continue

                plate_text, type_override = (None, None)
                if project.filename_has_plate:
                    plate_text, type_override = self._plate_from_filename(image_path.stem)
                    # One filename cannot label several plates unambiguously
                    if len(boxes) != 1:
                        plate_text = None

                yield RawItem(
                    source=f"{self.name}:{project.workspace}/{project.project}@v{version}",
                    source_url=project.landing_url,
                    license_raw=license_raw,
                    filename=image_path.name,
                    path=image_path,
                    query=f"{project.project}@v{version}",
                    suggested_type=type_override or project.plate_type,
                    type_is_authoritative=project.homogeneous_type,
                    boxes=boxes,
                    plate_text=plate_text,
                    plate_text_source="source_filename" if plate_text else "",
                )
                remaining -= 1
