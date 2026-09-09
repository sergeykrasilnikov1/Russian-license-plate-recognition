"""Roboflow Universe collector (official SDK, requires an API key).

Universe hosts several Russian-plate detection and type-classification
projects under CC BY 4.0. Access needs a personal API key, so without
`ROBOFLOW_API_KEY` the collector reports itself unavailable instead of
falling back to scraping.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .base import Collector, RawItem, YoloBox

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class RoboflowProject:
    workspace: str
    project: str
    version: int
    license_raw: str = "CC BY 4.0"
    plate_type: str = "type1"

    @property
    def landing_url(self) -> str:
        return f"https://universe.roboflow.com/{self.workspace}/{self.project}"


DEFAULT_PROJECTS: tuple[RoboflowProject, ...] = (
    RoboflowProject("nikolai-lebedev", "russian-license-plates-detector", 1, plate_type="type1"),
    RoboflowProject("roboflow-universe-projects", "license-plates-f8fx4", 1, plate_type="type1"),
)


class RoboflowCollector(Collector):
    name = "roboflow_universe"
    access_note = (
        "Official Roboflow SDK download (YOLOv8 export). Requires "
        "ROBOFLOW_API_KEY; skipped entirely when the key is absent."
    )

    def __init__(self, cache_dir, timeout: float = 60.0, projects: tuple[RoboflowProject, ...] = DEFAULT_PROJECTS, api_key: str | None = None) -> None:
        super().__init__(cache_dir, timeout)
        self.projects = projects
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")

    def check_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "no_api_key:ROBOFLOW_API_KEY"
        try:
            import roboflow  # noqa: F401
        except Exception as exc:
            return False, f"sdk_unavailable:{type(exc).__name__}"
        return True, "ok"

    def _download(self, project: RoboflowProject) -> Path | None:
        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key=self.api_key)
            ds = (
                rf.workspace(project.workspace)
                .project(project.project)
                .version(project.version)
                .download("yolov8", location=str(self.cache_dir / project.project))
            )
            return Path(ds.location)
        except Exception as exc:
            log.warning("roboflow download failed %s: %s", project.project, exc)
            return None

    @staticmethod
    def _label_for(image_path: Path) -> Path:
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

    def collect(self, limit: int) -> Iterator[RawItem]:
        remaining = limit
        for project in self.projects:
            if remaining <= 0:
                return
            root = self._download(project)
            if root is None:
                continue
            for image_path in sorted(root.rglob("*")):
                if remaining <= 0:
                    return
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label_path = self._label_for(image_path)
                boxes: list[YoloBox] = []
                if label_path.is_file():
                    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                boxes.append(
                                    YoloBox(int(float(parts[0])), *(float(v) for v in parts[1:5]))
                                )
                            except ValueError:
                                continue
                if not boxes:
                    continue
                yield RawItem(
                    source=f"{self.name}:{project.project}",
                    source_url=project.landing_url,
                    license_raw=project.license_raw,
                    filename=image_path.name,
                    path=image_path,
                    query=project.project,
                    suggested_type=project.plate_type,
                    boxes=boxes,
                )
                remaining -= 1
