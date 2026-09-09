"""Hugging Face Hub collector for ready-made, annotated plate datasets.

Downloads go through the official `huggingface_hub` SDK rather than HTML
scraping. The default dataset (AY000554/Car_plate_detecting_dataset, CC BY 4.0,
derived from AUTO.RIA Numberplate) ships YOLO bboxes, so its images enter
`meta.csv` immediately with a real bbox instead of waiting for Stage 4.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .base import Collector, RawItem, YoloBox

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class HfDatasetSpec:
    repo_id: str
    license_raw: str
    #: zip archives inside the repo, smallest first so a partial run still yields data
    archives: tuple[str, ...]
    plate_type: str = "type1"
    #: The dataset card documents a single plate type for every image
    homogeneous_type: bool = True
    url: str = ""

    @property
    def landing_url(self) -> str:
        return self.url or f"https://huggingface.co/datasets/{self.repo_id}"


DEFAULT_SPECS: tuple[HfDatasetSpec, ...] = (
    HfDatasetSpec(
        repo_id="AY000554/Car_plate_detecting_dataset",
        license_raw="CC BY 4.0",
        archives=("val.zip", "test.zip", "train.zip"),
        plate_type="type1",
    ),
)


class HuggingFaceCollector(Collector):
    name = "huggingface"
    access_note = (
        "Official huggingface_hub SDK (hf_hub_download). Datasets ship YOLO "
        "annotations, so bboxes come from the source, not from a model."
    )

    def __init__(self, cache_dir, timeout: float = 60.0, specs: tuple[HfDatasetSpec, ...] = DEFAULT_SPECS, archives: tuple[str, ...] | None = None) -> None:
        super().__init__(cache_dir, timeout)
        self.specs = specs
        self.archive_filter = archives

    def check_available(self) -> tuple[bool, str]:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return False, "huggingface_hub_not_installed"
        try:
            api = HfApi()
            for spec in self.specs:
                api.dataset_info(spec.repo_id, timeout=self.timeout)
            return True, "ok"
        except Exception as exc:
            return False, f"unreachable:{type(exc).__name__}"

    def _download(self, spec: HfDatasetSpec, archive: str) -> Path | None:
        from huggingface_hub import hf_hub_download

        try:
            return Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=archive,
                    repo_type="dataset",
                    cache_dir=str(self.cache_dir),
                )
            )
        except Exception as exc:
            log.warning("hf download failed %s/%s: %s", spec.repo_id, archive, exc)
            return None

    @staticmethod
    def _parse_yolo(text: str) -> list[YoloBox]:
        boxes: list[YoloBox] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue
            boxes.append(YoloBox(cls, xc, yc, w, h))
        return boxes

    def _iter_archive(self, spec: HfDatasetSpec, archive_path: Path, limit: int) -> Iterator[RawItem]:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            labels = {Path(n).stem: n for n in names if n.lower().endswith(".txt")}
            images = [n for n in names if Path(n).suffix.lower() in IMAGE_SUFFIXES]
            images.sort()

            produced = 0
            for name in images:
                if produced >= limit:
                    return
                stem = Path(name).stem
                label_name = labels.get(stem)
                if label_name is None:
                    continue
                try:
                    boxes = self._parse_yolo(zf.read(label_name).decode("utf-8", "ignore"))
                    if not boxes:
                        continue
                    data = zf.read(name)
                except Exception as exc:
                    log.warning("failed reading %s from %s: %s", name, archive_path.name, exc)
                    continue

                yield RawItem(
                    source=f"{self.name}:{spec.repo_id}",
                    source_url=spec.landing_url,
                    license_raw=spec.license_raw,
                    filename=f"{stem}{Path(name).suffix.lower()}",
                    data=data,
                    query=archive_path.name,
                    suggested_type=spec.plate_type,
                    type_is_authoritative=spec.homogeneous_type,
                    boxes=boxes,
                )
                produced += 1

    def collect(self, limit: int) -> Iterator[RawItem]:
        remaining = limit
        for spec in self.specs:
            archives = self.archive_filter or spec.archives
            for archive in archives:
                if remaining <= 0:
                    return
                path = self._download(spec, archive)
                if path is None:
                    continue
                for item in self._iter_archive(spec, path, remaining):
                    yield item
                    remaining -= 1
                    if remaining <= 0:
                        return
