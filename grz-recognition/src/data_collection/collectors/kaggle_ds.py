"""Kaggle collector over the documented REST API v1.

Authentication uses a Kaggle API token (`KGAT_...`) as a Bearer header, read
from `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` — both documented
locations. The `kaggle`/`kagglehub` SDKs are deliberately not used here: the
installed builds disagree on `kagglesdk` internals, while the REST endpoints
are stable and give us the dataset license directly.

Only datasets whose Kaggle-declared license permits redistribution and
derivatives are listed. Copyleft-licensed collections (e.g. the LGPL-3.0
Nomeroff dump) are excluded on purpose: the submitted dataset is published
under CC BY 4.0 and mixing copyleft into it would be a license conflict.
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from .base import Collector, RawItem, YoloBox

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
API_BASE = "https://www.kaggle.com/api/v1"
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"
LEGACY_FILE = Path.home() / ".kaggle" / "kaggle.json"


@dataclass(frozen=True)
class KaggleDataset:
    ref: str
    #: "yolo" | "voc" | "coco" | "none" — how annotations are stored
    annotation_format: str = "none"
    plate_type: str = "type1"
    homogeneous_type: bool = True
    max_bytes: int = 700_000_000

    @property
    def landing_url(self) -> str:
        return f"https://www.kaggle.com/datasets/{self.ref}"


DEFAULT_DATASETS: tuple[KaggleDataset, ...] = (
    # Non-Russian plates: exactly what the `other` class needs as negatives.
    KaggleDataset("andrewmvd/car-plate-detection", annotation_format="voc", plate_type="other"),
)

# Deliberately excluded, keep the reasoning with the code:
#   evgrafovmaxim/nomeroff-russian-license-plates — LGPL-3.0 copyleft, cannot
#     be relicensed into a CC BY 4.0 dataset.
#   adilshamim8/license-plate-recognition — 10125 Vietnamese plates and zero
#     annotation files; 1.1 GB of unlabeled foreign data.
#   egorandreasyan/car-number-segment — Russian plates, but already cropped to
#     the plate (38-145 px per side), so the boxes are trivial and the crops
#     fail the resolution filter. COCO parsing stays supported for reuse.


class KaggleCollector(Collector):
    name = "kaggle"
    access_note = (
        "Documented Kaggle REST API v1 with a Bearer API token from "
        "KAGGLE_API_TOKEN or ~/.kaggle/access_token. License taken from "
        "/datasets/view. Skipped when no token is present."
    )

    def __init__(self, cache_dir, timeout: float = 120.0, datasets: tuple[KaggleDataset, ...] = DEFAULT_DATASETS, token: str | None = None) -> None:
        super().__init__(cache_dir, timeout)
        self.datasets = datasets
        self.token = token or self._read_token()

    @staticmethod
    def _read_token() -> str:
        env = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        if env:
            return env
        if TOKEN_FILE.is_file():
            return TOKEN_FILE.read_text(encoding="utf-8").strip()
        if LEGACY_FILE.is_file():
            import json

            try:
                data = json.loads(LEGACY_FILE.read_text(encoding="utf-8"))
                return str(data.get("key", "")).strip()
            except Exception:
                return ""
        return ""

    def _session(self):
        session = super()._session()
        session.headers["Authorization"] = f"Bearer {self.token}"
        return session

    def check_available(self) -> tuple[bool, str]:
        if not self.token:
            return False, "no_token:KAGGLE_API_TOKEN or ~/.kaggle/access_token"
        try:
            session = self._session()
            r = session.get(f"{API_BASE}/datasets/list", params={"pageSize": 1}, timeout=self.timeout)
            if r.status_code == 200:
                return True, "ok"
            if r.status_code in (401, 403):
                return False, f"http_{r.status_code}:token_rejected_or_expired"
            return False, f"http_{r.status_code}"
        except Exception as exc:
            return False, f"unreachable:{type(exc).__name__}"

    def _license_of(self, session, dataset: KaggleDataset) -> str:
        try:
            r = session.get(f"{API_BASE}/datasets/view/{dataset.ref}", timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
            return str(payload.get("licenseNameNullable") or payload.get("licenseName") or "")
        except Exception as exc:
            log.warning("kaggle license lookup failed %s: %s", dataset.ref, exc)
            return ""

    def _download(self, session, dataset: KaggleDataset) -> Path | None:
        target = self.cache_dir / dataset.ref.replace("/", "__")
        extracted = target / "extracted"
        if extracted.is_dir() and any(extracted.rglob("*")):
            return extracted

        target.mkdir(parents=True, exist_ok=True)
        archive = target / "dataset.zip"
        if not archive.is_file():
            url = f"{API_BASE}/datasets/download/{dataset.ref}"
            try:
                with session.get(url, timeout=self.timeout, stream=True) as r:
                    r.raise_for_status()
                    declared = int(r.headers.get("Content-Length") or 0)
                    if declared and declared > dataset.max_bytes:
                        log.warning(
                            "kaggle %s is %.0f MB, above the %.0f MB budget; skipping",
                            dataset.ref,
                            declared / 1e6,
                            dataset.max_bytes / 1e6,
                        )
                        return None
                    written = 0
                    tmp = archive.with_suffix(".part")
                    with tmp.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            written += len(chunk)
                            if written > dataset.max_bytes:
                                log.warning("kaggle %s exceeded byte budget mid-download", dataset.ref)
                                tmp.unlink(missing_ok=True)
                                return None
                    tmp.replace(archive)
            except Exception as exc:
                log.warning("kaggle download failed %s: %s", dataset.ref, exc)
                return None

        try:
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)
            for inner in list(extracted.rglob("*.zip")):
                with zipfile.ZipFile(inner) as zf:
                    zf.extractall(inner.with_suffix(""))
        except Exception as exc:
            log.warning("kaggle unzip failed %s: %s", dataset.ref, exc)
            return None
        return extracted

    @staticmethod
    def _yolo_boxes(root: Path, image_path: Path) -> list[YoloBox]:
        candidates = [
            image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
            image_path.with_suffix(".txt"),
        ]
        for label_path in candidates:
            if not label_path.is_file():
                continue
            boxes: list[YoloBox] = []
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    boxes.append(YoloBox(int(float(parts[0])), *(float(v) for v in parts[1:5])))
                except ValueError:
                    continue
            if boxes:
                return boxes
        return []

    @staticmethod
    def _voc_boxes(root: Path, image_path: Path) -> list[YoloBox]:
        """Pascal VOC XML → normalized YOLO boxes."""
        candidates = list(root.rglob(f"{image_path.stem}.xml"))
        if not candidates:
            return []
        try:
            tree = ElementTree.parse(candidates[0])
        except Exception:
            return []
        size = tree.find("size")
        if size is None:
            return []
        try:
            iw = float(size.findtext("width") or 0)
            ih = float(size.findtext("height") or 0)
        except ValueError:
            return []
        if iw <= 0 or ih <= 0:
            return []

        boxes: list[YoloBox] = []
        for obj in tree.iter("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                x0 = float(box.findtext("xmin") or 0)
                y0 = float(box.findtext("ymin") or 0)
                x1 = float(box.findtext("xmax") or 0)
                y1 = float(box.findtext("ymax") or 0)
            except ValueError:
                continue
            w, h = (x1 - x0) / iw, (y1 - y0) / ih
            if w <= 0 or h <= 0:
                continue
            boxes.append(YoloBox(0, (x0 / iw) + w / 2, (y0 / ih) + h / 2, w, h))
        return boxes

    def _coco_index(self, root: Path) -> dict[str, list[YoloBox]]:
        """Build filename → YOLO boxes from every `*_annotations.coco.json`."""
        import json

        index: dict[str, list[YoloBox]] = {}
        for ann_path in root.rglob("*_annotations.coco.json"):
            try:
                payload = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("kaggle coco parse failed %s: %s", ann_path, exc)
                continue

            images = {img["id"]: img for img in payload.get("images", [])}
            for ann in payload.get("annotations", []):
                img = images.get(ann.get("image_id"))
                bbox = ann.get("bbox")
                if not img or not bbox or len(bbox) != 4:
                    continue
                iw, ih = float(img.get("width") or 0), float(img.get("height") or 0)
                x, y, bw, bh = (float(v) for v in bbox)
                if iw <= 0 or ih <= 0 or bw <= 0 or bh <= 0:
                    continue
                w, h = bw / iw, bh / ih
                index.setdefault(str(img.get("file_name")), []).append(
                    YoloBox(0, x / iw + w / 2, y / ih + h / 2, w, h)
                )
        return index

    def collect(self, limit: int) -> Iterator[RawItem]:
        session = self._session()
        remaining = limit

        for dataset in self.datasets:
            if remaining <= 0:
                return
            license_raw = self._license_of(session, dataset)
            root = self._download(session, dataset)
            if root is None:
                continue

            coco = self._coco_index(root) if dataset.annotation_format == "coco" else {}

            for image_path in sorted(root.rglob("*")):
                if remaining <= 0:
                    return
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                if dataset.annotation_format == "yolo":
                    boxes = self._yolo_boxes(root, image_path)
                elif dataset.annotation_format == "voc":
                    boxes = self._voc_boxes(root, image_path)
                elif dataset.annotation_format == "coco":
                    boxes = coco.get(image_path.name, [])
                else:
                    boxes = []

                yield RawItem(
                    source=f"{self.name}:{dataset.ref}",
                    source_url=dataset.landing_url,
                    license_raw=license_raw,
                    filename=image_path.name,
                    path=image_path,
                    query=dataset.ref,
                    suggested_type=dataset.plate_type,
                    type_is_authoritative=dataset.homogeneous_type,
                    boxes=boxes,
                )
                remaining -= 1
