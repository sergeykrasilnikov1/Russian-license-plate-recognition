"""Writers for the two dataset ledgers.

`meta.csv` keeps exactly the ten columns fixed by the brief and only accepts
fully labeled plates. Candidate images that still lack a bbox wait in
`raw_downloads/manifest.csv` until Stage 4 labels them, so the dataset stays
valid for the organizers' validator at every point in time.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

META_COLUMNS: Sequence[str] = (
    "image",
    "plate_num",
    "plate_type",
    "bbox",
    "quad",
    "is_vehicle",
    "is_synthetic",
    "source",
    "license",
    "conditions",
)

MANIFEST_COLUMNS: Sequence[str] = (
    "file",
    "source",
    "source_url",
    "license",
    "license_spdx",
    "query",
    "coarse_type",
    "coarse_confidence",
    "aspect",
    "yellow_ratio",
    "width",
    "height",
    "faces_blurred",
    "conditions",
    "label_status",
)


@dataclass
class MetaRow:
    image: str
    plate_num: str
    plate_type: str
    bbox: str
    quad: str
    is_vehicle: int
    is_synthetic: int
    source: str
    license: str
    conditions: str

    @staticmethod
    def format_bbox(x: float, y: float, w: float, h: float) -> str:
        return f"{int(round(x))},{int(round(y))},{int(round(w))},{int(round(h))}"

    @staticmethod
    def format_quad(points: Iterable[tuple[float, float]]) -> str:
        return ",".join(f"{int(round(px))},{int(round(py))}" for px, py in points)

    @staticmethod
    def quad_from_bbox(x: float, y: float, w: float, h: float) -> str:
        return MetaRow.format_quad(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        )


@dataclass
class ManifestRow:
    file: str
    source: str
    source_url: str
    license: str
    license_spdx: str
    query: str = ""
    coarse_type: str = ""
    coarse_confidence: str = ""
    aspect: str = ""
    yellow_ratio: str = ""
    width: int = 0
    height: int = 0
    faces_blurred: int = 0
    conditions: str = ""
    label_status: str = "pending_stage4"


class _CsvLedger:
    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = Path(path)
        self.columns = list(columns)
        self._rows: list[dict] = []
        self.loaded_from_disk = False
        if self.path.is_file() and self.path.stat().st_size > 0:
            with self.path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                if reader.fieldnames != self.columns:
                    # Silently starting empty here would make the next flush
                    # overwrite the ledger and orphan every collected image.
                    raise ValueError(
                        f"{self.path} has unexpected columns {reader.fieldnames}; "
                        f"expected {self.columns}. Refusing to overwrite it."
                    )
                self._rows = [dict(r) for r in reader]
                self.loaded_from_disk = True

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[dict]:
        return list(self._rows)

    def keys(self, column: str) -> set[str]:
        return {r[column] for r in self._rows if column in r}

    def append(self, row) -> None:
        """Values are stored as text so in-memory rows match reloaded ones."""
        data = asdict(row)
        self._rows.append({c: str(data[c]) for c in self.columns})

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, delimiter=";", lineterminator="\n")
            writer.writeheader()
            writer.writerows(self._rows)
        tmp.replace(self.path)


class MetaStore(_CsvLedger):
    """`dataset/meta.csv` — one row per fully labeled plate."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, META_COLUMNS)

    def existing_images(self) -> set[str]:
        return self.keys("image")

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._rows:
            counts[r["plate_type"]] = counts.get(r["plate_type"], 0) + 1
        return counts

    def unique_plates(self, plate_type: str | None = None) -> set[str]:
        return {
            r["plate_num"]
            for r in self._rows
            if "#" not in r["plate_num"]
            and (plate_type is None or r["plate_type"] == plate_type)
        }


class ManifestStore(_CsvLedger):
    """`dataset/raw_downloads/manifest.csv` — candidates awaiting Stage 4 labels."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, MANIFEST_COLUMNS)

    def existing_files(self) -> set[str]:
        return self.keys("file")

    def coarse_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._rows:
            counts[r["coarse_type"]] = counts.get(r["coarse_type"], 0) + 1
        return counts


def write_yolo_label(path: Path, class_id: int, bbox_xywh: tuple[float, float, float, float], image_wh: tuple[int, int]) -> None:
    """One `.txt` per image in YOLO format: class x_center y_center w h (normalized)."""
    x, y, w, h = bbox_xywh
    iw, ih = image_wh
    if iw <= 0 or ih <= 0:
        raise ValueError("image size must be positive")
    xc = (x + w / 2) / iw
    yc = (y + h / 2) / ih
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{class_id} {xc:.6f} {yc:.6f} {w / iw:.6f} {h / ih:.6f}\n", encoding="utf-8"
    )


def append_yolo_label(path: Path, class_id: int, bbox_xywh: tuple[float, float, float, float], image_wh: tuple[int, int]) -> None:
    x, y, w, h = bbox_xywh
    iw, ih = image_wh
    xc = (x + w / 2) / iw
    yc = (y + h / 2) / ih
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w / iw:.6f} {h / ih:.6f}\n")
