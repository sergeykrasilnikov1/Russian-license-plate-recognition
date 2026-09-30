"""Evaluate the pinned Lin-Lini pose+CRNN reference on our held-out corpus."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
REFERENCE = OUTPUT / "reference_solution"
COMMIT = "cc1dee8bc59cdb71db3ccd587841c8ea487be37a"
OCR_SHA256 = "96e4f374c3aae85dcd398fc899ac11ae8094204c8136fd23589bbde7cbce1dae"
DET_SHA256 = "9959f887421f80bbe247f6babccb8366bc1aef742eda1c5b339e60df77eaaa80"
IOU_THRESHOLD = 0.5


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if version("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.156"])


def download(relative: str, destination: Path, expected_sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/Lin-Lini/volga-it-2026-lpr/{COMMIT}/solution/{relative}"
    urllib.request.urlretrieve(url, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {relative}: {digest}")


for module in ("__init__.py", "detector.py", "ocr.py", "pipeline.py", "plates.py", "typing.py"):
    download(f"lpr/{module}", REFERENCE / "lpr" / module)
download("LICENSE", REFERENCE / "LICENSE")
download("weights/ocr.pt", REFERENCE / "ocr.pt", OCR_SHA256)
download("weights/det.pt", REFERENCE / "det.pt", DET_SHA256)
sys.path.insert(0, str(REFERENCE))

from lpr.ocr import apply_unknown
from lpr.pipeline import LPRPipeline, fix_lookalikes_rows, rectify, split_rows
from lpr.plates import fix_lookalikes


def find_data() -> Path:
    roots = [path.parent for path in INPUT.glob("**/meta.csv") if (path.parent / "splits/val.txt").is_file()]
    roots = sorted(set(roots))
    if len(roots) != 1:
        raise RuntimeError(f"expected one project dataset root, got {roots}")
    return roots[0]


DATA = find_data()


def relative_image(line: str) -> str:
    path = Path(line.strip())
    if not path.is_absolute():
        return path.as_posix().removeprefix("./")
    if path.parts.count("images") == 1:
        return Path(*path.parts[path.parts.index("images"):]).as_posix()
    return path.as_posix()


def split(name: str) -> set[str]:
    path = DATA / "splits" / f"{name}.txt"
    return {relative_image(line) for line in path.read_text().splitlines() if line.strip()}


def file_hash(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def parse_quad(raw: str) -> np.ndarray:
    values = [float(value) for value in str(raw).replace(";", ",").split(",") if value.strip()]
    if len(values) != 8:
        raise ValueError(f"expected 8 quad values, got {len(values)}")
    quad = np.asarray(values, dtype=np.float32).reshape(4, 2)
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    quad = quad[np.argsort(angles)]
    return np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)


def readable(text: str) -> bool:
    return bool(text) and "#" not in text


def normalise(text: str) -> str:
    return str(text).upper().translate(str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")).replace(" ", "")


def exact(prediction: str, target: str) -> bool:
    return normalise(prediction) == normalise(target)


def char_accuracy(prediction: str, target: str) -> float:
    pred, truth = normalise(prediction), normalise(target)
    length = max(len(pred), len(truth), 1)
    return sum(a == b for a, b in zip(pred.ljust(length, "\0"), truth.ljust(length, "\0"))) / length


def wilson(hits: int, count: int, z: float = 1.96) -> tuple[float, float, float]:
    if not count:
        return float("nan"), float("nan"), float("nan")
    p = hits / count
    z2, denominator = z * z, 1 + z * z / count
    centre = (p + z2 / (2 * count)) / denominator
    margin = z * ((p * (1 - p) / count + z2 / (4 * count * count)) ** 0.5) / denominator
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def bbox_from_quad(quad: np.ndarray) -> tuple[float, float, float, float]:
    minimum, maximum = quad.min(axis=0), quad.max(axis=0)
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match(gt_rows: list[dict], predictions: list[dict]):
    candidates = sorted(
        (
            (iou(gt["bbox"], prediction["bbox"]), gt_index, pred_index)
            for gt_index, gt in enumerate(gt_rows)
            for pred_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_gt, matched_pred, output = set(), set(), []
    for overlap, gt_index, pred_index in candidates:
        if overlap < IOU_THRESHOLD:
            break
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            output.append((gt_index, pred_index, overlap))
    return output, matched_gt, matched_pred


def group(row: dict) -> str:
    return f"{row['plate_type']}/" + ("synthetic" if row["is_synthetic"] else "real")


def summarize_ocr(rows: list[dict]) -> dict:
    result = {}
    for name in ["all"] + sorted({group(row) for row in rows}):
        subset = rows if name == "all" else [row for row in rows if group(row) == name]
        hits = sum(row["exact"] for row in subset)
        p, low, high = wilson(hits, len(subset))
        result[name] = {
            "n": len(subset),
            "exact": p,
            "exact_ci95": [low, high],
            "char_accuracy": float(np.mean([row["char_accuracy"] for row in subset])) if subset else None,
        }
    return result


meta = pd.read_csv(DATA / "meta.csv", sep=";", dtype={"plate_num": str})
train_paths, val_paths = split("train"), split("val")
train_hashes = {file_hash(DATA / rel) for rel in train_paths if (DATA / rel).is_file()}
val_paths_existing = sorted({str(row.image) for _, row in meta.iterrows() if str(row.image) in val_paths and (DATA / str(row.image)).is_file()})
selected_paths, excluded_train_overlap, excluded_eval_duplicates, seen_hashes = [], [], [], set()
for rel in val_paths_existing:
    digest = file_hash(DATA / rel)
    if digest in train_hashes:
        excluded_train_overlap.append(rel)
    elif digest in seen_hashes:
        excluded_eval_duplicates.append(rel)
    else:
        selected_paths.append(rel)
        seen_hashes.add(digest)

gt_by_image: dict[str, list[dict]] = defaultdict(list)
for _, row in meta[meta.image.astype(str).isin(selected_paths)].iterrows():
    quad = parse_quad(str(row.quad))
    gt_by_image[str(row.image)].append(
        {
            "image": str(row.image),
            "plate_num": str(row.plate_num),
            "plate_type": str(row.plate_type),
            "is_synthetic": bool(int(row.is_synthetic)),
            "quad": quad,
            "bbox": bbox_from_quad(quad),
        }
    )

pipeline = LPRPipeline(
    str(REFERENCE / "det.pt"),
    str(REFERENCE / "ocr.pt"),
    vehicle_weights=None,
    device="cpu",
    det_imgsz=960,
    det_conf=0.2,
    unk_thr=0.45,
    min_conf=0.2,
    vehicle_gate=False,
    emit_other=True,
    half=False,
    unk_drop_frac=0.34,
    unk_drop_conf=0.5,
)
pipeline.warmup()

# OCR-only upper bound with known type and project GT corners.
samples, tasks = [], []
for rel, gt_rows in gt_by_image.items():
    image = cv2.imread(str(DATA / rel))
    for gt in gt_rows:
        if not readable(gt["plate_num"]):
            continue
        owner = len(samples)
        samples.append(gt)
        if gt["plate_type"] == "type1a":
            top, bottom = split_rows(rectify(image, gt["quad"], (192, 112)))
            tasks.extend([(owner, "top", top), (owner, "bottom", bottom)])
        else:
            tasks.append((owner, "one", rectify(image, gt["quad"], (208, 48))))

decoded = defaultdict(dict)
ocr_started = time.perf_counter()
for offset in range(0, len(tasks), 128):
    batch = tasks[offset:offset + 128]
    outputs = pipeline.ocr.read_rows([task[2] for task in batch])
    for (owner, part, _), output in zip(batch, outputs):
        decoded[owner][part] = output
ocr_elapsed = time.perf_counter() - ocr_started
ocr_rows = []
for owner, gt in enumerate(samples):
    parts = decoded[owner]
    if gt["plate_type"] == "type1a":
        top = apply_unknown(parts["top"][0], parts["top"][1], 0.45)[0]
        bottom = apply_unknown(parts["bottom"][0], parts["bottom"][1], 0.45)[0]
        top, bottom = fix_lookalikes_rows(top, bottom)
        prediction, raw = top + bottom, parts["top"][0] + "/" + parts["bottom"][0]
    else:
        raw = parts["one"][0]
        prediction = fix_lookalikes(apply_unknown(raw, parts["one"][1], 0.45)[0], gt["plate_type"])
    ocr_rows.append(
        {
            **{key: gt[key] for key in ("image", "plate_num", "plate_type", "is_synthetic")},
            "raw": raw,
            "prediction": prediction,
            "exact": exact(prediction, gt["plate_num"]),
            "char_accuracy": char_accuracy(prediction, gt["plate_num"]),
        }
    )

# Full shipped reference pipeline, including pose corners and its typing/filtering.
prediction_rows, predictions_by_image, image_latencies = [], {}, []
for index, rel in enumerate(sorted(gt_by_image), 1):
    image = cv2.imread(str(DATA / rel))
    started = time.perf_counter()
    outputs = pipeline.process(image)
    image_latencies.append((time.perf_counter() - started) * 1000)
    predictions = []
    for output in outputs:
        prediction = {
            "image": rel,
            "plate_num": output.plate_num,
            "plate_type": output.plate_type,
            "confidence": output.confidence,
            "bbox": tuple(output.box),
            "quad": output.quad,
        }
        predictions.append(prediction)
        prediction_rows.append(prediction)
    predictions_by_image[rel] = predictions
    if index % 100 == 0:
        print(f"processed {index}/{len(gt_by_image)}", flush=True)

total_gt = total_predictions = total_matches = type_hits = false_positives = 0
readable_gt = matched_readable = exact_hits = 0
char_sum = 0.0
matched_rows = []
groups = defaultdict(lambda: {"n": 0, "detected": 0, "exact": 0})
for rel, gt_rows in gt_by_image.items():
    predictions = predictions_by_image.get(rel, [])
    pairs, matched_gt, matched_predictions = match(gt_rows, predictions)
    total_gt += len(gt_rows)
    total_predictions += len(predictions)
    total_matches += len(pairs)
    false_positives += len(predictions) - len(matched_predictions)
    for gt in gt_rows:
        if readable(gt["plate_num"]):
            readable_gt += 1
            groups[group(gt)]["n"] += 1
    for gt_index, pred_index, overlap in pairs:
        gt, prediction = gt_rows[gt_index], predictions[pred_index]
        type_hits += prediction["plate_type"] == gt["plate_type"]
        if readable(gt["plate_num"]):
            matched_readable += 1
            groups[group(gt)]["detected"] += 1
            is_exact = exact(prediction["plate_num"], gt["plate_num"])
            exact_hits += is_exact
            groups[group(gt)]["exact"] += is_exact
            chars = char_accuracy(prediction["plate_num"], gt["plate_num"])
            char_sum += chars
            matched_rows.append(
                {
                    "image": rel,
                    "ground_truth": gt["plate_num"],
                    "gt_type": gt["plate_type"],
                    "domain": "synthetic" if gt["is_synthetic"] else "real",
                    "prediction": prediction["plate_num"],
                    "predicted_type": prediction["plate_type"],
                    "iou": overlap,
                    "exact": is_exact,
                    "char_accuracy": chars,
                }
            )

e2e_p, e2e_low, e2e_high = wilson(exact_hits, readable_gt)
matched_p, matched_low, matched_high = wilson(exact_hits, matched_readable)
group_summary = {}
for name, values in sorted(groups.items()):
    p, low, high = wilson(values["exact"], values["n"])
    group_summary[name] = {
        **values,
        "detection_recall": values["detected"] / values["n"] if values["n"] else None,
        "e2e_exact": p,
        "e2e_exact_ci95": [low, high],
    }

latencies = np.asarray(image_latencies)
summary = {
    "protocol": {
        "reference_repo": "https://github.com/Lin-Lini/volga-it-2026-lpr",
        "reference_commit": COMMIT,
        "solution_license": "MIT; Ultralytics component AGPL-3.0",
        "weights": {"ocr_sha256": OCR_SHA256, "det_sha256": DET_SHA256},
        "dataset": "sergeniy/grz-plates-detector val",
        "leakage_rule": "exclude exact SHA-256 overlap with project OCR train; upstream reference-training overlap unknown",
        "matching": "greedy one-to-one IoU >= 0.5",
        "settings": "reference defaults: pose imgsz=960, det_conf=0.2, unk_thr=0.45, min_conf=0.2",
    },
    "environment": {
        "python": platform.python_version(),
        "torch": version("torch"),
        "ultralytics": version("ultralytics"),
        "opencv": cv2.__version__,
    },
    "corpus": {
        "images": len(gt_by_image),
        "gt_objects": sum(len(rows) for rows in gt_by_image.values()),
        "readable_gt": readable_gt,
        "excluded_train_overlap": excluded_train_overlap,
        "excluded_eval_duplicates": excluded_eval_duplicates,
    },
    "ocr_gt_quad": {
        "groups": summarize_ocr(ocr_rows),
        "latency": {
            "total_seconds_batched": ocr_elapsed,
            "mean_ms_per_plate_batched": 1000 * ocr_elapsed / len(ocr_rows),
            "batch": 128,
        },
    },
    "full_pipeline": {
        "gt_objects": total_gt,
        "predictions": total_predictions,
        "matches_iou_0_5": total_matches,
        "false_positives": false_positives,
        "detection_recall": total_matches / total_gt,
        "detection_precision": total_matches / total_predictions if total_predictions else None,
        "type_accuracy_on_matches": type_hits / total_matches if total_matches else None,
        "readable_gt": readable_gt,
        "matched_readable": matched_readable,
        "ocr_exact_given_match": matched_p,
        "ocr_exact_given_match_ci95": [matched_low, matched_high],
        "char_accuracy_given_match": char_sum / matched_readable if matched_readable else None,
        "end_to_end_exact": e2e_p,
        "end_to_end_exact_ci95": [e2e_low, e2e_high],
        "groups": group_summary,
        "latency_ms": {
            "mean": float(latencies.mean()),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
    },
}

pd.DataFrame(ocr_rows).to_csv(OUTPUT / "reference_ocr_predictions.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(OUTPUT / "reference_pipeline_predictions.csv", index=False)
pd.DataFrame(matched_rows).to_csv(OUTPUT / "reference_pipeline_matches.csv", index=False)
(OUTPUT / "reference_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
with (OUTPUT / "reference_pipeline.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, delimiter=";", lineterminator="\n")
    writer.writerow(["image", "plate_num", "plate_type", "confidence"])
    for row in prediction_rows:
        writer.writerow([row["image"], row["plate_num"], row["plate_type"], f"{row['confidence']:.4f}"])
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
