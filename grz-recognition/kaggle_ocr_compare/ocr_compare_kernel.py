"""Compare plate OCR candidates on one deduplicated held-out crop set."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/ocr_compare")
WORK.mkdir(parents=True, exist_ok=True)


def find_one(pattern: str) -> Path:
    matches = list(INPUT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, got {matches}")
    return matches[0]


assets = find_one("**/weights/ocr/en_PP-OCRv4_rec_mobile.onnx").parents[2]
sys.path.insert(0, str(assets))

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "fast-plate-ocr[onnx]"],
)

from fast_plate_ocr import LicensePlateRecognizer
from src.ocr.backends.ppocr import PPOCRV4Backend, PPOCRV5Backend
from src.ocr.charset import finalize_plate
from src.ocr.metrics import char_accuracy, full_plate_match, wilson_interval
from src.ocr.warp import parse_quad, warp_plate

DATA = next(
    path.parent
    for path in INPUT.glob("**/meta.csv")
    if (path.parent / "splits/val.txt").is_file()
)
meta = pd.read_csv(DATA / "meta.csv", sep=";", dtype={"plate_num": str})


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


train_paths, val_paths = split("train"), split("val")
train_hashes = set()
for rel in train_paths:
    path = DATA / rel
    if path.is_file():
        train_hashes.add(hashlib.sha256(path.read_bytes()).digest())

rows = []
excluded_leaks = []
for _, row in meta.iterrows():
    rel = str(row.image)
    path = DATA / rel
    gt = str(row.plate_num)
    if rel not in val_paths or not path.is_file() or not gt or "#" in gt:
        continue
    if hashlib.sha256(path.read_bytes()).digest() in train_hashes:
        excluded_leaks.append(rel)
        continue
    rows.append(row)

crops, labels = [], []
for row in rows:
    image = cv2.imread(str(DATA / str(row.image)))
    if image is None:
        continue
    try:
        crop = warp_plate(image, parse_quad(str(row.quad)), str(row.plate_type), type1a_mode="flatten", scale=0.8)
    except Exception as exc:
        print(f"skip warp {row.image}: {exc}")
        continue
    crops.append(crop)
    labels.append(row)

if not crops:
    raise RuntimeError("no evaluation crops")


class FastPlateCandidate:
    def __init__(self, model_name: str):
        self.name = model_name
        self.model = LicensePlateRecognizer(hub_ocr_model=model_name, device="cpu")

    def recognize_detailed(self, crop):
        mode = self.model.config.image_color_mode
        if mode == "grayscale":
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif mode == "rgb":
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"unsupported color mode {mode}")
        pred = self.model.run(image, return_confidence=True)[0]
        text = pred.plate
        scores = np.asarray(pred.char_probs, dtype=float).reshape(-1).tolist()[:len(text)]
        if len(scores) != len(text):
            raise RuntimeError("character confidence alignment failed")
        return text, scores


models = [
    PPOCRV4Backend(assets / "weights/ocr/en_PP-OCRv4_rec_mobile.onnx"),
    PPOCRV5Backend(assets / "weights/ocr/en_PP-OCRv5_rec_mobile.onnx"),
    FastPlateCandidate("cct-s-v2-global-model"),
    FastPlateCandidate("cct-xs-v2-global-model"),
]

pred_rows = []
summary = {
    "protocol": {
        "dataset": "sergeniy/grz-plates-detector",
        "split": "val",
        "leakage_rule": "exclude val images with exact SHA-256 match in train",
        "excluded_exact_leaks": excluded_leaks,
        "warp": "GT quad, type1a flatten, scale 0.8",
        "min_char_conf": 0.35,
        "fast_plate_ocr_version": importlib.metadata.version("fast-plate-ocr"),
        "note": "OCR-only validation. Synthetic type1a/type1b do not prove real-target quality.",
    },
    "models": {},
}

for model in models:
    for crop in crops[:5]:
        model.recognize_detailed(crop)
    model_rows = []
    timings = []
    for crop, row in zip(crops, labels):
        started = time.perf_counter()
        raw, scores = model.recognize_detailed(crop)
        timings.append((time.perf_counter() - started) * 1000)
        final, confidence = finalize_plate(raw, scores, 0.35)
        item = {
            "model": model.name,
            "image": str(row.image),
            "plate_type": str(row.plate_type),
            "domain": "synthetic" if int(row.is_synthetic) else "real",
            "ground_truth": str(row.plate_num),
            "raw": raw,
            "prediction": final,
            "confidence": confidence,
            "char_accuracy": char_accuracy(final, str(row.plate_num)),
            "exact": full_plate_match(final, str(row.plate_num)),
        }
        pred_rows.append(item)
        model_rows.append(item)
    groups = {}
    for group in ["all"] + sorted({f"{r['plate_type']}/{r['domain']}" for r in model_rows}):
        subset = model_rows if group == "all" else [r for r in model_rows if f"{r['plate_type']}/{r['domain']}" == group]
        hits = sum(r["exact"] for r in subset)
        p, lo, hi = wilson_interval(hits, len(subset))
        groups[group] = {
            "n": len(subset), "exact": p, "exact_ci95": [lo, hi],
            "char_accuracy": float(np.mean([r["char_accuracy"] for r in subset])),
        }
    summary["models"][model.name] = {
        "groups": groups,
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
        },
    }
    print(model.name, json.dumps(summary["models"][model.name], ensure_ascii=False))

pd.DataFrame(pred_rows).to_csv("/kaggle/working/ocr_candidate_predictions.csv", index=False)
Path("/kaggle/working/ocr_comparison_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(summary["protocol"], indent=2, ensure_ascii=False))
