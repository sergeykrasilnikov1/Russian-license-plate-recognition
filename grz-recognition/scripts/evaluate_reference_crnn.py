#!/usr/bin/env python3
"""Evaluate Lin-Lini's MIT-licensed CRNN on this project's held-out GT quads.

The reference source and checkpoint are supplied explicitly and are not copied
into this repository. This keeps the comparison reproducible without silently
changing the deployment bundle or its third-party licensing obligations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr.metrics import char_accuracy, full_plate_match, gt_is_readable, wilson_interval
from src.ocr.warp import parse_quad


def rectify(image: np.ndarray, quad: np.ndarray, size: tuple[int, int], margin: float = 0.03) -> np.ndarray:
    width, height = size
    expanded = quad.astype(np.float32).copy()
    centre = expanded.mean(axis=0)
    expanded = centre + (expanded - centre) * (1.0 + margin)
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    transform = cv2.getPerspectiveTransform(expanded, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def split_rows(rectified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = rectified.shape[0]
    return rectified[int(height * 0.06):int(height * 0.54)], rectified[int(height * 0.48):int(height * 0.97)]


def fix_two_rows(top: str, bottom: str, unknown: str = "#") -> tuple[str, str]:
    """Reference solution's position-wise correction for a type1a layout."""
    digit_to_letter = {"0": "O", "8": "B", "4": "A", "3": "E", "6": "B", "1": "T", "7": "T", "5": "C", "2": "E", "9": "P"}
    letter_to_digit = {"O": "0", "B": "8", "A": "4", "E": "3", "T": "7", "C": "0", "P": "9", "H": "4", "K": "4", "M": "4", "X": "4", "Y": "4"}

    def fix(text: str, mask: str) -> str:
        result = []
        for char, expected in zip(text, mask):
            if char == unknown:
                result.append(char)
            elif expected == "L" and char.isdigit():
                result.append(digit_to_letter.get(char, unknown))
            elif expected == "D" and char.isalpha():
                result.append(letter_to_digit.get(char, unknown))
            else:
                result.append(char)
        return "".join(result) + text[len(mask):]

    if len(top) == 4:
        top = fix(top, "LDDD")
    if 4 <= len(bottom) <= 5:
        bottom = fix(bottom, "LL" + "D" * (len(bottom) - 2))
    return top, bottom


def summarize(rows: list[dict]) -> dict:
    groups = ["all"] + sorted({f"{row['plate_type']}/{row['domain']}" for row in rows})
    output = {}
    for group in groups:
        subset = rows if group == "all" else [
            row for row in rows if f"{row['plate_type']}/{row['domain']}" == group
        ]
        hits = sum(bool(row["exact"]) for row in subset)
        accuracy, low, high = wilson_interval(hits, len(subset))
        output[group] = {
            "n": len(subset),
            "exact": accuracy,
            "exact_ci95": [low, high],
            "char_accuracy": float(np.mean([row["char_accuracy"] for row in subset])) if subset else None,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True, help="directory containing lpr/ and ocr.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "kaggle_upload/grz-plates-detector")
    parser.add_argument("--selection", type=Path, default=ROOT / "kaggle_results/pipeline_e2e_local/oracle_geometry_predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle_results/reference_crnn_gt_quad")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reference_root))
    from lpr.ocr import Recognizer, apply_unknown
    from lpr.plates import fix_lookalikes

    selected = pd.read_csv(args.selection, dtype={"plate_num": str})
    selected = selected[(selected.geometry == "gt_quad") & (selected.policy == "routed")]
    selected_keys = set(zip(selected.image.astype(str), selected.plate_num.astype(str), selected.plate_type.astype(str)))
    meta = pd.read_csv(args.dataset / "meta.csv", sep=";", dtype={"plate_num": str})

    samples, tasks = [], []
    for _, row in meta.iterrows():
        key = (str(row.image), str(row.plate_num), str(row.plate_type))
        if key not in selected_keys or not gt_is_readable(str(row.plate_num)):
            continue
        image = cv2.imread(str(args.dataset / str(row.image)))
        if image is None:
            continue
        quad = parse_quad(str(row.quad))
        index = len(samples)
        samples.append(
            {
                "image": str(row.image),
                "plate_num": str(row.plate_num),
                "plate_type": str(row.plate_type),
                "domain": "synthetic" if int(row.is_synthetic) else "real",
            }
        )
        if str(row.plate_type) == "type1a":
            top, bottom = split_rows(rectify(image, quad, (192, 112)))
            tasks.extend([(index, "top", top), (index, "bottom", bottom)])
        else:
            tasks.append((index, "one", rectify(image, quad, (208, 48))))

    recognizer = Recognizer(str(args.reference_root / "ocr.pt"), device="cpu", unk_thr=0.45)
    recognizer.read_rows([tasks[0][2], tasks[0][2]])
    decoded: dict[int, dict[str, tuple[str, list[float]]]] = defaultdict(dict)
    started = time.perf_counter()
    for offset in range(0, len(tasks), args.batch):
        batch = tasks[offset:offset + args.batch]
        predictions = recognizer.read_rows([task[2] for task in batch])
        for (owner, part, _), prediction in zip(batch, predictions):
            decoded[owner][part] = prediction
    elapsed = time.perf_counter() - started

    all_rows = []
    for threshold in (0.0, 0.35, 0.45):
        for index, sample in enumerate(samples):
            parts = decoded[index]
            if sample["plate_type"] == "type1a":
                top = apply_unknown(parts["top"][0], parts["top"][1], threshold)[0]
                bottom = apply_unknown(parts["bottom"][0], parts["bottom"][1], threshold)[0]
                top, bottom = fix_two_rows(top, bottom)
                prediction = top + bottom
                raw = parts["top"][0] + "/" + parts["bottom"][0]
            else:
                raw = parts["one"][0]
                prediction = apply_unknown(raw, parts["one"][1], threshold)[0]
                prediction = fix_lookalikes(prediction, sample["plate_type"])
            all_rows.append(
                {
                    **sample,
                    "threshold": threshold,
                    "raw": raw,
                    "prediction": prediction,
                    "exact": full_plate_match(prediction, sample["plate_num"]),
                    "char_accuracy": char_accuracy(prediction, sample["plate_num"]),
                }
            )

    summary = {
        "protocol": {
            "reference": "Lin-Lini/volga-it-2026-lpr main",
            "checkpoint_sha256": __import__("hashlib").sha256((args.reference_root / "ocr.pt").read_bytes()).hexdigest(),
            "selection": str(args.selection),
            "geometry": "GT quad; reference 3% margin; 208x48 or type1a 192x112 split into two rows",
            "note": "Same project validation used for model selection; not an independent final test.",
        },
        "counts": {"plates": len(samples), "ocr_rows": len(tasks)},
        "latency": {
            "total_seconds": elapsed,
            "mean_ms_per_plate_batched": 1000 * elapsed / max(1, len(samples)),
            "batch": args.batch,
            "hardware": "local CPU",
        },
        "thresholds": {},
    }
    for threshold in (0.0, 0.35, 0.45):
        rows = [row for row in all_rows if row["threshold"] == threshold]
        summary["thresholds"][str(threshold)] = summarize(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output / "predictions.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
