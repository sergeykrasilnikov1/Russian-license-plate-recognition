"""Leakage-aware full detector -> warp -> OCR evaluation on Kaggle CPU."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
IOU_THRESHOLD = 0.5
MIN_CHAR_CONF = 0.35


def find_root(marker: str, predicate=None) -> Path:
    matches = []
    for marker_path in INPUT.glob(f"**/{marker}"):
        root = marker_path.parent
        if predicate is None or predicate(root):
            matches.append(root)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected one root for {marker!r}, got {unique}")
    return unique[0]


if INPUT.is_dir():
    asset_markers = list(INPUT.glob("**/weights/ocr/en_PP-OCRv4_rec_mobile.onnx"))
    if len(asset_markers) != 1:
        raise RuntimeError(f"expected one OCR asset root, got {asset_markers}")
    ASSETS = asset_markers[0].parents[2]
    detector_weights = list(INPUT.glob("**/yolo26n_best.onnx"))
    if len(detector_weights) != 1:
        raise RuntimeError(f"expected one yolo26n_best.onnx kernel output, got {detector_weights}")
    DETECTOR_WEIGHTS = detector_weights[0]
    DATA = find_root("meta.csv", lambda p: (p / "splits/val.txt").is_file())
    V3 = find_root("data.yaml", lambda p: (p / "train/images").is_dir() and (p / "test/images").is_dir())
else:
    ASSETS = PROJECT_ROOT
    DETECTOR_WEIGHTS = PROJECT_ROOT / "weights/detector_yolo26n.onnx"
    DATA = PROJECT_ROOT / "kaggle_upload/grz-plates-detector"
    V3 = PROJECT_ROOT / "kaggle_upload/grz-plates-v3"
    OUTPUT = PROJECT_ROOT / "kaggle_results/pipeline_e2e_local"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ASSETS))

from src.ocr.backends.ppocr import PPOCRV4Backend, PPOCRV5Backend
from src.ocr.charset import finalize_plate
from src.ocr.metrics import char_accuracy, full_plate_match, gt_is_readable, wilson_interval
from src.ocr.warp import DegenerateQuadError, parse_quad, warp_plate, xyxy_to_quad
from src.utils.plate_mask import looks_like_special_plate


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    plate_type: str


def parse_model_names(raw) -> list[str] | None:
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, dict):
        indexed = {int(key): str(name) for key, name in value.items()}
        return [indexed[index] for index in range(len(indexed))]
    if isinstance(value, (list, tuple)):
        return [str(name) for name in value]
    return None


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if not len(boxes):
        return []
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0).clip(min=0) * (y1 - y0).clip(min=0)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        xx0 = np.maximum(x0[index], x0[order[1:]])
        yy0 = np.maximum(y0[index], y0[order[1:]])
        xx1 = np.minimum(x1[index], x1[order[1:]])
        yy1 = np.minimum(y1[index], y1[order[1:]])
        intersection = (xx1 - xx0).clip(min=0) * (yy1 - yy0).clip(min=0)
        overlap = intersection / (areas[index] + areas[order[1:]] - intersection + 1e-6)
        order = order[1:][overlap <= threshold]
    return keep


class OnnxYoloDetector:
    def __init__(self, weights: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.sess = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        metadata = self.sess.get_modelmeta().custom_metadata_map or {}
        self.names = parse_model_names(metadata.get("names")) or ["type1", "type1a", "type1b", "other"]
        expected = {"type1", "type1a", "type1b", "other"}
        if set(self.names) != expected:
            raise RuntimeError(f"unexpected detector classes: {self.names}")

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        scale = min(640 / height, 640 / width)
        new_h, new_w = int(round(height * scale)), int(round(width * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x = int(round((640 - new_w) / 2 - 0.1))
        pad_y = int(round((640 - new_h) / 2 - 0.1))
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        prediction = np.squeeze(self.sess.run(None, {self.input_name: blob})[0], axis=0)
        if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 12:
            prediction = prediction.T
        xywh = prediction[:, :4]
        class_scores = prediction[:, 4:4 + len(self.names)]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        mask = scores >= 0.25
        xywh, scores, class_ids = xywh[mask], scores[mask], class_ids[mask]
        x, y, box_w, box_h = xywh.T
        boxes = np.stack(
            [
                (x - box_w / 2 - pad_x) / scale,
                (y - box_h / 2 - pad_y) / scale,
                (x + box_w / 2 - pad_x) / scale,
                (y + box_h / 2 - pad_y) / scale,
            ],
            axis=1,
        )
        detections = []
        for index in nms(boxes, scores, 0.45)[:10]:
            x0, y0, x1, y1 = [float(value) for value in boxes[index]]
            box = (max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1))
            if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                detections.append(Detection(box, float(scores[index]), self.names[int(class_ids[index])]))
        return detections


def vehicle_context_score(image_bgr: np.ndarray, bbox) -> float:
    height, width = image_bgr.shape[:2]
    if width < 8 or height < 8:
        return 0.0
    x0, y0, x1, y1 = bbox
    box_w, box_h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    relative_width = box_w / width
    center_y = (y0 + y1) / 2 / height
    size_score = 1.0
    if relative_width < 0.04:
        size_score = relative_width / 0.04
    elif relative_width > 0.55:
        size_score = max(0.0, 1.0 - (relative_width - 0.55) / 0.55)
    position_score = 1.0
    if center_y < 0.15:
        position_score = center_y / 0.15
    elif center_y > 0.92:
        position_score = max(0.0, 1.0 - (center_y - 0.92) / 0.08)
    band_x0 = int(max(0, min(width - 1, x0 - 0.5 * box_w)))
    band_y0 = int(max(0, min(height - 1, y1)))
    band_x1 = int(max(0, min(width, x1 + 0.5 * box_w)))
    band_y1 = int(max(0, min(height, y1 + max(box_h * 1.2, height * 0.08))))
    band = image_bgr[band_y0:band_y1, band_x0:band_x1]
    if not band.size:
        return 0.0
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    darkness = 1.0 - min(1.0, float(gray.mean()) / 220.0)
    texture = min(1.0, float(gray.std()) / 40.0)
    below_score = 0.55 * darkness + 0.45 * texture
    if below_score < 0.1:
        return 0.0
    return float(np.clip(0.35 * size_score + 0.25 * position_score + 0.40 * below_score, 0.0, 1.0))


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def relative_image(line: str) -> str:
    path = Path(line.strip())
    if not path.is_absolute():
        return path.as_posix().removeprefix("./")
    if path.parts.count("images") == 1:
        return Path(*path.parts[path.parts.index("images"):]).as_posix()
    return path.as_posix()


def read_split(name: str) -> set[str]:
    path = DATA / "splits" / f"{name}.txt"
    return {relative_image(line) for line in path.read_text().splitlines() if line.strip()}


def parse_bbox(raw: str, quad: np.ndarray) -> tuple[float, float, float, float]:
    values = [float(value) for value in str(raw).replace(";", ",").split(",") if value.strip()]
    if len(values) == 4:
        x, y, width, height = values
        return x, y, x + width, y + height
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def match_one_to_one(gt_rows: list[dict], pred_rows: list[dict]) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for gi, gt in enumerate(gt_rows):
        for pi, pred in enumerate(pred_rows):
            overlap = iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if overlap >= IOU_THRESHOLD:
                candidates.append((overlap, gi, pi))
    matched_gt, matched_pred, matches = set(), set(), []
    for overlap, gi, pi in sorted(candidates, reverse=True):
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, overlap))
    return matches, matched_gt, matched_pred


def group_name(row: dict) -> str:
    return f"{row['plate_type']}/" + ("synthetic" if row["is_synthetic"] else "real")


def summarize_ocr(rows: list[dict]) -> dict:
    result = {}
    groups = ["all"] + sorted({group_name(row) for row in rows})
    for group in groups:
        subset = rows if group == "all" else [row for row in rows if group_name(row) == group]
        hits = sum(bool(row["exact"]) for row in subset)
        p, lo, hi = wilson_interval(hits, len(subset))
        result[group] = {
            "n": len(subset),
            "exact": p,
            "exact_ci95": [lo, hi],
            "char_accuracy": float(np.mean([row["char_accuracy"] for row in subset])) if subset else None,
        }
    return result


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


meta = pd.read_csv(DATA / "meta.csv", sep=";", dtype={"plate_num": str})
train_paths, val_paths = read_split("train"), read_split("val")

print("Hashing OCR train and detector source datasets...", flush=True)
ocr_train_hashes = {
    sha256(DATA / rel)
    for rel in sorted(train_paths)
    if (DATA / rel).is_file()
}
detector_source_images = sorted(
    path
    for split_name in ("train", "valid", "test")
    for path in (V3 / split_name / "images").glob("*")
    if path.is_file()
)
detector_source_hashes = {sha256(path) for path in detector_source_images}

val_meta = meta[meta["image"].astype(str).isin(val_paths)].copy()
candidate_paths = sorted({str(rel) for rel in val_meta["image"] if (DATA / str(rel)).is_file()})
selected_paths, selected_hashes = [], set()
excluded_ocr_overlap, excluded_detector_overlap, excluded_eval_duplicate = [], [], []
path_hashes = {}
for rel in candidate_paths:
    digest = sha256(DATA / rel)
    path_hashes[rel] = digest
    if digest in ocr_train_hashes:
        excluded_ocr_overlap.append(rel)
    elif digest in detector_source_hashes:
        excluded_detector_overlap.append(rel)
    elif digest in selected_hashes:
        excluded_eval_duplicate.append(rel)
    else:
        selected_paths.append(rel)
        selected_hashes.add(digest)

selected = val_meta[val_meta["image"].astype(str).isin(selected_paths)].copy()
gt_by_image: dict[str, list[dict]] = defaultdict(list)
skipped_gt = []
for _, row in selected.iterrows():
    try:
        quad = parse_quad(str(row.quad))
        bbox = parse_bbox(str(row.bbox), quad)
    except Exception as exc:
        skipped_gt.append({"image": str(row.image), "reason": f"{type(exc).__name__}: {exc}"})
        continue
    gt_by_image[str(row.image)].append(
        {
            "image": str(row.image),
            "plate_num": str(row.plate_num),
            "plate_type": str(row.plate_type),
            "is_synthetic": bool(int(row.is_synthetic)),
            "quad": quad,
            "bbox_xyxy": bbox,
        }
    )

detector = OnnxYoloDetector(DETECTOR_WEIGHTS)
v4_backend = PPOCRV4Backend(ASSETS / "weights/ocr/en_PP-OCRv4_rec_mobile.onnx")
v5_backend = PPOCRV5Backend(ASSETS / "weights/ocr/en_PP-OCRv5_rec_mobile.onnx")


def routed_backend(plate_type: str):
    return v5_backend if plate_type == "type1a" else v4_backend


def recognize(crop: np.ndarray, plate_type: str, policy: str) -> tuple[str, float, float]:
    started = time.perf_counter()
    if policy == "routed":
        raw, scores = routed_backend(plate_type).recognize_detailed(crop)
    elif policy == "ppocrv5_mobile":
        raw, scores = v5_backend.recognize_detailed(crop)
    else:
        raise KeyError(policy)
    elapsed = (time.perf_counter() - started) * 1000
    final, confidence = finalize_plate(raw, scores, MIN_CHAR_CONF)
    return final, confidence, elapsed


# Warm sessions before measuring.
first_image_rel = next(iter(gt_by_image))
first_image = cv2.imread(str(DATA / first_image_rel))
if first_image is None:
    raise RuntimeError(f"cannot read warmup image {first_image_rel}")
for _ in range(3):
    detector.predict(first_image)
warm_gt = next(row for rows in gt_by_image.values() for row in rows if gt_is_readable(row["plate_num"]))
warm_image = cv2.imread(str(DATA / warm_gt["image"]))
warm_crop = warp_plate(warm_image, warm_gt["quad"], warm_gt["plate_type"], type1a_mode="flatten", scale=0.8)
for policy in ("routed", "ppocrv5_mobile"):
    recognize(warm_crop, warm_gt["plate_type"], policy)

print(f"Evaluating oracle geometry on {sum(len(rows) for rows in gt_by_image.values())} GT rows...", flush=True)
oracle_rows = []
oracle_latency: dict[str, list[float]] = defaultdict(list)
for rel, rows in gt_by_image.items():
    image = cv2.imread(str(DATA / rel))
    if image is None:
        continue
    for gt in rows:
        if not gt_is_readable(gt["plate_num"]):
            continue
        for geometry, points in (("gt_quad", gt["quad"]), ("gt_bbox", xyxy_to_quad(gt["bbox_xyxy"]))):
            try:
                crop = warp_plate(image, points, gt["plate_type"], type1a_mode="flatten", scale=0.8)
            except DegenerateQuadError:
                continue
            for policy in ("routed", "ppocrv5_mobile"):
                prediction, confidence, elapsed = recognize(crop, gt["plate_type"], policy)
                oracle_latency[f"{geometry}/{policy}"].append(elapsed)
                oracle_rows.append(
                    {
                        **{key: gt[key] for key in ("image", "plate_num", "plate_type", "is_synthetic")},
                        "geometry": geometry,
                        "policy": policy,
                        "prediction": prediction,
                        "confidence": confidence,
                        "char_accuracy": char_accuracy(prediction, gt["plate_num"]),
                        "exact": full_plate_match(prediction, gt["plate_num"]),
                    }
                )

oracle_summary = {}
for geometry in ("gt_quad", "gt_bbox"):
    oracle_summary[geometry] = {}
    for policy in ("routed", "ppocrv5_mobile"):
        subset = [row for row in oracle_rows if row["geometry"] == geometry and row["policy"] == policy]
        oracle_summary[geometry][policy] = {
            "groups": summarize_ocr(subset),
            "latency_ms": latency_summary(oracle_latency[f"{geometry}/{policy}"]),
        }

paired = {}
for policy in ("routed", "ppocrv5_mobile"):
    by_key = defaultdict(dict)
    for row in oracle_rows:
        if row["policy"] == policy:
            by_key[(row["image"], row["plate_num"], row["plate_type"])][row["geometry"]] = bool(row["exact"])
    counts = {"both": 0, "quad_only": 0, "bbox_only": 0, "neither": 0}
    for pair in by_key.values():
        if set(pair) != {"gt_quad", "gt_bbox"}:
            continue
        if pair["gt_quad"] and pair["gt_bbox"]:
            counts["both"] += 1
        elif pair["gt_quad"]:
            counts["quad_only"] += 1
        elif pair["gt_bbox"]:
            counts["bbox_only"] += 1
        else:
            counts["neither"] += 1
    paired[policy] = counts

print(f"Evaluating detector pipeline on {len(gt_by_image)} images...", flush=True)
predictions_by_image: dict[str, list[dict]] = {}
detector_latency, total_latency = [], []
prediction_rows = []
for index, (rel, gt_rows) in enumerate(sorted(gt_by_image.items()), 1):
    image = cv2.imread(str(DATA / rel))
    if image is None:
        continue
    image_started = time.perf_counter()
    started = time.perf_counter()
    detections = detector.predict(image)
    detector_latency.append((time.perf_counter() - started) * 1000)
    preds = []
    for detection in detections:
        try:
            crop = warp_plate(
                image,
                xyxy_to_quad(detection.xyxy),
                detection.plate_type,
                type1a_mode="flatten",
                scale=0.8,
            )
        except DegenerateQuadError:
            continue
        vehicle_score = vehicle_context_score(image, detection.xyxy)
        passes_vehicle = vehicle_score >= 0.3
        policy_predictions = {}
        for policy in ("routed", "ppocrv5_mobile"):
            plate, ocr_conf, elapsed = recognize(crop, detection.plate_type, policy)
            output_type = "other" if looks_like_special_plate(plate) else detection.plate_type
            policy_predictions[policy] = {
                "plate_num": plate,
                "plate_type": output_type,
                "ocr_conf": ocr_conf,
                "confidence": min(float(detection.conf), float(ocr_conf)),
                "ocr_ms": elapsed,
            }
        pred = {
            "image": rel,
            "bbox_xyxy": detection.xyxy,
            "detector_type": detection.plate_type,
            "detector_conf": detection.conf,
            "passes_vehicle": bool(passes_vehicle),
            "vehicle_score": float(vehicle_score),
            "policies": policy_predictions,
        }
        preds.append(pred)
        for policy, value in policy_predictions.items():
            prediction_rows.append(
                {
                    "image": rel,
                    "policy": policy,
                    "x0": detection.xyxy[0],
                    "y0": detection.xyxy[1],
                    "x1": detection.xyxy[2],
                    "y1": detection.xyxy[3],
                    "detector_type": detection.plate_type,
                    "output_type": value["plate_type"],
                    "prediction": value["plate_num"],
                    "detector_conf": detection.conf,
                    "ocr_conf": value["ocr_conf"],
                    "confidence": value["confidence"],
                    "passes_vehicle": passes_vehicle,
                    "vehicle_score": vehicle_score,
                    "ocr_ms": value["ocr_ms"],
                }
            )
    predictions_by_image[rel] = preds
    total_latency.append((time.perf_counter() - image_started) * 1000)
    if index % 100 == 0:
        print(f"  {index}/{len(gt_by_image)}", flush=True)


def summarize_e2e(vehicle_filter: bool, policy: str) -> dict:
    total_gt = total_pred = total_matches = type_hits = output_type_hits = 0
    readable_gt = matched_readable = exact_matched = char_sum = 0
    per_group = defaultdict(lambda: {"n": 0, "detected": 0, "exact": 0})
    false_positives = 0
    match_rows = []
    for rel, gt_rows in gt_by_image.items():
        preds = predictions_by_image.get(rel, [])
        if vehicle_filter:
            preds = [pred for pred in preds if pred["passes_vehicle"]]
        matches, matched_gt, matched_pred = match_one_to_one(gt_rows, preds)
        total_gt += len(gt_rows)
        total_pred += len(preds)
        total_matches += len(matches)
        false_positives += len(preds) - len(matched_pred)
        for gt in gt_rows:
            if gt_is_readable(gt["plate_num"]):
                readable_gt += 1
                per_group[group_name(gt)]["n"] += 1
        for gi, pi, overlap in matches:
            gt, pred = gt_rows[gi], preds[pi]
            value = pred["policies"][policy]
            type_hits += pred["detector_type"] == gt["plate_type"]
            output_type_hits += value["plate_type"] == gt["plate_type"]
            if gt_is_readable(gt["plate_num"]):
                matched_readable += 1
                per_group[group_name(gt)]["detected"] += 1
                exact = full_plate_match(value["plate_num"], gt["plate_num"])
                exact_matched += exact
                per_group[group_name(gt)]["exact"] += exact
                char_value = char_accuracy(value["plate_num"], gt["plate_num"])
                char_sum += char_value
                match_rows.append(
                    {
                        "image": rel,
                        "policy": policy,
                        "vehicle_filter": vehicle_filter,
                        "ground_truth": gt["plate_num"],
                        "gt_type": gt["plate_type"],
                        "domain": "synthetic" if gt["is_synthetic"] else "real",
                        "prediction": value["plate_num"],
                        "predicted_type": value["plate_type"],
                        "iou": overlap,
                        "exact": exact,
                        "char_accuracy": char_value,
                    }
                )
    e2e_p, e2e_lo, e2e_hi = wilson_interval(exact_matched, readable_gt)
    matched_p, matched_lo, matched_hi = wilson_interval(exact_matched, matched_readable)
    groups = {}
    for name, values in sorted(per_group.items()):
        p, lo, hi = wilson_interval(values["exact"], values["n"])
        groups[name] = {
            **values,
            "detection_recall": values["detected"] / values["n"] if values["n"] else None,
            "e2e_exact": p,
            "e2e_exact_ci95": [lo, hi],
        }
    return {
        "vehicle_filter": vehicle_filter,
        "policy": policy,
        "gt_objects": total_gt,
        "predictions": total_pred,
        "matches_iou_0_5": total_matches,
        "false_positives": false_positives,
        "detection_recall": total_matches / total_gt if total_gt else None,
        "detection_precision": total_matches / total_pred if total_pred else None,
        "detector_type_accuracy_on_matches": type_hits / total_matches if total_matches else None,
        "output_type_accuracy_on_matches": output_type_hits / total_matches if total_matches else None,
        "readable_gt": readable_gt,
        "matched_readable": matched_readable,
        "ocr_exact_given_match": matched_p,
        "ocr_exact_given_match_ci95": [matched_lo, matched_hi],
        "char_accuracy_given_match": char_sum / matched_readable if matched_readable else None,
        "end_to_end_exact": e2e_p,
        "end_to_end_exact_ci95": [e2e_lo, e2e_hi],
        "groups": groups,
        "match_rows": match_rows,
    }


e2e_summary = {}
all_match_rows = []
for filter_name, enabled in (("unfiltered", False), ("vehicle_filtered", True)):
    e2e_summary[filter_name] = {}
    for policy in ("routed", "ppocrv5_mobile"):
        result = summarize_e2e(enabled, policy)
        all_match_rows.extend(result.pop("match_rows"))
        e2e_summary[filter_name][policy] = result

smoke_paths = set(sorted(gt_by_image)[:20])
smoke_rows = [
    row for row in prediction_rows
    if row["image"] in smoke_paths and row["policy"] == "routed" and row["passes_vehicle"]
]
with (OUTPUT / "pipeline_smoke.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, delimiter=";", lineterminator="\n")
    writer.writerow(["image", "plate_num", "plate_type", "confidence"])
    for row in smoke_rows:
        writer.writerow([row["image"], row["prediction"], row["output_type"], f"{row['confidence']:.4f}"])
header = (OUTPUT / "pipeline_smoke.csv").read_text(encoding="utf-8").splitlines()[0]
if header != "image;plate_num;plate_type;confidence":
    raise AssertionError(f"unexpected CSV header: {header}")

summary = {
    "protocol": {
        "dataset": "sergeniy/grz-plates-detector",
        "requested_split": "val",
        "detector": "YOLO26n ONNX selected on Roboflow v3",
        "detector_overlap_rule": "exclude exact SHA-256 overlap with every Roboflow v3 train/valid/test image",
        "ocr_overlap_rule": "exclude exact SHA-256 overlap with OCR train split",
        "eval_duplicate_rule": "one evaluation path per exact image hash",
        "matching": "greedy one-to-one IoU >= 0.5",
        "ocr": "type route: type1=v4, type1a=v5, type1b=v4, other=v4; fallback=v5",
        "warp": "GT quad vs GT axis-aligned bbox; full pipeline uses predicted axis-aligned bbox",
        "min_char_conf": MIN_CHAR_CONF,
        "note": "Selection and evaluation use the same project validation split. This is diagnostic evidence, not a final independent test.",
    },
    "environment": {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "onnxruntime": version("onnxruntime"),
        "providers": detector.sess.get_providers(),
    },
    "corpus": {
        "val_metadata_rows": len(val_meta),
        "val_existing_unique_paths": len(candidate_paths),
        "selected_unique_images": len(gt_by_image),
        "selected_gt_rows": sum(len(rows) for rows in gt_by_image.values()),
        "selected_readable_gt_rows": sum(
            gt_is_readable(row["plate_num"]) for rows in gt_by_image.values() for row in rows
        ),
        "excluded_ocr_train_overlap": excluded_ocr_overlap,
        "excluded_detector_source_overlap": excluded_detector_overlap,
        "excluded_eval_content_duplicates": excluded_eval_duplicate,
        "skipped_gt": skipped_gt,
    },
    "oracle_geometry": oracle_summary,
    "oracle_geometry_paired": paired,
    "full_pipeline": e2e_summary,
    "latency_ms": {
        "detector_per_image": latency_summary(detector_latency),
        "detector_plus_both_ocr_policies_per_image": latency_summary(total_latency),
        "note": "The total deliberately runs routed and universal-v5 OCR; deployment runs only one policy and is faster.",
    },
    "runtime_smoke": {
        "images": len(smoke_paths),
        "output_rows": len(smoke_rows),
        "passed": True,
        "note": "The same detector, vehicle heuristic, bbox warp, type route, finalizer and confidence rule were executed.",
    },
    "csv_contract": {"header": header, "passed": True},
}

pd.DataFrame(oracle_rows).to_csv(OUTPUT / "oracle_geometry_predictions.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(OUTPUT / "full_pipeline_predictions.csv", index=False)
pd.DataFrame(all_match_rows).to_csv(OUTPUT / "full_pipeline_matches.csv", index=False)
(OUTPUT / "full_pipeline_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
