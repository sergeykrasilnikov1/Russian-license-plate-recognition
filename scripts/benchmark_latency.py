#!/usr/bin/env python3
"""Compare OCR backends and time detect+warp+OCR vs the 100 ms budget.

OCR accuracy uses GT quads from dataset/meta.csv (fair crop, independent of
the 34-epoch detector stub). End-to-end latency uses the detector when a
runtime exists (ultralytics .pt or weights/detector.onnx); otherwise warp+OCR
on GT quads is reported separately.

--n 0 loads every readable GT crop that warps successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from src.detection.infer import DetectorUnavailable, load_detector
from src.ocr.backends import BACKENDS, BackendUnavailable, create_backend
from src.ocr.charset import finalize_plate
from src.ocr.metrics import char_accuracy, full_plate_match, gt_is_readable, wilson_interval
from src.ocr.warp import parse_quad, warp_plate
from src.pipeline.infer import GrzPipeline, load_yaml, iter_images
from src.pipeline.reference import ReferencePoseCrnnPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark GRZ OCR backends and pipeline latency")
    p.add_argument("--input", type=Path, default=ROOT / "dataset" / "images", help="Image directory for end-to-end timing")
    p.add_argument(
        "--n",
        type=int,
        default=400,
        help="Max successfully warped readable crops (0 = all readable in meta.csv)",
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "pipeline.yaml")
    p.add_argument("--ocr-config", type=Path, default=ROOT / "configs" / "ocr.yaml")
    p.add_argument("--meta", type=Path, default=ROOT / "dataset" / "meta.csv")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    p.add_argument("--split", type=Path, default=ROOT / "dataset/splits/val.txt", help="Held-out image list for OCR evaluation")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--backends", nargs="*", default=list(BACKENDS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latency-only", action="store_true", help="Time the configured pipeline without GT data or OCR comparison")
    p.add_argument("--prefer-onnx", action="store_true", help="Benchmark the legacy ONNX fallback")
    return p.parse_args()


def benchmark_pipeline(args, ocr_cfg: dict) -> int:
    pipe_cfg = load_yaml(args.config)
    paths = iter_images(args.input, pipe_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    if not paths:
        print(f"No images in {args.input}")
        return 1
    if args.n > 0:
        paths = paths[:args.n + max(0, args.warmup)]
    try:
        if pipe_cfg.get("engine") == "reference_pose_crnn" and not args.prefer_onnx:
            pipeline = ReferencePoseCrnnPipeline(pipe_cfg)
        else:
            pipeline = GrzPipeline(pipe_cfg, ocr_cfg, prefer_onnx=args.prefer_onnx)
        elapsed = []
        processed = 0
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            started = time.perf_counter()
            pipeline.process_image(image, path.name)
            duration = (time.perf_counter() - started) * 1000
            if processed >= max(0, args.warmup):
                elapsed.append(duration)
            processed += 1
        if not elapsed:
            print("No images timed after warmup; reduce --warmup")
            return 1
        budget = float(pipe_cfg.get("latency_budget_ms", 100))
        mean = float(np.mean(elapsed))
        print(f"engine={type(pipeline).__name__} n={len(elapsed)} "
              f"mean={mean:.2f} median={np.median(elapsed):.2f} "
              f"p95={np.percentile(elapsed,95):.2f} ms/image "
              f"budget={budget:.0f} fits={mean <= budget}")
        return 0
    except (DetectorUnavailable, BackendUnavailable) as exc:
        print(f"runtime unavailable: {exc}")
        return 1


def _readable_pool(meta: Path) -> pd.DataFrame:
    df = pd.read_csv(meta, sep=";")
    df["plate_num"] = df["plate_num"].astype(str)
    return df[df["plate_num"].map(gt_is_readable)].copy()


def _stratified_order(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle within (plate_type, is_synthetic) so a prefix stays mixed."""
    rng = np.random.default_rng(seed)
    parts = []
    for (ptype, syn), g in df.groupby(["plate_type", "is_synthetic"], sort=False):
        idx = rng.permutation(g.index.to_numpy())
        parts.append(g.loc[idx])
    # Round-robin merge so early prefixes contain every stratum.
    buckets = [p.reset_index(drop=True) for p in parts if len(p)]
    rows = []
    i = 0
    while buckets:
        b = buckets[i % len(buckets)]
        if b.empty:
            buckets.pop(i % len(buckets))
            continue
        rows.append(b.iloc[0])
        buckets[i % len(buckets)] = b.iloc[1:].reset_index(drop=True)
        i += 1
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df.iloc[0:0]


def _load_crop(dataset: Path, row, type1a_mode: str, scale: float) -> np.ndarray | None:
    path = dataset / str(row["image"])
    image = cv2.imread(str(path))
    if image is None:
        return None
    try:
        quad = parse_quad(row["quad"])
    except ValueError:
        return None
    try:
        return warp_plate(image, quad, str(row["plate_type"]), type1a_mode=type1a_mode, scale=scale)
    except Exception:
        return None


def _time_recognize(backend, crops: list[np.ndarray], warmup: int) -> tuple[list[tuple[str, float, list[float]]], np.ndarray]:
    for i in range(min(warmup, len(crops))):
        backend.recognize(crops[i])
    times = np.empty(len(crops), dtype=np.float64)
    preds = []
    for i, crop in enumerate(crops):
        t0 = time.perf_counter()
        detailed = getattr(backend, "recognize_detailed", None)
        if detailed:
            text, scores = detailed(crop)
            conf = float(min(scores)) if scores else 0.0
        else:
            text, conf = backend.recognize(crop)
            scores = [conf]
        times[i] = (time.perf_counter() - t0) * 1000.0
        preds.append((text, conf, scores))
    return preds, times


def _providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _sanity_v4_v5(crop: np.ndarray) -> None:
    print("\n=== sanity: rec-head tensor on one crop (v4 vs v5 must differ) ===")
    prints = []
    for name in ("ppocrv4_mobile", "ppocrv5_mobile"):
        be = create_backend(name)
        fp = be.rec_logits_fingerprint(crop)
        text, conf = be.recognize(crop)
        prints.append((name, fp, text, conf))
        print(
            f"{name}: graph={fp.get('graph')!r} shape={fp.get('shape')} "
            f"sha256_16={fp.get('sha256_16')} decode={text!r}"
        )
    a, b = prints[0][1], prints[1][1]
    same_hash = a["sha256_16"] == b["sha256_16"]
    same_shape = a["shape"] == b["shape"]
    if same_hash:
        print("FAIL: identical rec tensors — backend switch is broken")
    else:
        print(f"ok: tensors differ (same_shape={same_shape})")
    crop_hash = hashlib.sha256(crop.tobytes()).hexdigest()[:12]
    print(f"crop sha256_12={crop_hash}  crop_shape={crop.shape}")


def main() -> int:
    args = parse_args()
    ocr_cfg = load_yaml(args.ocr_config)
    if args.latency_only:
        return benchmark_pipeline(args, ocr_cfg)
    type1a_mode = str(ocr_cfg.get("type1a_mode", "flatten"))
    scale = float(ocr_cfg.get("warp_scale", 0.8))
    min_char = float(ocr_cfg.get("min_char_conf", 0.35))
    parseq_w = ROOT / ocr_cfg["parseq_weights"] if ocr_cfg.get("parseq_weights") else None

    pool = _readable_pool(args.meta)
    split_paths = set()
    for line in args.split.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        split_paths.add(str(path.resolve()))
        split_paths.add(str((args.dataset / path).resolve()))
        if line.startswith("./"):
            split_paths.add(str((args.split.parent / path).resolve()))
    pool = pool[pool.image.map(lambda p: str((args.dataset / str(p)).resolve()) in split_paths)]
    pool = _stratified_order(pool, args.seed)
    target = len(pool) if args.n <= 0 else args.n
    crops: list[np.ndarray] = []
    gts: list[str] = []
    types: list[str] = []
    strata: list[str] = []
    skipped = 0
    for _, row in pool.iterrows():
        if len(crops) >= target:
            break
        crop = _load_crop(args.dataset, row, type1a_mode, scale)
        if crop is None:
            skipped += 1
            continue
        crops.append(crop)
        gts.append(str(row["plate_num"]))
        types.append(str(row["plate_type"]))
        strata.append(f"{row['plate_type']}/" + ("synthetic" if int(row["is_synthetic"]) else "real"))
    mix = pd.Series(strata).value_counts().to_dict() if strata else {}
    print(
        f"crops={len(crops)} skipped_warp={skipped} target={target} "
        f"readable_pool={len(pool)} type1a_mode={type1a_mode} mix={mix} providers={_providers()}"
    )
    if not crops:
        print("no crops loaded")
        return 1

    if {"ppocrv4_mobile", "ppocrv5_mobile"}.issubset(args.backends):
        _sanity_v4_v5(crops[0])

    print()
    print(
        f"{'backend':<22} {'n':>5} {'acc@char':>9} {'acc@full':>9} {'full 95% CI':>18} "
        f"{'mean':>8} {'median':>8} {'p95':>8}  status"
    )
    print("-" * 110)
    table = []
    evaluated = 0
    for name in args.backends:
        try:
            rec_w = ocr_cfg.get("rec_weights_by_backend", {}).get(name)
            if rec_w is None and name == "ppocrv4_mobile":
                rec_w = ocr_cfg.get("rec_weights")
            backend = create_backend(name, parseq_w, ROOT / rec_w if rec_w else None)
            preds, times = _time_recognize(backend, crops, args.warmup)
        except BackendUnavailable as exc:
            print(f"{name:<22} {'—':>5} {'—':>9} {'—':>9} {'—':>18} {'—':>8} {'—':>8} {'—':>8}  {exc}")
            table.append(name)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<22} {'—':>5} {'—':>9} {'—':>9} {'—':>18} {'—':>8} {'—':>8} {'—':>8}  {exc}")
            continue
        evaluated += 1
        readable_idx = [i for i, g in enumerate(gts) if gt_is_readable(g)]
        n = len(readable_idx)
        if n:
            char_scores = [
                char_accuracy(finalize_plate(preds[i][0], preds[i][2], min_char)[0], gts[i]) for i in readable_idx
            ]
            full_hits = [
                1.0 if full_plate_match(finalize_plate(preds[i][0], preds[i][2], min_char)[0], gts[i]) else 0.0
                for i in readable_idx
            ]
            acc_c = float(np.mean(char_scores))
            acc_f = float(np.mean(full_hits))
            _, lo, hi = wilson_interval(int(sum(full_hits)), n)
            ci = f"[{lo:.3f},{hi:.3f}]"
        else:
            acc_c = acc_f = float("nan")
            ci = "—"
        mean_ms = float(times.mean())
        med_ms = float(np.median(times))
        p95_ms = float(np.percentile(times, 95))
        print(
            f"{name:<22} {n:5d} {acc_c:9.3f} {acc_f:9.3f} {ci:>18} "
            f"{mean_ms:8.2f} {med_ms:8.2f} {p95_ms:8.2f}  ok"
        )
        for ptype in sorted(set(strata)):
            indices = [i for i, t in enumerate(strata) if t == ptype]
            hits = sum(full_plate_match(finalize_plate(preds[i][0], preds[i][2], min_char)[0], gts[i]) for i in indices)
            print(f"  {ptype}: n={len(indices)} exact={hits / len(indices):.3f}")
        table.append((name, n, acc_c, acc_f, mean_ms, med_ms))

    print("\n=== end-to-end (detect + warp + OCR) ===")
    latency_status = benchmark_pipeline(args, ocr_cfg)

    if "parseq" in args.backends and not (parseq_w and parseq_w.is_file()):
        print("PARSeq was not evaluated: local weights are missing.")
    return 0 if evaluated and latency_status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
