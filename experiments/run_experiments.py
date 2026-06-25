#!/usr/bin/env python3
"""
run_experiments.py — batch evaluation over the COCO-128 dataset.

Runs the full SVD+YOLO steganography pipeline on every image for two operating
points (High-Capacity and Robust), under a battery of attacks, and writes:

    results/metrics_per_image.csv   one row per image (all metrics, both modes)
    results/summary.json            aggregated statistics (means / medians)
    results/secret_compression.csv  truncated-SVD codec quality vs rank k

Usage
-----
    python experiments/run_experiments.py                  # all 128 images
    python experiments/run_experiments.py --limit 100      # first 100 images
    python experiments/run_experiments.py --no-yolo        # saliency fallback
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dataset, steganography as steg, yolo_guidance as yg
from src import pipeline as P
from src.pipeline import hc_config, rb_config

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _prefix(d: dict, pref: str, drop=("_arrays",), keep_shared=()) -> dict:
    """Prefix mode-specific keys; pass through a few shared keys unprefixed."""
    out = {}
    for key, val in d.items():
        if key in drop:
            continue
        if key in keep_shared:
            out[key] = val
        else:
            out[f"{pref}_{key}"] = val
    return out


def secret_compression_table(secret_size: int = 64,
                             ks=(2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32)) -> pd.DataFrame:
    """Truncated-SVD codec quality vs rank k (Eckart–Young / compression)."""
    secret = dataset.default_secret(secret_size)
    rows = [steg.secret_compression_report(secret, k)
            for k in ks if k <= secret_size]
    return pd.DataFrame(rows)


def run(limit: int | None, weights: str, use_yolo: bool, seed: int) -> None:
    np.random.seed(seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)

    paths = dataset.list_images(limit=limit)
    print(f"[experiments] {len(paths)} COCO-128 images | YOLO={'on' if use_yolo else 'off'}")

    detector = yg.YOLODetector(weights) if use_yolo else None
    if detector is not None and not detector.available:
        print("[experiments] YOLO weights unavailable → spectral-residual fallback")

    hc, rb = hc_config(), rb_config()
    secret_hc = dataset.default_secret(hc.secret_size)
    secret_rb = dataset.default_secret(rb.secret_size)

    rows, t0 = [], time.time()
    for path in tqdm(paths, desc="images", ncols=80):
        try:
            cover = dataset.load_image(path, hc.cover_size)
            saliency, cover_dets = yg.priority_map(cover, detector)
            pre = (saliency, cover_dets)

            row = {"image": path.name}
            # High-Capacity: imperceptibility + downstream + clean recovery + attacks
            out_hc = P.process_image(cover, secret_hc, detector, hc,
                                     run_baseline=True, run_attacks=True,
                                     run_downstream=True, precomputed=pre)
            row.update(_prefix(out_hc, "hc",
                               keep_shared=("n_cover_detections", "yolo_available")))
            # Robust: recovery + attacks (no baseline/downstream needed)
            out_rb = P.process_image(cover, secret_rb, detector, rb,
                                     run_baseline=False, run_attacks=True,
                                     run_downstream=False, precomputed=pre)
            row.update(_prefix(out_rb, "rb"))
            rows.append(row)
        except Exception as e:  # keep going on a bad image
            print(f"  [warn] {path.name}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS / "metrics_per_image.csv"
    df.to_csv(csv_path, index=False)
    print(f"[experiments] wrote {csv_path}  ({len(df)} rows, {df.shape[1]} cols)")

    # secret-compression analysis (independent of covers)
    sc = secret_compression_table(hc.secret_size)
    sc.to_csv(RESULTS / "secret_compression.csv", index=False)

    summary = build_summary(df, sc, hc, rb, n_images=len(df),
                            elapsed=time.time() - t0,
                            yolo=bool(detector and detector.available))
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)


# ---------------------------------------------------------------------------
def _stat(df, col):
    if col not in df:
        return None
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    return {"mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}


def build_summary(df, sc, hc, rb, n_images, elapsed, yolo) -> dict:
    attacks = P.PipelineConfig().attacks
    summ = {
        "n_images": int(n_images),
        "elapsed_sec": round(elapsed, 1),
        "yolo_available": yolo,
        "config": {
            "hc": {"block": hc.embed.block, "delta": hc.embed.delta,
                   "repeat": hc.embed.repeat, "channels": list(hc.embed.channels),
                   "secret_size": hc.secret_size, "k": hc.k},
            "rb": {"block": rb.embed.block, "delta": rb.embed.delta,
                   "repeat": rb.embed.repeat, "channels": list(rb.embed.channels),
                   "secret_size": rb.secret_size, "k": rb.k},
        },
        "hc": {}, "rb": {}, "attacks": {"hc": {}, "rb": {}},
    }
    hc_cols = ["hc_guided_psnr", "hc_guided_ssim", "hc_guided_mse",
               "hc_guided_obj_psnr", "hc_guided_bg_psnr",
               "hc_baseline_psnr", "hc_baseline_obj_psnr", "hc_baseline_bg_psnr",
               "hc_guided_secret_nc", "hc_guided_secret_psnr", "hc_guided_ber",
               "hc_baseline_secret_nc", "hc_capacity_bpp", "hc_payload_bits",
               "hc_det_preservation_rate", "hc_det_map50", "hc_det_mean_iou",
               "hc_det_mean_conf_delta", "hc_baseline_det_preservation_rate",
               "hc_baseline_det_map50", "n_cover_detections"]
    for c in hc_cols:
        st = _stat(df, c)
        if st:
            summ["hc"][c.replace("hc_", "", 1)] = st
    rb_cols = ["rb_guided_psnr", "rb_guided_ssim", "rb_guided_secret_nc",
               "rb_guided_secret_psnr", "rb_guided_ber", "rb_capacity_bpp",
               "rb_payload_bits"]
    for c in rb_cols:
        st = _stat(df, c)
        if st:
            summ["rb"][c.replace("rb_", "", 1)] = st
    for a in attacks:
        for mode in ("hc", "rb"):
            entry = {}
            for metric in ("ber", "nc"):
                st = _stat(df, f"{mode}_atk_{a}_{metric}")
                if st:
                    entry[metric] = st["mean"]
            if entry:
                summ["attacks"][mode][a] = entry
    # codec curve
    summ["secret_compression"] = sc.to_dict(orient="records")
    return summ


def print_summary(s: dict) -> None:
    print("\n" + "=" * 64)
    print(f"  SUMMARY over {s['n_images']} images  "
          f"(YOLO={'on' if s['yolo_available'] else 'fallback'}, "
          f"{s['elapsed_sec']}s)")
    print("=" * 64)

    def g(mode, key, sub="mean"):
        return s[mode].get(key, {}).get(sub, float("nan"))

    print("\n[High-Capacity mode]")
    print(f"  cover→stego PSNR     : {g('hc','guided_psnr'):.2f} dB   "
          f"SSIM {g('hc','guided_ssim'):.4f}")
    print(f"  object PSNR  guided  : {g('hc','guided_obj_psnr'):.2f} dB   "
          f"baseline {g('hc','baseline_obj_psnr'):.2f} dB")
    print(f"  backgr PSNR  guided  : {g('hc','guided_bg_psnr'):.2f} dB   "
          f"baseline {g('hc','baseline_bg_psnr'):.2f} dB")
    print(f"  clean secret NC      : {g('hc','guided_secret_nc'):.3f}   "
          f"BER {g('hc','guided_ber'):.5f}   bpp {g('hc','capacity_bpp'):.4f}")
    print(f"  detection preserve   : rate {g('hc','det_preservation_rate'):.3f}   "
          f"mAP@0.5 {g('hc','det_map50'):.3f}   |Δconf| {g('hc','det_mean_conf_delta'):.3f}")

    print("\n[Robust mode]")
    print(f"  cover→stego PSNR     : {g('rb','guided_psnr'):.2f} dB   "
          f"SSIM {g('rb','guided_ssim'):.4f}")
    print(f"  clean secret NC      : {g('rb','guided_secret_nc'):.3f}   "
          f"BER {g('rb','guided_ber'):.5f}")

    print("\n[Robustness — mean NC of recovered secret under attack]")
    print(f"  {'attack':>14} {'HC-NC':>8} {'RB-NC':>8} {'HC-BER':>8} {'RB-BER':>8}")
    for a in s["attacks"]["hc"]:
        hc = s["attacks"]["hc"].get(a, {})
        rb = s["attacks"]["rb"].get(a, {})
        print(f"  {a:>14} {hc.get('nc',float('nan')):>8.3f} "
              f"{rb.get('nc',float('nan')):>8.3f} {hc.get('ber',float('nan')):>8.3f} "
              f"{rb.get('ber',float('nan')):>8.3f}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="max images (default all 128)")
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--no-yolo", action="store_true", help="use saliency fallback")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.limit, args.weights, not args.no_yolo, args.seed)
