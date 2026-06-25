#!/usr/bin/env python3
"""
demo.py — single-image demonstration of the SVD+YOLO steganography pipeline.

Usage
-----
    python scripts/demo.py                       # first COCO-128 image, HC mode
    python scripts/demo.py --index 7 --mode rb   # robust mode
    python scripts/demo.py --cover path.jpg --attack jpeg_q50

Outputs a side-by-side figure in ``results/examples/`` and prints the metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dataset, steganography as steg, metrics as M, attacks as A
from src import yolo_guidance as yg
from src.pipeline import hc_config, rb_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=0, help="COCO-128 image index")
    ap.add_argument("--cover", type=str, default=None, help="custom cover image path")
    ap.add_argument("--mode", choices=["hc", "rb"], default="hc")
    ap.add_argument("--attack", type=str, default="none",
                    help="attack applied before extraction (see src/attacks.py)")
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--out", type=str, default="results/examples")
    args = ap.parse_args()

    cfg = hc_config() if args.mode == "hc" else rb_config()
    secret = dataset.default_secret(cfg.secret_size)

    if args.cover:
        cover = dataset.load_image(args.cover, cfg.cover_size)
        name = Path(args.cover).stem
    else:
        path = dataset.list_images()[args.index]
        cover = dataset.load_image(path, cfg.cover_size)
        name = path.stem

    det = yg.YOLODetector(args.weights)
    saliency, cover_dets = yg.priority_map(cover, det)

    # hide + reveal -------------------------------------------------------
    res, payload = steg.hide_secret(cover, secret, cfg.k, cfg.embed, saliency)
    stego = res.stego
    attacked = A.apply_attack(stego, args.attack)
    recovered = steg.reveal_secret(attacked, cfg.embed, saliency)

    fid = M.summarize_fidelity(cover, stego)
    print(f"\nImage: {name}   mode={args.mode}   attack={args.attack}")
    print(f"  YOLO available     : {det.available}  ({len(cover_dets)} detections)")
    print(f"  payload            : {res.num_bits} bits "
          f"({res.num_bits/8:.0f} B), capacity {res.capacity_bits} bits, "
          f"bpp={M.capacity_bpp(res.capacity_bits, *cover.shape[:2]):.4f}")
    print(f"  cover vs stego     : PSNR={fid['psnr']:.2f} dB  SSIM={fid['ssim']:.4f}")
    print(f"  recovered secret   : NC={M.normalized_correlation(secret, recovered):.3f}  "
          f"PSNR={M.psnr(secret, recovered):.2f} dB")
    if det.available:
        dp = yg.detection_preservation(cover_dets, det.detect(stego))
        print(f"  detection preserve : rate={dp['preservation_rate']:.2f}  "
              f"mAP@0.5={dp['map50']:.2f}  |Δconf|={dp['mean_conf_delta']:.3f}")

    # figure --------------------------------------------------------------
    Path(args.out).mkdir(parents=True, exist_ok=True)
    diff = np.abs(cover.astype(float) - stego.astype(float)).mean(axis=2)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    ax[0, 0].imshow(cover); ax[0, 0].set_title(f"Cover ({name})")
    ax[0, 1].imshow(stego); ax[0, 1].set_title(f"Stego  PSNR={fid['psnr']:.1f} dB")
    im = ax[0, 2].imshow(diff, cmap="inferno")
    ax[0, 2].set_title("|Cover − Stego| (mean over RGB)")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    ax[1, 0].imshow(saliency, cmap="viridis")
    ax[1, 0].set_title("YOLO saliency / priority map")
    ax[1, 1].imshow(secret, cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title(f"Secret {secret.shape[0]}×{secret.shape[1]} (rank-{cfg.k})")
    ax[1, 2].imshow(recovered, cmap="gray", vmin=0, vmax=255)
    ax[1, 2].set_title(f"Recovered  NC={M.normalized_correlation(secret, recovered):.2f}"
                       f"\n(attack: {args.attack})")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    out = Path(args.out) / f"demo_{name}_{args.mode}_{args.attack}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  figure saved       : {out}\n")


if __name__ == "__main__":
    main()
