#!/usr/bin/env python3
"""
make_figures.py — generate every figure used in the report and the slides.

Reads the experiment outputs in ``results/`` and writes PNGs to
``results/figures/``.  Run it after ``run_experiments.py``:

    python experiments/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dataset, svd_core, steganography as steg, yolo_guidance as yg
from src import metrics as M, attacks as A
from src.pipeline import hc_config, rb_config

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  saved {p.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. SVD spectrum + Eckart–Young (computed live on a sample COCO image)
# ---------------------------------------------------------------------------
def fig_svd_spectrum(sample_index=0):
    img = dataset.load_image(dataset.list_images()[sample_index], 512)
    gray = np.asarray(img).mean(axis=2)
    svd = svd_core.svd_decompose(gray)
    s = svd.s
    ks = np.arange(1, len(s) + 1)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].semilogy(ks, s, color="tab:blue")
    ax[0].set(title="Singular-value spectrum (grayscale cover)",
              xlabel="index i", ylabel=r"$\sigma_i$ (log scale)")
    ax[1].plot(ks, energy, color="tab:green")
    for thr in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(energy, thr) + 1)
        ax[1].axhline(thr, ls="--", color="gray", alpha=0.6)
        ax[1].annotate(f"{int(thr*100)}% → k={k}", (k, thr),
                       textcoords="offset points", xytext=(8, -12))
    ax[1].set(title="Cumulative energy  " r"$\sum\sigma_i^2$",
              xlabel="rank k", ylabel="retained energy ratio", ylim=(0, 1.02))
    fig.tight_layout()
    _save(fig, "fig_svd_spectrum.png")


def fig_eckart_young(sample_index=0):
    img = dataset.load_image(dataset.list_images()[sample_index], 512)
    gray = np.asarray(img).mean(axis=2)
    svd = svd_core.svd_decompose(gray)
    ks = np.arange(1, min(120, len(svd.s)))
    spec_theory, frob_theory, spec_meas, frob_meas, cf = [], [], [], [], []
    for k in ks:
        e = svd_core.eckart_young_errors(svd, k)
        spec_theory.append(e["spectral_sigma_k1"]); frob_theory.append(e["frobenius_tail"])
        Ak = svd.reconstruct(k)
        spec_meas.append(np.linalg.norm(gray - Ak, 2))
        frob_meas.append(np.linalg.norm(gray - Ak, "fro"))
        cf.append(svd_core.compression_factor(*gray.shape, k))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(ks, frob_theory, label=r"theory $\sqrt{\sum_{i>k}\sigma_i^2}$", lw=2)
    ax[0].plot(ks, frob_meas, "--", label=r"measured $\|A-A_k\|_F$")
    ax[0].plot(ks, spec_theory, label=r"theory $\sigma_{k+1}$", lw=2)
    ax[0].plot(ks, spec_meas, "--", label=r"measured $\|A-A_k\|_2$")
    ax[0].set(title="Eckart–Young: theory vs measured error",
              xlabel="rank k", ylabel="approximation error")
    ax[0].legend(fontsize=9)
    ax[1].plot(ks, cf, color="tab:red")
    ax[1].axhline(1.0, ls=":", color="gray")
    ax[1].set(title=r"Compression factor  $k(m+n+1)/mn$",
              xlabel="rank k", ylabel="footprint ratio")
    fig.tight_layout()
    _save(fig, "fig_eckart_young.png")


def fig_lowrank_montage(sample_index=0):
    img = dataset.load_image(dataset.list_images()[sample_index], 512)
    gray = np.asarray(img).mean(axis=2)
    svd = svd_core.svd_decompose(gray)
    ks = [2, 5, 10, 25, 50, 512]
    fig, ax = plt.subplots(1, len(ks), figsize=(2.2 * len(ks), 2.6))
    for a, k in zip(ax, ks):
        Ak = np.clip(svd.reconstruct(k), 0, 255)
        a.imshow(Ak, cmap="gray", vmin=0, vmax=255)
        cf = svd_core.compression_factor(*gray.shape, k)
        a.set_title(f"k={k}\ncf={cf:.2f}" if k < 512 else "full", fontsize=10)
        a.axis("off")
    fig.suptitle("Low-rank reconstruction of a cover (truncated SVD)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_lowrank_montage.png")


# ---------------------------------------------------------------------------
# 2. Secret compression (from results/secret_compression.csv)
# ---------------------------------------------------------------------------
def fig_secret_compression():
    csv = RES / "secret_compression.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(df["k"], df["recon_psnr"], "o-", label="recon PSNR (dB)")
    ax[0].set(title="Secret recovery vs rank k (compression quality)",
              xlabel="rank k", ylabel="PSNR (dB)")
    ax2 = ax[0].twinx(); ax2.grid(False)
    ax2.plot(df["k"], df["energy_ratio"], "s--", color="tab:green", label="energy")
    ax2.set_ylabel("retained energy", color="tab:green")
    ax[1].plot(df["k"], df["compression_factor"], "d-", color="tab:red")
    ax[1].set(title="Secret payload footprint vs rank k",
              xlabel="rank k", ylabel=r"compression factor $k(m+n+1)/mn$")
    fig.tight_layout()
    _save(fig, "fig_secret_compression.png")


def fig_secret_reconstructions():
    sec = dataset.default_secret(64)
    ks = [2, 4, 8, 12, 20, 32]
    fig, ax = plt.subplots(1, len(ks) + 1, figsize=(2.0 * (len(ks) + 1), 2.4))
    ax[0].imshow(sec, cmap="gray", vmin=0, vmax=255); ax[0].set_title("original"); ax[0].axis("off")
    for a, k in zip(ax[1:], ks):
        rec = steg.decompress_secret(steg.compress_secret(sec, k))
        a.imshow(rec, cmap="gray", vmin=0, vmax=255)
        a.set_title(f"k={k}\nNC={M.normalized_correlation(sec, rec):.2f}", fontsize=9)
        a.axis("off")
    fig.suptitle("Truncated-SVD compression of the secret image", y=1.04)
    fig.tight_layout()
    _save(fig, "fig_secret_reconstructions.png")


# ---------------------------------------------------------------------------
# 3. Imperceptibility + YOLO object protection (from metrics CSV)
# ---------------------------------------------------------------------------
def fig_object_protection(df):
    cols = ["hc_guided_obj_psnr", "hc_baseline_obj_psnr",
            "hc_guided_bg_psnr", "hc_baseline_bg_psnr"]
    if not all(c in df for c in cols):
        return
    means = [pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
             for c in cols]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(2); w = 0.35
    ax[0].bar(x - w/2, [means[0], means[2]], w, label="YOLO-guided", color="tab:blue")
    ax[0].bar(x + w/2, [means[1], means[3]], w, label="baseline (no YOLO)", color="tab:orange")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["object region", "background"])
    ax[0].set(title="Region PSNR: object protection via YOLO", ylabel="PSNR (dB)")
    for i, v in enumerate([means[0], means[2]]):
        ax[0].text(i - w/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate([means[1], means[3]]):
        ax[0].text(i + w/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax[0].legend()

    g = pd.to_numeric(df["hc_guided_obj_psnr"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(df["hc_baseline_obj_psnr"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ax[1].hist(b, bins=25, alpha=0.6, label="baseline", color="tab:orange")
    ax[1].hist(g, bins=25, alpha=0.6, label="YOLO-guided", color="tab:blue")
    ax[1].set(title=f"Object-region PSNR distribution ({len(df)} images)",
              xlabel="object PSNR (dB)", ylabel="# images")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "fig_object_protection.png")


def fig_imperceptibility(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for col, lab, c in [("hc_guided_psnr", "High-Capacity", "tab:blue"),
                        ("rb_guided_psnr", "Robust", "tab:green")]:
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            ax[0].hist(v, bins=25, alpha=0.6, label=f"{lab} (μ={v.mean():.1f} dB)", color=c)
    ax[0].set(title="Cover→stego PSNR distribution", xlabel="PSNR (dB)", ylabel="# images")
    ax[0].legend()
    for col, lab, c in [("hc_guided_ssim", "High-Capacity", "tab:blue"),
                        ("rb_guided_ssim", "Robust", "tab:green")]:
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            ax[1].hist(v, bins=25, alpha=0.6, label=f"{lab} (μ={v.mean():.3f})", color=c)
    ax[1].set(title="Cover→stego SSIM distribution", xlabel="SSIM", ylabel="# images")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "fig_imperceptibility.png")


def fig_detection_preservation(df):
    if "hc_det_map50" not in df:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, title, xl in [
        (ax[0], "hc_det_preservation_rate", "Detection preservation rate", "rate"),
        (ax[1], "hc_det_map50", "mAP@0.5 (stego vs cover)", "mAP@0.5"),
        (ax[2], "hc_det_mean_conf_delta", "Mean |Δ confidence|", "|Δconf|")]:
        v = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        a.hist(v, bins=20, color="tab:purple", alpha=0.8)
        a.axvline(v.mean(), color="k", ls="--", label=f"μ={v.mean():.3f}")
        a.set(title=title, xlabel=xl, ylabel="# images"); a.legend()
    fig.suptitle("Downstream YOLO detection preserved after embedding (High-Capacity)", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_detection_preservation.png")


# ---------------------------------------------------------------------------
# 4. Robustness + capacity/robustness trade-off
# ---------------------------------------------------------------------------
def fig_robustness(df):
    attacks = [c[len("hc_atk_"):-len("_nc")] for c in df.columns
               if c.startswith("hc_atk_") and c.endswith("_nc")]
    if not attacks:
        return
    def mean(col):
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
    hc_nc = [mean(f"hc_atk_{a}_nc") for a in attacks]
    rb_nc = [mean(f"rb_atk_{a}_nc") for a in attacks]
    hc_ber = [mean(f"hc_atk_{a}_ber") for a in attacks]
    rb_ber = [mean(f"rb_atk_{a}_ber") for a in attacks]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    x = np.arange(len(attacks)); w = 0.38
    ax[0].bar(x - w/2, hc_nc, w, label="High-Capacity", color="tab:blue")
    ax[0].bar(x + w/2, rb_nc, w, label="Robust", color="tab:green")
    ax[0].set_xticks(x); ax[0].set_xticklabels(attacks, rotation=40, ha="right")
    ax[0].set(title="Recovered-secret NC under attack", ylabel="normalised correlation", ylim=(0, 1))
    ax[0].legend()
    ax[1].bar(x - w/2, hc_ber, w, label="High-Capacity", color="tab:blue")
    ax[1].bar(x + w/2, rb_ber, w, label="Robust", color="tab:green")
    ax[1].set_xticks(x); ax[1].set_xticklabels(attacks, rotation=40, ha="right")
    ax[1].set(title="Payload BER under attack", ylabel="bit error rate")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "fig_robustness.png")


def fig_tradeoff(df):
    def mean(col):
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
    attacks = [c[len("hc_atk_"):-len("_nc")] for c in df.columns
               if c.startswith("hc_atk_") and c.endswith("_nc") and "none" not in c]
    pts = []
    for mode, c in [("hc", "tab:blue"), ("rb", "tab:green")]:
        bpp = mean(f"{mode}_capacity_bpp")
        ncs = [mean(f"{mode}_atk_{a}_nc") for a in attacks]
        psnr = mean(f"{mode}_guided_psnr")
        pts.append((mode, bpp, np.nanmean(ncs), psnr, c))
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, bpp, nc, psnr, c in pts:
        ax.scatter(bpp, nc, s=180, color=c,
                   label=f"{'High-Capacity' if mode=='hc' else 'Robust'}\n"
                         f"(bpp={bpp:.4f}, PSNR={psnr:.1f} dB)")
        ax.annotate(mode.upper(), (bpp, nc), textcoords="offset points", xytext=(8, 6))
    ax.set(title="Capacity ↔ Robustness trade-off",
           xlabel="capacity (bits per pixel)",
           ylabel="mean recovered-secret NC under attack", ylim=(-0.05, 1.0))
    ax.set_xscale("log"); ax.legend(loc="center right")
    fig.tight_layout()
    _save(fig, "fig_tradeoff.png")


# ---------------------------------------------------------------------------
# 5. Qualitative montage (cover/stego/diff/saliency/secret/recovered)
# ---------------------------------------------------------------------------
def fig_qualitative(indices=(0, 1, 2)):
    cfg = hc_config()
    secret = dataset.default_secret(cfg.secret_size)
    det = yg.YOLODetector("yolov8n.pt")
    paths = dataset.list_images()
    rows = len(indices)
    fig, ax = plt.subplots(rows, 5, figsize=(15, 3.0 * rows))
    if rows == 1:
        ax = ax[None, :]
    for r, idx in enumerate(indices):
        cover = dataset.load_image(paths[idx], cfg.cover_size)
        sal, dets = yg.priority_map(cover, det)
        res, _ = steg.hide_secret(cover, secret, cfg.k, cfg.embed, sal)
        rec = steg.reveal_secret(res.stego, cfg.embed, sal)
        diff = np.abs(cover.astype(float) - res.stego.astype(float)).mean(axis=2)
        psnr = M.psnr(cover, res.stego)
        ax[r, 0].imshow(cover); ax[r, 0].set_ylabel(paths[idx].stem, fontsize=8)
        ax[r, 0].set_title("cover" if r == 0 else "")
        ax[r, 1].imshow(res.stego); ax[r, 1].set_title(f"stego ({psnr:.1f} dB)" if r == 0 else f"{psnr:.1f} dB")
        im = ax[r, 2].imshow(diff, cmap="inferno"); ax[r, 2].set_title("|cover−stego|" if r == 0 else "")
        ax[r, 3].imshow(sal, cmap="viridis"); ax[r, 3].set_title("YOLO saliency" if r == 0 else "")
        ax[r, 4].imshow(rec, cmap="gray", vmin=0, vmax=255)
        ax[r, 4].set_title("recovered secret" if r == 0 else "")
        for c in range(5):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle("Qualitative results — YOLO-guided embedding avoids object regions", y=1.0)
    fig.tight_layout()
    _save(fig, "fig_qualitative.png")


def main():
    print("[figures] generating …")
    csv = RES / "metrics_per_image.csv"
    df = pd.read_csv(csv) if csv.exists() else None
    # SVD theory figures (live)
    fig_svd_spectrum(); fig_eckart_young(); fig_lowrank_montage()
    fig_secret_compression(); fig_secret_reconstructions()
    # experiment figures
    if df is not None:
        fig_imperceptibility(df); fig_object_protection(df)
        fig_detection_preservation(df); fig_robustness(df); fig_tradeoff(df)
    else:
        print("  [warn] metrics_per_image.csv not found — run experiments first")
    fig_qualitative()
    print("[figures] done.")


if __name__ == "__main__":
    main()
