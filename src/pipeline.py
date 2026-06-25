"""
pipeline.py
===========
End-to-end orchestration for a single cover image:

    cover ─► YOLO priority map ─► hide secret (guided & baseline)
          ─► fidelity metrics (global / object / background)
          ─► blind recovery + robustness under attacks
          ─► downstream detection-preservation

The same routine powers both the single-image demo and the batch experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import steganography as steg
from . import metrics as M
from . import attacks as A
from . import yolo_guidance as yg
from .steganography import EmbedConfig


@dataclass
class PipelineConfig:
    cover_size: int = 512
    secret_size: int = 64
    k: int = 10                       # truncated-SVD rank of the secret
    embed: EmbedConfig = None
    attacks: tuple[str, ...] = ("jpeg_q90", "jpeg_q75", "jpeg_q50",
                                "gauss_noise5", "blur_3x3", "median_3x3",
                                "saltpepper_1", "rescale_50")

    def __post_init__(self):
        if self.embed is None:
            self.embed = EmbedConfig()


# Two reference operating points used throughout the project ----------------
def hc_config() -> "PipelineConfig":
    """High-Capacity: max payload + best imperceptibility (no redundancy)."""
    return PipelineConfig(cover_size=512, secret_size=64, k=10,
                          embed=EmbedConfig(block=8, delta=24.0,
                                            channels=(0, 1, 2), repeat=1))


def rb_config() -> "PipelineConfig":
    """Robust: larger SVD blocks + repetition code → survives common attacks."""
    return PipelineConfig(cover_size=512, secret_size=16, k=3,
                          embed=EmbedConfig(block=16, delta=128.0,
                                            channels=(0, 1, 2), repeat=3))


def _recover_quality(secret: np.ndarray, payload: bytes,
                     stego: np.ndarray, cfg: EmbedConfig,
                     priority_map, k: int) -> dict:
    """Reveal the secret from ``stego`` and score it against the original."""
    true_bits = steg.bytes_to_bits(payload)
    try:
        recovered = steg.reveal_secret(stego, cfg, priority_map)
    except Exception:
        recovered = np.zeros_like(secret)
    # BER on the raw payload bits (length we actually embedded)
    est_bits = steg.extract_bits(stego, len(true_bits), cfg, priority_map)
    return {
        "recovered": recovered,
        "secret_psnr": M.psnr(secret, recovered) if recovered.shape == secret.shape else 0.0,
        "secret_nc": M.normalized_correlation(secret, recovered)
        if recovered.shape == secret.shape else 0.0,
        "ber": M.ber(true_bits, est_bits),
    }


def process_image(cover: np.ndarray, secret: np.ndarray,
                  detector: yg.YOLODetector | None,
                  cfg: PipelineConfig,
                  run_baseline: bool = True,
                  run_attacks: bool = True,
                  run_downstream: bool = True,
                  precomputed: tuple | None = None) -> dict:
    """
    Run the full evaluation for one cover/secret pair. Returns a flat dict.

    ``precomputed`` may carry ``(saliency, cover_dets)`` so that YOLO is run only
    once per cover when several embedding configurations are evaluated.
    """
    ec = cfg.embed
    out: dict = {}

    # --- YOLO priority map + reference detections -------------------------
    if precomputed is not None:
        saliency, cover_dets = precomputed
    else:
        saliency, cover_dets = yg.priority_map(cover, detector)
    out["n_cover_detections"] = len(cover_dets)
    out["yolo_available"] = bool(detector and detector.available)
    obj_mask, bg_mask = yg.object_background_masks(cover.shape, cover_dets)

    # --- Guided embedding (YOLO background-first) -------------------------
    res_g, payload = steg.hide_secret(cover, secret, cfg.k, ec, priority_map=saliency)
    stego_g = res_g.stego
    out["payload_bits"] = res_g.num_bits
    out["capacity_bits"] = res_g.capacity_bits
    out["capacity_bpp"] = M.capacity_bpp(res_g.capacity_bits, *cover.shape[:2])
    out["bits_used_frac"] = res_g.num_bits / max(1, res_g.capacity_bits)

    fg = M.summarize_fidelity(cover, stego_g)
    out.update({f"guided_{k}": v for k, v in fg.items()})
    out["guided_obj_psnr"] = M.region_psnr(cover, stego_g, obj_mask)
    out["guided_bg_psnr"] = M.region_psnr(cover, stego_g, bg_mask)

    rg = _recover_quality(secret, payload, stego_g, ec, saliency, cfg.k)
    out["guided_secret_psnr"] = rg["secret_psnr"]
    out["guided_secret_nc"] = rg["secret_nc"]
    out["guided_ber"] = rg["ber"]

    # downstream detection preservation (guided stego)
    if run_downstream and detector and detector.available:
        stego_dets = detector.detect(stego_g)
        dp = yg.detection_preservation(cover_dets, stego_dets)
        out.update({f"det_{k}": v for k, v in dp.items()})

    # --- Baseline embedding (keyed permutation, no YOLO) -----------------
    if run_baseline:
        res_b, payload_b = steg.hide_secret(cover, secret, cfg.k, ec, priority_map=None)
        stego_b = res_b.stego
        fb = M.summarize_fidelity(cover, stego_b)
        out.update({f"baseline_{k}": v for k, v in fb.items()})
        out["baseline_obj_psnr"] = M.region_psnr(cover, stego_b, obj_mask)
        out["baseline_bg_psnr"] = M.region_psnr(cover, stego_b, bg_mask)
        rb = _recover_quality(secret, payload_b, stego_b, ec, None, cfg.k)
        out["baseline_secret_nc"] = rb["secret_nc"]
        out["baseline_ber"] = rb["ber"]
        if run_downstream and detector and detector.available:
            dpb = yg.detection_preservation(cover_dets, detector.detect(stego_b))
            out["baseline_det_preservation_rate"] = dpb["preservation_rate"]
            out["baseline_det_map50"] = dpb["map50"]

    # --- Robustness under attacks (guided stego) -------------------------
    if run_attacks:
        true_bits = steg.bytes_to_bits(payload)
        for name in cfg.attacks:
            att = A.apply_attack(stego_g, name)
            est = steg.extract_bits(att, len(true_bits), ec, saliency)
            out[f"atk_{name}_ber"] = M.ber(true_bits, est)
            try:
                rec = steg.reveal_secret(att, ec, saliency)
                out[f"atk_{name}_nc"] = (M.normalized_correlation(secret, rec)
                                         if rec.shape == secret.shape else 0.0)
            except Exception:
                out[f"atk_{name}_nc"] = 0.0

    # stash a few arrays for the demo (not used by the batch CSV)
    out["_arrays"] = {"cover": cover, "stego_guided": stego_g,
                      "saliency": saliency, "recovered": rg["recovered"],
                      "cover_dets": cover_dets}
    return out
