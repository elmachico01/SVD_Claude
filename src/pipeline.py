"""End-to-end experimental pipeline for SVD/QIM steganography."""
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
    k: int = 10
    embed: EmbedConfig | None = None
    attacks: tuple[str, ...] = (
        "jpeg_q90", "jpeg_q75", "jpeg_q50", "gauss_noise5",
        "blur_3x3", "median_3x3", "saltpepper_1", "rescale_50")

    def __post_init__(self):
        if self.embed is None:
            self.embed = EmbedConfig()


def hc_config() -> PipelineConfig:
    """High-capacity operating point: small blocks, no repetition."""
    return PipelineConfig(512, 64, 10,
                          EmbedConfig(block=8, delta=24.0, repeat=1))


def rb_config() -> PipelineConfig:
    """Robust operating point: larger blocks, larger QIM step, R=3."""
    return PipelineConfig(512, 16, 3,
                          EmbedConfig(block=16, delta=128.0, repeat=3))


def _recover_quality(secret, payload, image, cfg, priority_map) -> dict:
    true_bits = steg.bytes_to_bits(payload)
    estimated = steg.extract_bits(image, len(true_bits), cfg, priority_map)
    recovered, status = steg.reveal_secret(
        image, cfg, priority_map, return_status=True)
    shape_ok = recovered is not None and recovered.shape == secret.shape
    return {
        "recovered": recovered if shape_ok else np.zeros_like(secret),
        "secret_psnr": M.psnr(secret, recovered) if shape_ok else 0.0,
        "secret_nc": M.normalized_correlation(secret, recovered) if shape_ok else 0.0,
        "ber": M.ber(true_bits, estimated),
        "header_ok": int(status.header_ok),
        "crc_ok": int(status.crc_ok),
        "decode_success": int(shape_ok and status.crc_ok),
    }


def _store_recovery(out: dict, prefix: str, result: dict) -> None:
    for key in ("secret_psnr", "secret_nc", "ber", "header_ok", "crc_ok", "decode_success"):
        out[f"{prefix}_{key}"] = result[key]


def process_image(cover: np.ndarray, secret: np.ndarray,
                  detector: yg.YOLODetector | None,
                  cfg: PipelineConfig,
                  run_baseline: bool = True,
                  run_attacks: bool = True,
                  run_downstream: bool = True,
                  precomputed: tuple | None = None) -> dict:
    ec = cfg.embed
    out: dict = {}

    if precomputed is None:
        saliency, cover_dets, guidance_source = yg.priority_map(
            cover, detector, return_source=True)
    else:
        if len(precomputed) == 3:
            saliency, cover_dets, guidance_source = precomputed
        else:
            saliency, cover_dets = precomputed
            guidance_source = "yolo" if cover_dets else "fallback"

    out["n_cover_detections"] = len(cover_dets)
    out["yolo_available"] = int(bool(detector and detector.available))
    out["guidance_source"] = guidance_source
    obj_mask, bg_mask = yg.object_background_masks(cover.shape, cover_dets)

    # Guided mode. The original priority map is intentionally reused at extraction:
    # this mode therefore requires side information and is not called fully blind.
    res_g, payload = steg.hide_secret(cover, secret, cfg.k, ec, saliency)
    stego_g = res_g.stego
    out["payload_bits"] = res_g.num_bits
    out["capacity_bits"] = res_g.capacity_bits
    out["capacity_bpp"] = M.capacity_bpp(res_g.capacity_bits, *cover.shape[:2])
    out["bits_used_frac"] = res_g.num_bits / max(1, res_g.capacity_bits)
    out["qim_corrected_sites"] = res_g.corrected_sites

    out.update({f"guided_{k}": v for k, v in M.summarize_fidelity(cover, stego_g).items()})
    out["guided_obj_psnr"] = M.region_psnr(cover, stego_g, obj_mask)
    out["guided_bg_psnr"] = M.region_psnr(cover, stego_g, bg_mask)
    guided_rec = _recover_quality(secret, payload, stego_g, ec, saliency)
    _store_recovery(out, "guided", guided_rec)

    if run_downstream and detector and detector.available:
        consistency = yg.detection_preservation(cover_dets, detector.detect(stego_g))
        out.update({f"det_{k}": v for k, v in consistency.items()})

    if run_baseline:
        res_b, payload_b = steg.hide_secret(cover, secret, cfg.k, ec, None)
        stego_b = res_b.stego
        out.update({f"baseline_{k}": v for k, v in M.summarize_fidelity(cover, stego_b).items()})
        out["baseline_obj_psnr"] = M.region_psnr(cover, stego_b, obj_mask)
        out["baseline_bg_psnr"] = M.region_psnr(cover, stego_b, bg_mask)
        baseline_rec = _recover_quality(secret, payload_b, stego_b, ec, None)
        _store_recovery(out, "baseline", baseline_rec)
        if run_downstream and detector and detector.available:
            c = yg.detection_preservation(cover_dets, detector.detect(stego_b))
            out["baseline_det_preservation_rate"] = c["preservation_rate"]
            out["baseline_det_consistency_ap50"] = c["consistency_ap50"]

    if run_attacks:
        true_bits = steg.bytes_to_bits(payload)
        for name in cfg.attacks:
            attacked = A.apply_attack(stego_g, name)
            estimated = steg.extract_bits(attacked, len(true_bits), ec, saliency)
            out[f"atk_{name}_ber"] = M.ber(true_bits, estimated)
            recovered, status = steg.reveal_secret(attacked, ec, saliency, return_status=True)
            shape_ok = recovered is not None and recovered.shape == secret.shape
            out[f"atk_{name}_nc"] = M.normalized_correlation(secret, recovered) if shape_ok else 0.0
            out[f"atk_{name}_decode_success"] = int(shape_ok and status.crc_ok)

    out["_arrays"] = {
        "cover": cover, "stego_guided": stego_g, "saliency": saliency,
        "recovered": guided_rec["recovered"], "cover_dets": cover_dets}
    return out
