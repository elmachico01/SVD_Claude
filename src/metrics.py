"""
metrics.py
==========
Quantitative quality measures used throughout the project.

Two families:
  * **Imperceptibility / fidelity** (cover vs. stego, or secret vs. recovered):
        MSE, PSNR, SSIM.
  * **Payload integrity** (bitstream / secret recovery):
        BER (bit error rate), NC (normalised correlation), capacity.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _ssim


# ---------------------------------------------------------------------------
# Image-fidelity metrics
# ---------------------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (∞ for identical images)."""
    error = mse(a, b)
    if error <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / error))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity index (handles grayscale and colour)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == 3:
        return float(_ssim(a, b, data_range=data_range, channel_axis=2))
    return float(_ssim(a, b, data_range=data_range))


def region_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                data_range: float = 255.0) -> float:
    """PSNR computed only on the pixels selected by a boolean ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2 and a.ndim == 3:
        mask = np.repeat(mask[:, :, None], a.shape[2], axis=2)
    if mask.sum() == 0:
        return float("nan")
    err = float(np.mean((a.astype(np.float64)[mask] - b.astype(np.float64)[mask]) ** 2))
    if err <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / err))


# ---------------------------------------------------------------------------
# Payload-integrity metrics
# ---------------------------------------------------------------------------
def ber(bits_true: np.ndarray, bits_est: np.ndarray) -> float:
    """Bit error rate between two equal-length bit arrays."""
    bits_true = np.asarray(bits_true).ravel().astype(np.uint8)
    bits_est = np.asarray(bits_est).ravel().astype(np.uint8)
    n = min(len(bits_true), len(bits_est))
    if n == 0:
        return 1.0
    return float(np.mean(bits_true[:n] != bits_est[:n]))


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalised correlation (zero-mean Pearson) between two images, in [-1, 1].
    The standard robustness measure for recovered watermarks/secrets.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def capacity_bpp(num_bits: int, height: int, width: int) -> float:
    """Embedding capacity expressed in bits per pixel."""
    return float(num_bits) / float(height * width)


def summarize_fidelity(cover: np.ndarray, stego: np.ndarray) -> dict[str, float]:
    """Convenience bundle: MSE / PSNR / SSIM for a cover–stego pair."""
    return {"mse": mse(cover, stego),
            "psnr": psnr(cover, stego),
            "ssim": ssim(cover, stego)}
