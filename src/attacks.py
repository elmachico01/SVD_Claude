"""
attacks.py
==========
Image-processing "attacks" used to assess the **robustness** of the embedded
secret.  A robust stego scheme should still allow the secret to be recovered
(low BER / high NC) after the stego image undergoes common, non-malicious
manipulations such as recompression, resampling or noise.

Each attack takes a uint8 image (H, W, 3) and returns a uint8 image of the
same shape.
"""
from __future__ import annotations

from typing import Callable
import cv2
import numpy as np


def _as_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(img), 0, 255).astype(np.uint8)


def jpeg_compression(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """Lossy JPEG re-encoding at the given quality factor (0–100)."""
    ok, enc = cv2.imencode(".jpg", _as_uint8(img),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return _as_uint8(img)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def gaussian_noise(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Additive zero-mean Gaussian noise with standard deviation ``sigma``."""
    noise = np.random.normal(0.0, sigma, np.asarray(img).shape)
    return _as_uint8(img.astype(np.float64) + noise)


def gaussian_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gaussian low-pass blur with a (ksize × ksize) kernel."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(_as_uint8(img), (ksize, ksize), 0)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filtering (effective against salt-and-pepper noise)."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(_as_uint8(img), ksize)


def salt_pepper(img: np.ndarray, amount: float = 0.01) -> np.ndarray:
    """Replace a fraction ``amount`` of pixels with black/white impulses."""
    out = _as_uint8(img).copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    out[ys[: n // 2], xs[: n // 2]] = 0
    out[ys[n // 2:], xs[n // 2:]] = 255
    return out


def rescale(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Down-then-up sampling (loses high-frequency detail)."""
    img = _as_uint8(img)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# Registry of attacks used by the experiment harness.
# Each entry: name -> (callable, kwargs).
ATTACKS: dict[str, tuple[Callable, dict]] = {
    "none":         (lambda x: _as_uint8(x), {}),
    "jpeg_q90":     (jpeg_compression, {"quality": 90}),
    "jpeg_q75":     (jpeg_compression, {"quality": 75}),
    "jpeg_q50":     (jpeg_compression, {"quality": 50}),
    "gauss_noise5": (gaussian_noise,   {"sigma": 5.0}),
    "blur_3x3":     (gaussian_blur,    {"ksize": 3}),
    "median_3x3":   (median_filter,    {"ksize": 3}),
    "saltpepper_1": (salt_pepper,      {"amount": 0.01}),
    "rescale_50":   (rescale,          {"factor": 0.5}),
}


def apply_attack(img: np.ndarray, name: str) -> np.ndarray:
    """Apply a named attack from the :data:`ATTACKS` registry."""
    fn, kw = ATTACKS[name]
    return _as_uint8(fn(img, **kw))
