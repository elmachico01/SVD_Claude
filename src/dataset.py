"""
dataset.py
==========
COCO-128 access (a 128-image subset of the COCO dataset) and the secret image.

COCO-128 is the canonical small COCO sample shipped by Ultralytics; it gives us
>= 100 *real* COCO images for the experiments without downloading the full
multi-gigabyte dataset.  If the images are missing they are fetched from the
official GitHub mirror.
"""
from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COCO_DIR = DATA_DIR / "coco128" / "images" / "train2017"
# GitHub mirror (the ultralytics.com host is often blocked by proxies).
COCO128_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"


def ensure_coco128() -> Path:
    """Download + unzip COCO-128 if it is not already present."""
    if COCO_DIR.exists() and any(COCO_DIR.glob("*.jpg")):
        return COCO_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "coco128.zip"
    print(f"[dataset] downloading COCO-128 from {COCO128_URL} …")
    urllib.request.urlretrieve(COCO128_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)
    if not (COCO_DIR.exists() and any(COCO_DIR.glob("*.jpg"))):
        raise RuntimeError("COCO-128 download/extract failed")
    return COCO_DIR


def list_images(limit: int | None = None) -> list[Path]:
    """Sorted list of COCO-128 image paths (downloads the set if needed)."""
    ensure_coco128()
    paths = sorted(COCO_DIR.glob("*.jpg"))
    return paths[:limit] if limit else paths


def load_image(path: str | Path, size: int = 512) -> np.ndarray:
    """Load an image as a square RGB uint8 array of side ``size``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def default_secret(size: int = 64) -> np.ndarray:
    """
    A deterministic synthetic grayscale secret (logo) of side ``size``.

    The design — a smooth gradient background plus a few bold geometric shapes —
    is recognisable at any resolution (16 px … 64 px) and has a non-trivial,
    fast-decaying singular-value spectrum, which makes it an ideal subject for
    the truncated-SVD compression demonstration.
    """
    s = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    s += 50 + 70 * (xx / size) + 50 * (yy / size)        # smooth low-rank gradient
    c = size // 2
    cv2.circle(s, (c, c), int(size * 0.38), 235, -1)      # bold disk (dominant component)
    cv2.circle(s, (c, c), int(size * 0.20), 40, -1)       # inner contrast disk (the "ring")
    cv2.rectangle(s, (int(size * 0.12), int(size * 0.44)),
                  (int(size * 0.88), int(size * 0.56)), 255, -1)   # central bar
    if size >= 48:                                        # text only when legible
        cv2.putText(s, "SVD", (int(size * 0.07), size - max(4, size // 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 110.0, 15, 1, cv2.LINE_AA)
    return np.clip(s, 0, 255).astype(np.uint8)


def image_as_secret(path: str | Path, size: int = 64) -> np.ndarray:
    """Use any image (e.g. a COCO photo) as a grayscale secret of side ``size``."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
