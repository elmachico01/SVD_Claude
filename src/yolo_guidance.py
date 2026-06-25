"""
yolo_guidance.py
================
The YOLO half of the project.  YOLOv8 plays two roles:

1. **Content-adaptive embedding guidance.**
   Object detections are turned into a per-pixel *saliency / priority map*.
   The steganography engine fills low-priority (background) blocks first and
   leaves the salient object regions almost untouched, which (a) improves
   perceptual quality where the human eye looks and (b) preserves the
   information a downstream detector relies on.

2. **Downstream-task evaluation.**
   By running YOLO on both the cover and the stego image we measure how well
   the *semantic content* survives embedding (detection-preservation rate,
   IoU agreement, confidence drift, mAP@0.5).  A good stego image should be
   indistinguishable to a detector, not only to the eye.

If ``ultralytics`` / ``torch`` / the model weights are unavailable, the module
degrades gracefully to a classical OpenCV spectral-residual saliency map so the
embedding pipeline still runs (detection metrics are then disabled).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # xyxy in pixel coords
    cls: int
    conf: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class YOLODetector:
    """Thin wrapper around Ultralytics YOLOv8 with a saliency fallback."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25,
                 imgsz: int = 640, device: str = "cpu"):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.available = False
        self.names: dict[int, str] = {}
        self._model = self._load()

    def _load(self):
        if self.weights in _MODEL_CACHE:
            self.available = True
            m = _MODEL_CACHE[self.weights]
            self.names = m.names
            return m
        try:
            from ultralytics import YOLO
            m = YOLO(self.weights)
            _MODEL_CACHE[self.weights] = m
            self.available = True
            self.names = m.names
            return m
        except Exception as e:   # pragma: no cover - environment dependent
            print(f"[yolo_guidance] YOLO unavailable ({type(e).__name__}); "
                  f"using saliency fallback.")
            self.available = False
            return None

    def detect(self, img_rgb: np.ndarray) -> list[Detection]:
        """Run detection on an RGB uint8 image; empty list if YOLO unavailable."""
        if not self.available:
            return []
        res = self._model(img_rgb[:, :, ::-1], conf=self.conf, imgsz=self.imgsz,
                          device=self.device, verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for b, c, cf in zip(xyxy, cls, conf):
            dets.append(Detection(box=tuple(map(float, b)), cls=int(c), conf=float(cf)))
        return dets


# ---------------------------------------------------------------------------
# Priority / saliency map
# ---------------------------------------------------------------------------
def saliency_from_detections(shape, detections: list[Detection],
                             sigma_frac: float = 0.04) -> np.ndarray:
    """
    Build a per-pixel object-saliency map in [0, 1] from YOLO boxes.

    Object interiors get a value equal to the detection confidence; a Gaussian
    blur creates soft margins so blocks straddling an object boundary are also
    (partly) protected.  Background → 0.
    """
    H, W = shape[:2]
    sal = np.zeros((H, W), dtype=np.float32)
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            sal[y1:y2, x1:x2] = np.maximum(sal[y1:y2, x1:x2], d.conf)
    ksig = max(1, int(sigma_frac * max(H, W)))
    sal = cv2.GaussianBlur(sal, (0, 0), ksig)
    if sal.max() > 0:
        sal /= sal.max()
    return sal


def spectral_residual_saliency(img_rgb: np.ndarray) -> np.ndarray:
    """Classical fallback saliency (OpenCV spectral residual), in [0, 1]."""
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, m = sal.computeSaliency(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if ok:
            m = m.astype(np.float32)
            return (m - m.min()) / (np.ptp(m) + 1e-8)
    except Exception:
        pass
    # last resort: gradient-magnitude texture map
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.hypot(gx, gy)
    return (m - m.min()) / (np.ptp(m) + 1e-8)


def priority_map(img_rgb: np.ndarray, detector: YOLODetector | None = None,
                 detections: list[Detection] | None = None) -> tuple[np.ndarray, list[Detection]]:
    """
    Return ``(saliency, detections)``.  Higher saliency ⇒ filled later, so the
    payload migrates to the background.  Uses YOLO when available, otherwise the
    spectral-residual fallback.
    """
    if detections is None and detector is not None and detector.available:
        detections = detector.detect(img_rgb)
    if detections:
        return saliency_from_detections(img_rgb.shape, detections), detections
    return spectral_residual_saliency(img_rgb), (detections or [])


# ---------------------------------------------------------------------------
# Detection-preservation metrics (downstream-task evaluation)
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _average_precision(gt: list[Detection], pred: list[Detection],
                       iou_thr: float = 0.5) -> float:
    """All-point AP@iou for a single image, cover detections as ground truth."""
    if not gt:
        return float("nan")
    if not pred:
        return 0.0
    pred = sorted(pred, key=lambda d: d.conf, reverse=True)
    matched = [False] * len(gt)
    tp = np.zeros(len(pred)); fp = np.zeros(len(pred))
    for i, p in enumerate(pred):
        best, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if g.cls != p.cls or matched[j]:
                continue
            iou = _iou(p.box, g.box)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_thr and best_j >= 0:
            tp[i] = 1; matched[best_j] = True
        else:
            fp[i] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / len(gt)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # all-point interpolation
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_preservation(cover_dets: list[Detection],
                           stego_dets: list[Detection],
                           iou_thr: float = 0.5) -> dict:
    """
    Compare detections on cover vs. stego (cover treated as reference).
    Returns preservation rate, mean matched IoU, mean |Δconf|, mAP@0.5.
    """
    n_cover = len(cover_dets)
    if n_cover == 0:
        return {"n_cover": 0, "n_stego": len(stego_dets),
                "preservation_rate": float("nan"), "mean_iou": float("nan"),
                "mean_conf_delta": float("nan"), "map50": float("nan")}
    matched = [False] * len(stego_dets)
    ious, dconf, n_match = [], [], 0
    for g in cover_dets:
        best, best_j = iou_thr, -1
        for j, s in enumerate(stego_dets):
            if s.cls != g.cls or matched[j]:
                continue
            iou = _iou(g.box, s.box)
            if iou >= best:
                best, best_j = iou, j
        if best_j >= 0:
            matched[best_j] = True
            n_match += 1
            ious.append(_iou(g.box, stego_dets[best_j].box))
            dconf.append(abs(g.conf - stego_dets[best_j].conf))
    return {
        "n_cover": n_cover,
        "n_stego": len(stego_dets),
        "preservation_rate": n_match / n_cover,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_conf_delta": float(np.mean(dconf)) if dconf else float("nan"),
        "map50": _average_precision(cover_dets, stego_dets, iou_thr),
    }


def object_background_masks(shape, detections: list[Detection]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (object_mask, background_mask) from detection boxes."""
    H, W = shape[:2]
    obj = np.zeros((H, W), dtype=bool)
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        obj[y1:y2, x1:x2] = True
    return obj, ~obj
