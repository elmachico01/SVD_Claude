"""YOLO guidance and downstream detection-consistency metrics."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    cls: int
    conf: float


class YOLODetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25,
                 imgsz: int = 640, device: str = "cpu"):
        self.weights, self.conf, self.imgsz, self.device = weights, conf, imgsz, device
        self.available = False
        self.names: dict[int, str] = {}
        self._model = self._load()

    def _load(self):
        if self.weights in _MODEL_CACHE:
            model = _MODEL_CACHE[self.weights]
            self.available = True
            self.names = model.names
            return model
        try:
            from ultralytics import YOLO
            model = YOLO(self.weights)
            _MODEL_CACHE[self.weights] = model
            self.available = True
            self.names = model.names
            return model
        except Exception as exc:  # environment-dependent
            print(f"[yolo_guidance] YOLO unavailable ({type(exc).__name__}); using fallback")
            return None

    def detect(self, img_rgb: np.ndarray) -> list[Detection]:
        if not self.available:
            return []
        result = self._model(img_rgb[:, :, ::-1], conf=self.conf, imgsz=self.imgsz,
                             device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidence = result.boxes.conf.cpu().numpy()
        return [Detection(tuple(map(float, b)), int(c), float(cf))
                for b, c, cf in zip(boxes, classes, confidence)]


def saliency_from_detections(shape, detections: list[Detection],
                             sigma_frac: float = 0.04) -> np.ndarray:
    H, W = shape[:2]
    sal = np.zeros((H, W), dtype=np.float32)
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            sal[y1:y2, x1:x2] = np.maximum(sal[y1:y2, x1:x2], d.conf)
    sigma = max(1, int(sigma_frac * max(H, W)))
    sal = cv2.GaussianBlur(sal, (0, 0), sigma)
    return sal / sal.max() if sal.max() > 0 else sal


def spectral_residual_saliency(img_rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Return saliency and the actual fallback implementation used."""
    try:
        module = getattr(cv2, "saliency", None)
        if module is not None:
            detector = module.StaticSaliencySpectralResidual_create()
            ok, sal = detector.computeSaliency(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            if ok:
                sal = sal.astype(np.float32)
                return (sal - sal.min()) / (np.ptp(sal) + 1e-8), "spectral_residual"
    except Exception:
        pass
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sal = np.hypot(gx, gy)
    return (sal - sal.min()) / (np.ptp(sal) + 1e-8), "gradient_fallback"


def priority_map(img_rgb: np.ndarray, detector: YOLODetector | None = None,
                 detections: list[Detection] | None = None,
                 return_source: bool = False):
    if detections is None and detector is not None and detector.available:
        detections = detector.detect(img_rgb)
    detections = detections or []
    if detections:
        result = (saliency_from_detections(img_rgb.shape, detections), detections, "yolo")
    else:
        saliency, source = spectral_residual_saliency(img_rgb)
        result = (saliency, detections, source)
    return result if return_source else result[:2]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _consistency_ap(reference: list[Detection], candidate: list[Detection],
                    iou_thr: float = 0.5) -> float:
    """AP-like agreement, using cover detections as pseudo-reference, not COCO GT."""
    if not reference:
        return float("nan")
    if not candidate:
        return 0.0
    candidate = sorted(candidate, key=lambda d: d.conf, reverse=True)
    matched = [False] * len(reference)
    tp, fp = np.zeros(len(candidate)), np.zeros(len(candidate))
    for i, pred in enumerate(candidate):
        best, best_j = 0.0, -1
        for j, ref in enumerate(reference):
            if matched[j] or pred.cls != ref.cls:
                continue
            value = _iou(pred.box, ref.box)
            if value > best:
                best, best_j = value, j
        if best >= iou_thr:
            tp[i], matched[best_j] = 1, True
        else:
            fp[i] = 1
    recall = np.cumsum(tp) / len(reference)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_preservation(cover_dets: list[Detection], stego_dets: list[Detection],
                           iou_thr: float = 0.5) -> dict:
    """Measure consistency with cover detections; this is not dataset mAP."""
    if not cover_dets:
        return {"n_cover": 0, "n_stego": len(stego_dets),
                "preservation_rate": float("nan"), "mean_iou": float("nan"),
                "mean_conf_delta": float("nan"), "consistency_ap50": float("nan")}
    used = [False] * len(stego_dets)
    ious, conf_delta = [], []
    for ref in cover_dets:
        best, best_j = iou_thr, -1
        for j, pred in enumerate(stego_dets):
            if used[j] or pred.cls != ref.cls:
                continue
            value = _iou(ref.box, pred.box)
            if value >= best:
                best, best_j = value, j
        if best_j >= 0:
            used[best_j] = True
            ious.append(best)
            conf_delta.append(abs(ref.conf - stego_dets[best_j].conf))
    return {
        "n_cover": len(cover_dets), "n_stego": len(stego_dets),
        "preservation_rate": len(ious) / len(cover_dets),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_conf_delta": float(np.mean(conf_delta)) if conf_delta else float("nan"),
        "consistency_ap50": _consistency_ap(cover_dets, stego_dets, iou_thr),
    }


def object_background_masks(shape, detections: list[Detection]):
    H, W = shape[:2]
    obj = np.zeros((H, W), dtype=bool)
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        obj[y1:y2, x1:x2] = True
    return obj, ~obj
