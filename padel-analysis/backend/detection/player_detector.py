from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from backend.ingestion.phone_preprocess import preprocess_phone_frame
from backend.utils.logging import get_logger
from backend.utils.types import BBox, CourtState, PlayerDetection

logger = get_logger(__name__)


class PlayerDetector:
    """Stage 3: YOLOv11 with phone preprocessing and temporal smoothing."""

    def __init__(self, config: dict):
        self.cfg = config["models"]
        self.pcfg = config.get("phone", {})
        self.conf_threshold = self.cfg.get("person_conf", 0.35)
        self.smooth_frames = self.cfg.get("temporal_smooth_frames", 3)
        self.phone_preprocess = self.pcfg.get("enabled", True)
        self.device = self._resolve_device(self.cfg.get("device", "auto"))
        self.model = None
        self._history: deque[list[PlayerDetection]] = deque(maxlen=self.smooth_frames)
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            weights = self.cfg.get("player_detector", "yolo11n.pt")
            self.model = YOLO(weights)
            if self.cfg.get("use_onnx", False):
                onnx_path = weights.replace(".pt", ".onnx")
                try:
                    self.model = YOLO(onnx_path)
                    logger.info("Loaded ONNX player detector: %s", onnx_path)
                except Exception:
                    logger.info("ONNX not found; using PyTorch weights")
            logger.info("Loaded player detector: %s conf=%.2f", weights, self.conf_threshold)
        except Exception as exc:
            logger.warning("YOLO unavailable (%s); using motion heuristic", exc)
            self.model = None

    def detect(self, frame: np.ndarray, court: CourtState | None = None) -> list[PlayerDetection]:
        if self.phone_preprocess:
            frame = preprocess_phone_frame(frame, denoise=self.pcfg.get("denoise", True))
        if self.model is not None:
            dets = self._detect_yolo(frame, court)
        else:
            dets = self._detect_heuristic(frame)
        self._history.append(dets)
        if not dets and len(self._history) >= 2:
            return self._history[-2]
        return dets

    def _detect_yolo(self, frame: np.ndarray, court: CourtState | None) -> list[PlayerDetection]:
        results = self.model.predict(
            frame, classes=[0], conf=self.conf_threshold, iou=0.4, verbose=False, device=self.device
        )
        detections: list[PlayerDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bbox = BBox(x1, y1, x2, y2, conf)
                if not self._valid_person_box(bbox, frame.shape):
                    continue
                court_xy = None
                if court and court.homography:
                    from backend.court.court_detector import CourtDetector

                    cd = CourtDetector.__new__(CourtDetector)
                    court_xy = cd.pixel_to_court(*bbox.centroid, court)
                detections.append(PlayerDetection(track_id=-1, bbox=bbox, court_xy=court_xy))
        return detections[:4]

    def _detect_heuristic(self, frame: np.ndarray) -> list[PlayerDetection]:
        h, w = frame.shape[:2]
        fg = cv2.createBackgroundSubtractorMOG2().apply(frame)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: list[PlayerDetection] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < h * 0.08 or bh > h * 0.6:
                continue
            aspect = bh / max(bw, 1)
            if aspect < 1.2:
                continue
            dets.append(
                PlayerDetection(track_id=-1, bbox=BBox(float(x), float(y), float(x + bw), float(y + bh), 0.5))
            )
        dets.sort(key=lambda d: d.bbox.area, reverse=True)
        return dets[:2]

    def _valid_person_box(self, bbox: BBox, shape: tuple) -> bool:
        h, w = shape[:2]
        area = bbox.area
        if area < (h * w) * 0.002 or area > (h * w) * 0.35:
            return False
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        aspect = bh / max(bw, 1)
        return 1.0 <= aspect <= 4.5
