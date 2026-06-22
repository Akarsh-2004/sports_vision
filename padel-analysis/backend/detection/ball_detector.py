from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import BallDetection

logger = get_logger(__name__)


class BallDetector:
    """
    Stage 6: Padel ball detection.

    Modes (set via config['models']['ball_detector']):
      - 'yolo'      → fine-tuned YOLOv11n on padel ball dataset (recommended)
      - 'heuristic' → color mask + motion streak (fallback, no weights needed)

    With 'yolo' mode:
      - Loads weights from config['models']['ball_yolo_weights']
        (default: weights/padel_ball_yolo11n.pt)
      - Falls back to heuristic if weights not found
      - YOLO confidence is passed directly to the Kalman tracker
      - Fuses YOLO detection with motion streak for extra robustness on fast balls
    """

    def __init__(self, config: dict):
        self.mode = config["models"].get("ball_detector", "heuristic")
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=3)
        self.prev_gray: np.ndarray | None = None
        self.model = None
        self.bounce_threshold = config.get("ball", {}).get("bounce_speed_drop", 0.4)
        self._prev_speed = 0.0
        self._device = self._resolve_device(config.get("models", {}).get("device", "auto"))

        # Weight path resolution
        self._weights_path: str | None = None
        if self.mode == "yolo":
            raw = config.get("models", {}).get(
                "ball_yolo_weights", "weights/padel_ball_yolo11n.pt"
            )
            # Support both absolute and relative-to-project paths
            p = Path(raw)
            if not p.is_absolute():
                # Try relative to the project root (parent of scripts/)
                project_root = Path(__file__).resolve().parents[2]
                p = project_root / raw
            if p.exists():
                self._weights_path = str(p)
                self._load_yolo(self._weights_path)
            else:
                logger.warning(
                    "Ball YOLO weights not found at '%s'. "
                    "Run: python scripts/train_ball_detector.py\n"
                    "Falling back to heuristic detector.",
                    p,
                )
                self.mode = "heuristic"

    # ── device resolution ─────────────────────────────────────────────────────

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_yolo(self, weights: str) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(weights)
            logger.info(
                "Ball YOLO loaded: %s  device=%s", Path(weights).name, self._device
            )
        except Exception as exc:
            logger.warning(
                "Failed to load ball YOLO (%s). Falling back to heuristic.", exc
            )
            self.model = None
            self.mode = "heuristic"

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> BallDetection:
        self.frame_buffer.append(frame.copy())

        if self.mode == "yolo" and self.model is not None:
            det = self._detect_yolo(frame)
            # Fuse with motion streak for extra robustness on very fast balls
            if not det.visible or det.confidence < 0.35:
                streak = self._detect_motion_streak(frame)
                if streak.visible and streak.confidence > det.confidence:
                    # Blend positions weighted by confidence
                    if det.visible:
                        w1 = det.confidence
                        w2 = streak.confidence * 0.7  # down-weight heuristic
                        total = w1 + w2
                        blended_x = (det.x * w1 + streak.x * w2) / total
                        blended_y = (det.y * w1 + streak.y * w2) / total
                        blended_c = max(det.confidence, streak.confidence * 0.7)
                        return BallDetection(blended_x, blended_y, blended_c, True)
                    return streak
            return det

        # Heuristic path
        det = self._detect_heuristic(frame)
        if not det.visible:
            streak = self._detect_motion_streak(frame)
            if streak.visible:
                return streak
        return det

    def detect_bounce(self, speed: float) -> bool:
        bounced = self._prev_speed > 5.0 and speed < self._prev_speed * self.bounce_threshold
        self._prev_speed = speed
        return bounced

    # ── YOLO inference ────────────────────────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> BallDetection:
        """Run YOLOv11n inference and return highest-confidence ball detection."""
        try:
            results = self.model.predict(
                frame,
                conf=0.20,          # Low threshold — Kalman will handle noise
                iou=0.4,
                imgsz=640,
                verbose=False,
                device=self._device,
                classes=[0],        # 'sports ball' class (dataset has nc=1)
                max_det=5,
            )
        except Exception as exc:
            logger.debug("YOLO ball inference error: %s", exc)
            return BallDetection(0, 0, 0.0, False)

        best_conf = 0.0
        best_det = BallDetection(0, 0, 0.0, False)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf <= best_conf:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                # Sanity: padel ball is small — reject huge boxes
                frame_h, frame_w = frame.shape[:2]
                if w > frame_w * 0.12 or h > frame_h * 0.12:
                    continue
                # Reject tiny detections below 3px (noise)
                if w < 3 or h < 3:
                    continue
                best_conf = conf
                best_det = BallDetection(cx, cy, conf, True)

        return best_det

    # ── heuristic detectors ───────────────────────────────────────────────────

    def _detect_motion_streak(self, frame: np.ndarray) -> BallDetection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return BallDetection(0, 0, 0.0, False)
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, motion = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Fast ball motion blur → elongated streak
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8 or area > 2000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect < 2.0:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            score = aspect * min(area, 200)
            if score > best_score:
                best_score = score
                best = (cx, cy, min(1.0, score / 150.0))
        if best:
            return BallDetection(x=best[0], y=best[1], confidence=best[2] * 0.85, visible=True)
        return BallDetection(0, 0, 0.0, False)

    def _detect_heuristic(self, frame: np.ndarray) -> BallDetection:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([40, 255, 255]))
        mask_w = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        mask = cv2.bitwise_or(mask_y, mask_w)
        if len(self.frame_buffer) >= 2:
            diff = cv2.absdiff(self.frame_buffer[-1], self.frame_buffer[-2])
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, motion = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, motion)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3 or area > 800:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (np.pi * radius * radius + 1e-6)
            if circularity < 0.35:
                continue
            score = circularity * min(area, 50)
            if score > best_score:
                best_score = score
                best = (x, y, min(1.0, score / 30.0))
        if best:
            return BallDetection(x=best[0], y=best[1], confidence=best[2], visible=True)
        return BallDetection(x=0, y=0, confidence=0.0, visible=False)

    def _detect_tracknet(self) -> BallDetection:
        return BallDetection(x=0, y=0, confidence=0.0, visible=False)
