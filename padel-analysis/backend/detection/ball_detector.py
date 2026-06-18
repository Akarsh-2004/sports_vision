from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import BallDetection

logger = get_logger(__name__)


class BallDetector:
    """Stage 6: heuristic + motion-streak detection for fast balls."""

    def __init__(self, config: dict):
        self.mode = config["models"].get("ball_detector", "heuristic")
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=3)
        self.prev_gray: np.ndarray | None = None
        self.model = None
        self.bounce_threshold = config.get("ball", {}).get("bounce_speed_drop", 0.4)
        self._prev_speed = 0.0
        if self.mode == "tracknet":
            self._try_load_tracknet()

    def _try_load_tracknet(self) -> None:
        logger.info("TrackNet weights not bundled; using heuristic + streak detection")
        self.mode = "heuristic"

    def detect(self, frame: np.ndarray) -> BallDetection:
        self.frame_buffer.append(frame.copy())
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
