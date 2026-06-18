from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from backend.court.calibration import load_calibration
from backend.court.geometry import CANONICAL_COURT
from backend.utils.logging import get_logger
from backend.utils.types import CourtState

logger = get_logger(__name__)


class CourtDetector:
    """Stage 2: keypoint homography with manual calibration fallback."""

    def __init__(self, config: dict):
        court_cfg = config["court"]
        self.recompute_interval = court_cfg["recompute_interval"]
        self.court_length = court_cfg["court_length_m"]
        self.court_width = court_cfg["court_width_m"]
        self.singles_width = court_cfg["singles_width_m"]
        self.use_calibration = court_cfg.get("use_manual_calibration", True)
        self.video_stem: str | None = None
        self._cached_h: np.ndarray | None = None
        self._ema_alpha = 0.3

    def set_video_source(self, video_path: str) -> None:
        from pathlib import Path

        self.video_stem = Path(video_path).stem
        if self.use_calibration and self.video_stem:
            cal = load_calibration(self.video_stem)
            if cal is not None:
                self._cached_h = cal
                logger.info("Loaded manual court calibration for %s", self.video_stem)

    def detect(self, frame: np.ndarray, frame_idx: int) -> CourtState:
        if self._cached_h is not None and frame_idx % self.recompute_interval != 0:
            return CourtState(
                frame_idx=frame_idx,
                homography=self._cached_h.tolist(),
                confidence=0.85,
                zone="full_court",
                lines_detected=4,
                valid_for_analytics=True,
            )

        keypoints = self._detect_white_line_keypoints(frame)
        h_mat, conf = self._homography_from_keypoints(frame, keypoints)
        if h_mat is None:
            lines = self._detect_lines(frame)
            h_mat, conf = self._compute_homography(frame, lines)

        if h_mat is not None:
            if self._cached_h is not None:
                h_mat = self._ema_alpha * h_mat + (1 - self._ema_alpha) * self._cached_h
            self._cached_h = h_mat

        return CourtState(
            frame_idx=frame_idx,
            homography=self._cached_h.tolist() if self._cached_h is not None else None,
            confidence=conf,
            zone=self._infer_zone(self._cached_h),
            lines_detected=len(keypoints) if keypoints else 0,
            valid_for_analytics=conf >= 0.35 and self._cached_h is not None,
        )

    def pixel_to_court(self, x: float, y: float, court: CourtState) -> tuple[float, float] | None:
        if court.homography is None:
            return None
        H = np.array(court.homography, dtype=np.float64)
        pt = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def court_to_pixel(self, cx: float, cy: float, court: CourtState) -> tuple[float, float] | None:
        if court.homography is None:
            return None
        H = np.array(court.homography, dtype=np.float64)
        H_inv = np.linalg.inv(H)
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H_inv)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def is_in_court(self, cx: float, cy: float) -> bool:
        return 0 <= cx <= self.singles_width and 0 <= cy <= self.court_length

    def is_in_service_box(self, cx: float, cy: float, near_end: bool = True) -> bool:
        """Singles service box (simplified)."""
        if not (0 <= cx <= self.singles_width):
            return False
        if near_end:
            return 0 <= cy <= 6.4
        return self.court_length - 6.4 <= cy <= self.court_length

    def _detect_white_line_keypoints(self, frame: np.ndarray) -> list[tuple[float, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        corners = cv2.goodFeaturesToTrack(white, maxCorners=80, qualityLevel=0.01, minDistance=20)
        if corners is None:
            return []
        pts = [(float(c[0][0]), float(c[0][1])) for c in corners]
        h, w = frame.shape[:2]
        # Prefer points in lower half (court surface) with spread
        pts = [p for p in pts if p[1] > h * 0.25]
        if len(pts) < 4:
            return pts
        pts.sort(key=lambda p: p[1])
        near = pts[: len(pts) // 3]
        far = pts[-len(pts) // 3 :]
        if not near or not far:
            return pts[:4]
        near.sort(key=lambda p: p[0])
        far.sort(key=lambda p: p[0])
        return [near[0], near[-1], far[0], far[-1]]

    def _homography_from_keypoints(
        self, frame: np.ndarray, keypoints: list[tuple[float, float]]
    ) -> tuple[np.ndarray | None, float]:
        if len(keypoints) < 4:
            return None, 0.0
        src = np.float32(keypoints[:4])
        H, mask = cv2.findHomography(src, CANONICAL_COURT, cv2.RANSAC, 5.0)
        conf = float(mask.sum() / 4.0) if mask is not None else 0.4
        return H, conf

    def _detect_lines(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=15)
        if raw is None:
            return []
        lines = [tuple(map(int, l[0])) for l in raw]
        return self._cluster_lines(lines)

    def _cluster_lines(self, lines: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if len(lines) < 2:
            return lines
        features = []
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1)
            mid = ((x1 + x2) / 2, (y1 + y2) / 2)
            features.append([mid[0] / 100, mid[1] / 100, np.sin(angle), np.cos(angle)])
        clustering = DBSCAN(eps=0.35, min_samples=1).fit(features)
        merged: list[tuple[int, int, int, int]] = []
        for label in set(clustering.labels_):
            cluster = [lines[i] for i, l in enumerate(clustering.labels_) if l == label]
            xs = [p for ln in cluster for p in (ln[0], ln[2])]
            ys = [p for ln in cluster for p in (ln[1], ln[3])]
            merged.append((min(xs), min(ys), max(xs), max(ys)))
        return merged

    def _compute_homography(
        self, frame: np.ndarray, lines: list[tuple[int, int, int, int]]
    ) -> tuple[np.ndarray | None, float]:
        h, w = frame.shape[:2]
        src = np.float32(
            [
                [w * 0.2, h * 0.55],
                [w * 0.8, h * 0.55],
                [w * 0.1, h * 0.95],
                [w * 0.9, h * 0.95],
            ]
        )
        if len(lines) >= 4:
            horizontals = []
            for x1, y1, x2, y2 in lines:
                if abs(y2 - y1) < abs(x2 - x1) * 0.3:
                    horizontals.append((y1 + y2) / 2)
            if len(horizontals) >= 2:
                horizontals.sort()
                near_y, far_y = horizontals[0], horizontals[-1]
                src = np.float32(
                    [
                        [w * 0.25, near_y],
                        [w * 0.75, near_y],
                        [w * 0.1, far_y],
                        [w * 0.9, far_y],
                    ]
                )
        H, mask = cv2.findHomography(src, CANONICAL_COURT, cv2.RANSAC, 5.0)
        conf = float(mask.sum() / 4.0) if mask is not None else 0.5
        return H, conf

    def _infer_zone(self, H: np.ndarray | None) -> str:
        return "full_court" if H is not None else "unknown"
