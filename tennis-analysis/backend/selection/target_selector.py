from __future__ import annotations

import cv2
import numpy as np

from backend.utils.types import PlayerDetection


class TargetSelector:
    """Stage 5: click-to-select + jersey color affinity."""

    def __init__(self):
        self.target_track_id: int | None = None
        self.jersey_hist: np.ndarray | None = None

    def select_by_click(
        self, click_xy: tuple[float, float], players: list[PlayerDetection], frame: np.ndarray
    ) -> int | None:
        if not players:
            return None
        cx, cy = click_xy
        best = min(
            players,
            key=lambda p: np.hypot(p.bbox.centroid[0] - cx, p.bbox.centroid[1] - cy),
        )
        self.target_track_id = best.track_id
        self.jersey_hist = self._extract_jersey_hist(frame, best.bbox)
        return self.target_track_id

    def select_auto_far_baseline(self, players: list[PlayerDetection]) -> int | None:
        if not players:
            return None
        best = max(players, key=lambda p: p.bbox.centroid[1])
        self.target_track_id = best.track_id
        return self.target_track_id

    def confirm_track(
        self, players: list[PlayerDetection], frame: np.ndarray | None = None
    ) -> int | None:
        if self.target_track_id is None:
            return self.select_auto_far_baseline(players)
        for p in players:
            if p.track_id == self.target_track_id:
                return self.target_track_id
        if frame is not None and self.jersey_hist is not None:
            best_id, best_sim = None, -1.0
            for p in players:
                hist = self._extract_jersey_hist(frame, p.bbox)
                sim = cv2.compareHist(self.jersey_hist, hist, cv2.HISTCMP_CORREL)
                if sim > best_sim:
                    best_sim, best_id = sim, p.track_id
            if best_sim > 0.5:
                self.target_track_id = best_id
                return best_id
        return self.select_auto_far_baseline(players)

    def _extract_jersey_hist(self, frame: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        crop = frame[max(0, y1) : y2, max(0, x1) : x2]
        if crop.size == 0:
            return np.zeros((180, 1), dtype=np.float32)
        torso = crop[: max(1, crop.shape[0] // 2), :]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        cv2.normalize(hist, hist)
        return hist
