"""Phase 9: ball-racket contact frame detection."""

from __future__ import annotations

import numpy as np

from backend.utils.types import BallContact, PlayerDetection, PoseKeypoints


class BallContactDetector:
    """Minimum racket-ball distance + acceleration spike."""

    def __init__(self, max_dist_px: float = 100.0, accel_threshold: float = 15.0):
        self.max_dist_px = max_dist_px
        self.accel_threshold = accel_threshold
        self.contacts: list[BallContact] = []
        self._prev_speed: float | None = None

    def detect(
        self,
        frame_idx: int,
        players: list[PlayerDetection],
        ball_xy: tuple[float, float],
        ball_speed_kmh: float,
        pose: PoseKeypoints | None,
        court_xy: tuple[float, float] | None,
    ) -> BallContact | None:
        accel_spike = False
        if self._prev_speed is not None:
            accel_spike = (ball_speed_kmh - self._prev_speed) > self.accel_threshold
        self._prev_speed = ball_speed_kmh

        bx, by = ball_xy
        best_id = None
        best_dist = float("inf")

        for p in players:
            cx, cy = p.bbox.centroid
            d = float(np.hypot(cx - bx, cy - by))
            if d < best_dist:
                best_dist = d
                best_id = p.track_id

        if pose and pose.keypoints:
            for key in ("right_wrist", "left_wrist", "RIGHT_WRIST", "LEFT_WRIST"):
                if key in pose.keypoints:
                    wx, wy, _ = pose.keypoints[key]
                    d = float(np.hypot(wx - bx, wy - by))
                    if d < best_dist:
                        best_dist = d
                        best_id = pose.track_id

        if best_id is None or best_dist > self.max_dist_px:
            return None
        if not accel_spike and best_dist > self.max_dist_px * 0.6:
            return None

        contact = BallContact(
            frame_idx=frame_idx,
            track_id=best_id,
            court_xy=court_xy or (0.0, 0.0),
            ball_speed_kmh=ball_speed_kmh,
        )
        self.contacts.append(contact)
        return contact
