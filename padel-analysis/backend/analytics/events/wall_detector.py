"""Phase 6: wall / net / ground collision detection (padel-specific)."""

from __future__ import annotations

import numpy as np

from backend.utils.types import WallEvent, WallHitType


class WallDetector:
    """
    Detect ball-wall interactions from trajectory in court coordinates.

    MVP: direction-change heuristics on smoothed ball path.
    Future: YOLO glass/fence/net segmentation.
    """

    def __init__(self, config: dict):
        self.direction_change_deg = config.get("ball", {}).get("wall_direction_change_deg", 45)
        self.court_length = config["court"]["court_length_m"]
        self.court_width = config["court"]["court_width_m"]
        self.wall_offset = config["court"].get("wall_offset_m", 3.0)
        self.events: list[WallEvent] = []

    def analyze_trajectory(
        self,
        trajectory: list[tuple[int, float, float]],
        ground_bounce_frames: set[int] | None = None,
    ) -> list[WallEvent]:
        """trajectory: (frame, court_x, court_y) in meters."""
        ground_bounce_frames = ground_bounce_frames or set()
        found: list[WallEvent] = []
        if len(trajectory) < 3:
            return found

        for i in range(2, len(trajectory)):
            f0, x0, y0 = trajectory[i - 2]
            f1, x1, y1 = trajectory[i - 1]
            f2, x2, y2 = trajectory[i]
            v1 = np.array([x1 - x0, y1 - y0], dtype=np.float64)
            v2 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 0.05 or n2 < 0.05:
                continue
            cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            if angle_deg < self.direction_change_deg:
                continue

            hit_type = self._classify_hit(x1, y1, f1 in ground_bounce_frames)
            ev = WallEvent(frame_idx=f1, hit_type=hit_type, court_xy=(x1, y1))
            found.append(ev)
            self.events.append(ev)

        return found

    def _classify_hit(self, x: float, y: float, is_ground_bounce: bool) -> WallHitType:
        if is_ground_bounce:
            return WallHitType.GROUND
        if y < -0.5 or y > self.court_length + self.wall_offset:
            return WallHitType.GLASS
        if x < -0.3 or x > self.court_width + 0.3:
            return WallHitType.FENCE
        if abs(y - self.court_length / 2) < 0.4:
            return WallHitType.NET
        return WallHitType.UNKNOWN
