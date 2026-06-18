from __future__ import annotations

import numpy as np

from backend.court.court_detector import CourtDetector
from backend.utils.types import CourtState, ShotQuality


class ShotQualityEstimator:
    """Stage 11: power, placement, consistency proxies."""

    def __init__(self, config: dict):
        self.court_detector = CourtDetector(config)
        self.court_length = config["court"]["court_length_m"]
        self.singles_width = config["court"]["singles_width_m"]
        self.shots: list[ShotQuality] = []

    def estimate_shot(
        self,
        frame_idx: int,
        ball_xy: tuple[float, float],
        court: CourtState,
        ball_speed_kmh: float,
        is_bounce: bool = False,
    ) -> ShotQuality:
        court_xy = self.court_detector.pixel_to_court(ball_xy[0], ball_xy[1], court)
        in_court = False
        placement = 0.0
        zone = "unknown"

        if court_xy:
            cx, cy = court_xy
            in_court = self.court_detector.is_in_court(cx, cy)
            dist_sideline = min(cx, self.singles_width - cx)
            dist_baseline = min(cy, self.court_length - cy)
            placement = float(
                np.clip(
                    (1.0 - dist_sideline / (self.singles_width / 2)) * 0.5
                    + (1.0 - dist_baseline / (self.court_length / 2)) * 0.5,
                    0,
                    1,
                )
            )
            zone = self._zone_label(cx, cy)

        sq = ShotQuality(
            frame_idx=frame_idx,
            power_kmh=ball_speed_kmh,
            placement_score=placement,
            in_court=in_court,
            landing_xy=court_xy,
            zone=zone,
        )
        if is_bounce or in_court:
            self.shots.append(sq)
        return sq

    def consistency_score(self, window: int = 20) -> float:
        recent = self.shots[-window:]
        if not recent:
            return 0.0
        in_count = sum(1 for s in recent if s.in_court)
        return in_count / len(recent)

    def aggression_index(self) -> float:
        if not self.shots:
            return 0.0
        speeds = [s.power_kmh for s in self.shots]
        speed_pct = np.mean(speeds) / 200.0 if speeds else 0
        depth_pct = np.mean([s.placement_score for s in self.shots])
        return float(np.clip(0.3 * speed_pct + 0.3 * depth_pct + 0.4 * self.consistency_score(), 0, 1))

    def _zone_label(self, cx: float, cy: float) -> str:
        mid_x = self.singles_width / 2
        if cx < mid_x - 1:
            side = "ad"
        elif cx > mid_x + 1:
            side = "deuce"
        else:
            side = "T"
        if cy < self.court_length * 0.4:
            depth = "net"
        elif cy > self.court_length * 0.75:
            depth = "deep"
        else:
            depth = "mid"
        return f"{side}_{depth}"
