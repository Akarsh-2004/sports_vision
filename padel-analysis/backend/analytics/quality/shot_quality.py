"""Phase 11: shot quality on padel court."""

from __future__ import annotations

import numpy as np

from backend.court.court_detector import CourtDetector
from backend.utils.types import CourtState, ShotQuality


class ShotQualityEstimator:
    def __init__(self, config: dict):
        self.court_detector = CourtDetector(config)
        self.court_length = config["court"]["court_length_m"]
        self.court_width = config["court"]["court_width_m"]
        self.shots: list[ShotQuality] = []

    def estimate_shot(
        self,
        frame_idx: int,
        ball_xy: tuple[float, float],
        court: CourtState,
        ball_speed_kmh: float,
        is_bounce: bool = False,
        off_wall: bool = False,
    ) -> ShotQuality:
        court_xy = self.court_detector.pixel_to_court(ball_xy[0], ball_xy[1], court)
        in_court = False
        placement = 0.0
        zone = "unknown"

        if court_xy:
            cx, cy = court_xy
            in_court = self.court_detector.is_in_court(cx, cy)
            dist_sideline = min(cx, self.court_width - cx)
            dist_baseline = min(cy, self.court_length - cy)
            placement = float(
                np.clip(
                    (1.0 - dist_sideline / (self.court_width / 2)) * 0.5
                    + (1.0 - dist_baseline / (self.court_length / 2)) * 0.5,
                    0,
                    1,
                )
            )
            zone = self.court_detector.court_zone(cx, cy)

        sq = ShotQuality(
            frame_idx=frame_idx,
            power_kmh=ball_speed_kmh,
            placement_score=placement,
            in_court=in_court,
            landing_xy=court_xy,
            zone=zone,
            off_wall=off_wall,
        )
        if is_bounce or in_court or off_wall:
            self.shots.append(sq)
        return sq
