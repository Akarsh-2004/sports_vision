"""Project vision detections into court-coordinate entities."""

from __future__ import annotations

import math

import numpy as np

from backend.court.court_detector import CourtDetector
from backend.intelligence.domain import CourtZone, SideRole
from backend.intelligence.geometry.entities import (
    BallEntity,
    CourtGeometry,
    GeometryFrame,
    Kinematics,
    PlayerEntity,
    Vec2,
)
from backend.utils.types import CourtState, PlayerDetection


class GeometryProjector:
    """Layer 2 — converts pixel sensors → court-coordinate entities."""

    def __init__(self, config: dict):
        self.court_detector = CourtDetector(config)
        court = config["court"]
        self.court_geom = CourtGeometry(
            width_m=court["court_width_m"],
            length_m=court["court_length_m"],
            net_y_m=court["court_length_m"] / 2,
            service_depth_m=3.0,
            wall_offset_m=court.get("wall_offset_m", 3.0),
        )
        self.fps = config["pipeline"]["target_fps"]
        self._prev_player_pos: dict[int, tuple[float, float]] = {}
        self._prev_ball_pos: tuple[float, float] | None = None
        self._prev_ball_speed: float = 0.0

    def project_frame(
        self,
        frame_idx: int,
        players: list[PlayerDetection],
        ball_xy: tuple[float, float],
        ball_conf: float,
        ball_speed_kmh: float,
        ball_visible: bool,
        court_state: CourtState,
        active_play: bool,
    ) -> GeometryFrame:
        self.court_geom.homography_valid = court_state.valid_for_analytics
        self.court_geom.homography_confidence = court_state.confidence

        player_entities: list[PlayerEntity] = []
        for p in players:
            cx, cy = self._to_court(p.bbox.centroid, court_state)
            if cx is None:
                continue
            kin = self._kinematics(p.track_id, cx, cy)
            zone = self.court_geom.zone_at(cx, cy)
            role = self.court_geom.side_role_for(cx)
            player_entities.append(
                PlayerEntity(
                    track_id=p.track_id,
                    team_id=p.team_id,
                    kinematics=kin,
                    zone=zone,
                    side_role=role,
                    pixel_bbox=(p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2),
                )
            )

        ball_entity = None
        if ball_conf > 0.05 or ball_visible:
            bx, by = self._to_court(ball_xy, court_state) or (ball_xy[0] * 0.02, ball_xy[1] * 0.02)
            bkin = self._ball_kinematics(bx, by, ball_speed_kmh, ball_conf)
            ball_entity = BallEntity(kinematics=bkin, visible=ball_visible)

        return GeometryFrame(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps,
            court=self.court_geom,
            players=player_entities,
            ball=ball_entity,
            active_play=active_play,
        )

    def _to_court(
        self, pixel_xy: tuple[float, float], court_state: CourtState
    ) -> tuple[float, float] | None:
        return self.court_detector.pixel_to_court(pixel_xy[0], pixel_xy[1], court_state)

    def _kinematics(self, track_id: int, x: float, y: float) -> Kinematics:
        prev = self._prev_player_pos.get(track_id)
        speed_mps = 0.0
        direction = 0.0
        accel = 0.0
        if prev:
            dx, dy = x - prev[0], y - prev[1]
            speed_mps = float(np.hypot(dx, dy) * self.fps)
            direction = float(math.degrees(math.atan2(dy, dx)))
            accel = speed_mps  # simplified; refined in world layer
        self._prev_player_pos[track_id] = (x, y)
        return Kinematics(
            position=Vec2(x, y),
            speed_mps=speed_mps,
            speed_kmh=speed_mps * 3.6,
            direction_deg=direction,
            acceleration_mps2=accel,
        )

    def _ball_kinematics(self, x: float, y: float, speed_kmh: float, conf: float) -> Kinematics:
        speed_mps = speed_kmh / 3.6
        direction = 0.0
        accel = 0.0
        if self._prev_ball_pos:
            dx, dy = x - self._prev_ball_pos[0], y - self._prev_ball_pos[1]
            direction = float(math.degrees(math.atan2(dy, dx)))
            prev_speed = self._prev_ball_speed / 3.6
            accel = (speed_mps - prev_speed) * self.fps
        self._prev_ball_pos = (x, y)
        self._prev_ball_speed = speed_kmh
        return Kinematics(
            position=Vec2(x, y),
            speed_mps=speed_mps,
            speed_kmh=speed_kmh,
            direction_deg=direction,
            acceleration_mps2=accel,
            confidence=conf,
        )
