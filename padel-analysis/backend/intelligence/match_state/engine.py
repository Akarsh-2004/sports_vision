"""Layer 3 — match state finite-state machine."""

from __future__ import annotations

from backend.intelligence.domain import CourtZone, MatchState
from backend.intelligence.geometry.entities import GeometryFrame, StrokeObservation
from backend.utils.types import StrokeType


class MatchStateEngine:
    """
    Every frame belongs to a match state.

    Transitions driven by ball activity, player zones, and stroke context.
    """

    def __init__(self, config: dict):
        self.fps = config["pipeline"]["target_fps"]
        self._current = MatchState.IDLE
        self._rally_frames = 0
        self._serve_cooldown = 0
        self._wall_streak = 0

    @property
    def state(self) -> MatchState:
        return self._current

    def update(
        self,
        geo: GeometryFrame,
        stroke: StrokeObservation | None,
        wall_hit: str | None,
        ball_bounce: bool,
    ) -> MatchState:
        if not geo.active_play:
            self._current = MatchState.DEAD_TIME
            return self._current

        ball = geo.ball
        ball_active = ball is not None and (ball.visible or ball.kinematics.speed_kmh > 3)
        max_player_speed = max((p.kinematics.speed_kmh for p in geo.players), default=0)

        if wall_hit:
            self._wall_streak += 1
        else:
            self._wall_streak = max(0, self._wall_streak - 1)

        if stroke and stroke.stroke_type == StrokeType.SERVE:
            self._current = MatchState.SERVE
            self._rally_frames = 0
            self._serve_cooldown = int(2 * self.fps)
            return self._current

        if self._serve_cooldown > 0:
            self._serve_cooldown -= 1
            if ball_active and stroke:
                self._current = MatchState.RETURN
                return self._current

        if stroke and stroke.stroke_type == StrokeType.LOB:
            self._current = MatchState.LOB_DEFENSE
            self._rally_frames += 1
            return self._current

        net_attackers = sum(1 for p in geo.players if p.zone == CourtZone.NET and p.kinematics.speed_kmh > 5)
        if net_attackers >= 1 and ball_active:
            self._current = MatchState.NET_ATTACK
            self._rally_frames += 1
            return self._current

        if self._wall_streak >= 2:
            self._current = MatchState.WALL_EXCHANGE
            self._rally_frames += 1
            return self._current

        if ball_active or ball_bounce or (stroke and stroke.confidence > 0.5):
            self._current = MatchState.RALLY
            self._rally_frames += 1
            return self._current

        if max_player_speed < 1.5 and not ball_active:
            if self._rally_frames > int(1.5 * self.fps):
                self._current = MatchState.POINT_OVER
            else:
                self._current = MatchState.RESET
            self._rally_frames = 0
            return self._current

        if self._current in (MatchState.POINT_OVER, MatchState.RESET):
            self._current = MatchState.IDLE

        return self._current
