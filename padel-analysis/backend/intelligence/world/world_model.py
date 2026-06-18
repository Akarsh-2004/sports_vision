"""
World Model — continuously updated match state (source of truth).

Every downstream module reads from WorldModel.current — never recomputes in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.intelligence.confidence.propagation import ConfidenceTracker, ModuleConfidence
from backend.intelligence.court.semantic_regions import SemanticRegion, classify_region
from backend.intelligence.domain import MatchState
from backend.intelligence.geometry.entities import GeometryFrame
from backend.intelligence.interaction.graph import InteractionNode
from backend.intelligence.match_state.engine import MatchStateEngine
from backend.intelligence.physics.ball_physics import BallPhysicsEngine, PhysicsState
from backend.intelligence.shot.understanding import ShotUnderstanding


@dataclass
class CourtWorld:
    homography_valid: bool = False
    homography_confidence: float = 0.0
    width_m: float = 10.0
    length_m: float = 20.0
    net_y_m: float = 10.0
    wall_offset_m: float = 3.0


@dataclass
class PlayerWorld:
    track_id: int
    team_id: int | None
    position: tuple[float, float]
    speed_kmh: float
    direction_deg: float
    region: SemanticRegion
    zone: str
    side: str
    confidence: float = 0.9


@dataclass
class BallWorld:
    position: tuple[float, float]
    velocity_mps: tuple[float, float]
    speed_kmh: float
    visible: bool
    confidence: float
    physics: PhysicsState | None = None
    height_m: float = 0.0


@dataclass
class MatchWorld:
    state: MatchState = MatchState.IDLE
    rally_id: int = 0
    possession_player_id: int | None = None
    pressure: str = "neutral"  # low | medium | high
    active_play: bool = True
    frame_in_rally: int = 0


@dataclass
class EventsWorld:
    """Recent events — full history in interaction graph."""
    recent_hits: list[InteractionNode] = field(default_factory=list)
    recent_bounces: list[InteractionNode] = field(default_factory=list)
    recent_walls: list[InteractionNode] = field(default_factory=list)


@dataclass
class WorldSnapshot:
    """Complete world state at one frame — the digital twin cell."""

    frame_idx: int
    timestamp_s: float
    court: CourtWorld
    players: list[PlayerWorld]
    ball: BallWorld | None
    match: MatchWorld
    events: EventsWorld
    shot: ShotUnderstanding | None = None
    confidence: ModuleConfidence = field(default_factory=ModuleConfidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_idx,
            "time_s": round(self.timestamp_s, 2),
            "court": {
                "homography": self.court.homography_valid,
                "confidence": round(self.court.homography_confidence, 3),
            },
            "players": [
                {
                    "id": p.track_id,
                    "team": p.team_id,
                    "pos": list(p.position),
                    "speed_kmh": round(p.speed_kmh, 1),
                    "region": p.region.value,
                }
                for p in self.players
            ],
            "ball": (
                {
                    "pos": list(self.ball.position),
                    "speed_kmh": round(self.ball.speed_kmh, 1),
                    "confidence": round(self.ball.confidence, 3),
                }
                if self.ball
                else None
            ),
            "match": {
                "state": self.match.state.value,
                "rally_id": self.match.rally_id,
                "possession": self.match.possession_player_id,
                "pressure": self.match.pressure,
            },
            "shot": self.shot.to_dict() if self.shot else None,
            "confidence": self.confidence.to_dict(),
        }


class WorldModel:
    """
    Continuously updated world state — highest priority architecture component.

    Vision sensors write via `update()`; all analytics read via `current` or `history`.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fps = config["pipeline"]["target_fps"]
        court = config["court"]
        self.history: list[WorldSnapshot] = []
        self._state_engine = MatchStateEngine(config)
        self._physics = BallPhysicsEngine(
            court_width=court["court_width_m"],
            court_length=court["court_length_m"],
            wall_offset=court.get("wall_offset_m", 3.0),
        )
        self._confidence_tracker = ConfidenceTracker()
        self._all_interactions: list[InteractionNode] = []
        self._rally_id = 0
        self._frame_in_rally = 0
        self._prev_state = MatchState.IDLE

    @property
    def current(self) -> WorldSnapshot | None:
        return self.history[-1] if self.history else None

    def update(
        self,
        geometry: GeometryFrame,
        physics_state: PhysicsState | None,
        conf: ModuleConfidence,
        stroke: ShotUnderstanding | None = None,
        wall_hit: str | None = None,
        ball_bounce: bool = False,
        interactions: list[InteractionNode] | None = None,
    ) -> WorldSnapshot:
        court_w = CourtWorld(
            homography_valid=geometry.court.homography_valid,
            homography_confidence=geometry.court.homography_confidence,
            width_m=geometry.court.width_m,
            length_m=geometry.court.length_m,
            net_y_m=geometry.court.net_y_m,
        )

        players_w = []
        for p in geometry.players:
            x, y = p.position
            players_w.append(
                PlayerWorld(
                    track_id=p.track_id,
                    team_id=p.team_id,
                    position=(x, y),
                    speed_kmh=p.kinematics.speed_kmh,
                    direction_deg=p.kinematics.direction_deg,
                    region=classify_region(x, y),
                    zone=p.zone.value,
                    side=p.side_role.value,
                    confidence=conf.players,
                )
            )

        ball_w = None
        if geometry.ball or physics_state:
            pos = physics_state.position if physics_state else geometry.ball.kinematics.position
            vel = physics_state.velocity if physics_state else None
            speed = geometry.ball.kinematics.speed_kmh if geometry.ball else 0
            ball_w = BallWorld(
                position=pos.as_tuple(),
                velocity_mps=(vel.x, vel.y) if vel else (0, 0),
                speed_kmh=speed,
                visible=physics_state.visible if physics_state else geometry.ball.visible,
                confidence=conf.ball,
                physics=physics_state,
            )

        match_state = self._state_engine.update(
            geometry,
            None,
            wall_hit,
            ball_bounce,
        )
        if match_state == MatchState.RALLY and self._prev_state != MatchState.RALLY:
            self._rally_id += 1
            self._frame_in_rally = 0
        elif match_state in (MatchState.RALLY, MatchState.NET_ATTACK, MatchState.WALL_EXCHANGE):
            self._frame_in_rally += 1
        else:
            self._frame_in_rally = 0
        self._prev_state = match_state

        possession = stroke.player_id if stroke else None
        pressure = stroke.pressure if stroke else "neutral"

        events = EventsWorld()
        if interactions:
            for n in interactions:
                self._all_interactions.append(n)
                if n.interaction_type.value == "player_hit":
                    events.recent_hits.append(n)
                elif "wall" in n.interaction_type.value or n.interaction_type.value == "ball_net":
                    events.recent_walls.append(n)
                elif n.interaction_type.value == "ball_ground":
                    events.recent_bounces.append(n)

        snap = WorldSnapshot(
            frame_idx=geometry.frame_idx,
            timestamp_s=geometry.timestamp_s,
            court=court_w,
            players=players_w,
            ball=ball_w,
            match=MatchWorld(
                state=match_state,
                rally_id=self._rally_id,
                possession_player_id=possession,
                pressure=pressure,
                active_play=geometry.active_play,
                frame_in_rally=self._frame_in_rally,
            ),
            events=events,
            shot=stroke,
            confidence=conf,
        )
        self.history.append(snap)
        self._confidence_tracker.record(conf)
        return snap

    @property
    def physics_engine(self) -> BallPhysicsEngine:
        return self._physics

    @property
    def all_interactions(self) -> list[InteractionNode]:
        return self._all_interactions

    def active_snapshots(self) -> list[WorldSnapshot]:
        return [s for s in self.history if s.match.active_play]

    def timeline_events(self) -> list[dict]:
        """Events for interactive dashboard timeline."""
        events = []
        for n in self._all_interactions:
            events.append(
                {
                    "frame": n.frame_idx,
                    "time_s": round(n.frame_idx / self.fps, 2),
                    "type": n.interaction_type.value,
                    "stroke": n.stroke_type.value if n.stroke_type else None,
                    "actor": n.actor_id,
                    "intent": n.shot_intent.value,
                }
            )
        for s in self.history:
            if s.shot:
                events.append(
                    {
                        "frame": s.frame_idx,
                        "time_s": round(s.timestamp_s, 2),
                        "type": "shot_understanding",
                        "stroke": s.shot.stroke.value,
                        "intent": s.shot.intent.value,
                        "epv_delta": round(s.shot.epv_after - s.shot.epv_before, 3),
                        "decision": s.shot.decision_quality.value,
                    }
                )
        events.sort(key=lambda e: e["frame"])
        return events

    def self_evaluation(self) -> dict:
        avg = self._confidence_tracker.match_average()
        return {
            "module_confidence": avg.to_dict(),
            "reliability_note": self._confidence_tracker.reliability_note(),
            "frames_processed": len(self.history),
            "interactions_total": len(self._all_interactions),
        }

    def summary(self) -> dict:
        active = self.active_snapshots()
        states: dict[str, int] = {}
        for s in active:
            states[s.match.state.value] = states.get(s.match.state.value, 0) + 1
        return {
            "total_frames": len(self.history),
            "active_frames": len(active),
            "state_distribution": states,
            "rallies_detected": self._rally_id,
            "interaction_count": len(self._all_interactions),
        }
