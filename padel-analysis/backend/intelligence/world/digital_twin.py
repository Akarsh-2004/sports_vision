"""Layer 3+4 hub — Digital Twin: source of truth for every frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.intelligence.domain import Formation, MatchState
from backend.intelligence.geometry.entities import GeometryFrame, StrokeObservation
from backend.intelligence.interaction.graph import InteractionNode
from backend.intelligence.match_state.engine import MatchStateEngine
from backend.intelligence.tactical.insights import TacticalSnapshot


@dataclass
class WorldFrame:
    """
    Complete world model for one frame.

    Vision sensors write geometry; reasoning layers write state + interactions.
    """

    geometry: GeometryFrame
    match_state: MatchState = MatchState.IDLE
    formation_team_a: Formation = Formation.UNKNOWN
    formation_team_b: Formation = Formation.UNKNOWN
    recent_interactions: list[InteractionNode] = field(default_factory=list)
    tactical: TacticalSnapshot | None = None
    stroke: StrokeObservation | None = None

    @property
    def frame_idx(self) -> int:
        return self.geometry.frame_idx

    @property
    def timestamp_s(self) -> float:
        return self.geometry.timestamp_s

    @property
    def active_play(self) -> bool:
        return self.geometry.active_play

    def to_dict(self) -> dict[str, Any]:
        g = self.geometry
        return {
            "frame_idx": g.frame_idx,
            "timestamp_s": g.timestamp_s,
            "active_play": g.active_play,
            "match_state": self.match_state.value,
            "players": [
                {
                    "id": p.track_id,
                    "team": p.team_id,
                    "position": p.position,
                    "speed_kmh": round(p.kinematics.speed_kmh, 1),
                    "zone": p.zone.value,
                    "side": p.side_role.value,
                }
                for p in g.players
            ],
            "ball": (
                {
                    "position": g.ball.kinematics.position.as_tuple(),
                    "speed_kmh": round(g.ball.kinematics.speed_kmh, 1),
                    "visible": g.ball.visible,
                }
                if g.ball
                else None
            ),
            "interactions": [n.to_dict() for n in self.recent_interactions[-5:]],
        }


class DigitalTwin:
    """
    Maintains the full digital world across a match.

    Everything downstream reads from here — never raw pixels.
    """

    def __init__(self, config: dict):
        self.config = config
        self.frames: list[WorldFrame] = []
        self._state_engine = MatchStateEngine(config)
        self._interaction_buffer: list[InteractionNode] = []

    def push(
        self,
        geometry: GeometryFrame,
        stroke: StrokeObservation | None = None,
        wall_hit: str | None = None,
        ball_bounce: bool = False,
    ) -> WorldFrame:
        state = self._state_engine.update(geometry, stroke, wall_hit, ball_bounce)
        wf = WorldFrame(
            geometry=geometry,
            match_state=state,
            stroke=stroke,
            recent_interactions=list(self._interaction_buffer[-10:]),
        )
        self.frames.append(wf)
        return wf

    def add_interaction(self, node: InteractionNode) -> None:
        self._interaction_buffer.append(node)
        if self.frames:
            self.frames[-1].recent_interactions.append(node)

    def attach_tactical(self, frame_idx: int, snapshot: TacticalSnapshot) -> None:
        for wf in self.frames:
            if wf.frame_idx == frame_idx:
                wf.tactical = snapshot
                break

    def active_frames(self) -> list[WorldFrame]:
        return [f for f in self.frames if f.active_play]

    def export_timeline(self) -> list[dict]:
        return [f.to_dict() for f in self.frames if f.active_play]

    def summary(self) -> dict:
        active = self.active_frames()
        states: dict[str, int] = {}
        for f in active:
            states[f.match_state.value] = states.get(f.match_state.value, 0) + 1
        return {
            "total_frames": len(self.frames),
            "active_frames": len(active),
            "state_distribution": states,
            "interaction_count": len(self._interaction_buffer),
        }
