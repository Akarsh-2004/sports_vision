"""Build interaction graph from world frames — replaces isolated stroke labels."""

from __future__ import annotations

from backend.intelligence.domain import InteractionType, ShotIntent
from backend.intelligence.geometry.entities import GeometryFrame, StrokeObservation
from backend.intelligence.interaction.graph import InteractionGraph, InteractionNode
from backend.intelligence.tactical.rules import infer_shot_intent, infer_stroke_context
from backend.intelligence.world.digital_twin import WorldFrame
from backend.utils.types import StrokeType


class InteractionBuilder:
    """Layer 4 — detect interactions, not just strokes."""

    def __init__(self):
        self.graph = InteractionGraph()
        self._last_hitter: int | None = None

    def process_frame(
        self,
        world: WorldFrame,
        stroke: StrokeObservation | None,
        wall_hit: str | None,
        ball_bounce: bool,
        hitter_id: int | None,
    ) -> list[InteractionNode]:
        if not world.active_play:
            return []

        created: list[InteractionNode] = []
        geo = world.geometry
        ball_pos = geo.ball.kinematics.position.as_tuple() if geo.ball else None
        ball_speed = geo.ball.kinematics.speed_kmh if geo.ball else 0.0

        if stroke and stroke.confidence > 0.45 and hitter_id is not None:
            intent = infer_shot_intent(stroke.stroke_type, geo, hitter_id)
            node = InteractionNode(
                frame_idx=geo.frame_idx,
                interaction_type=InteractionType.PLAYER_HIT,
                actor_id=hitter_id,
                position=ball_pos,
                stroke_type=stroke.stroke_type,
                shot_intent=intent,
                speed_kmh=ball_speed,
            )
            self.graph.add(node)
            created.append(node)
            self._last_hitter = hitter_id

        if wall_hit:
            itype = InteractionType.BALL_WALL_GLASS
            if wall_hit == "fence":
                itype = InteractionType.BALL_WALL_FENCE
            elif wall_hit == "net":
                itype = InteractionType.BALL_NET
            node = InteractionNode(
                frame_idx=geo.frame_idx,
                interaction_type=itype,
                position=ball_pos,
                speed_kmh=ball_speed,
                metadata={"after_player": self._last_hitter},
            )
            self.graph.add(node)
            created.append(node)

        if ball_bounce and ball_pos:
            node = InteractionNode(
                frame_idx=geo.frame_idx,
                interaction_type=InteractionType.BALL_GROUND,
                position=ball_pos,
                speed_kmh=ball_speed,
            )
            self.graph.add(node)
            created.append(node)

        if world.match_state.value == "point_over" and self._last_hitter:
            node = InteractionNode(
                frame_idx=geo.frame_idx,
                interaction_type=InteractionType.POINT_END,
                actor_id=self._last_hitter,
                outcome="point_over",
            )
            self.graph.add(node)
            created.append(node)

        return created
