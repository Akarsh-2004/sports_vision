"""Layer 4 — interaction event graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.intelligence.domain import InteractionType, ShotIntent
from backend.utils.types import StrokeType


@dataclass
class InteractionNode:
    """One node in the rally interaction graph."""

    frame_idx: int
    interaction_type: InteractionType
    actor_id: int | None = None  # player track_id when applicable
    position: tuple[float, float] | None = None
    stroke_type: StrokeType | None = None
    shot_intent: ShotIntent = ShotIntent.UNKNOWN
    speed_kmh: float = 0.0
    outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_idx,
            "type": self.interaction_type.value,
            "actor": self.actor_id,
            "position": self.position,
            "stroke": self.stroke_type.value if self.stroke_type else None,
            "intent": self.shot_intent.value,
            "speed_kmh": round(self.speed_kmh, 1),
            "outcome": self.outcome,
            **self.metadata,
        }


@dataclass
class RallyGraph:
    """Ordered interaction chain for one rally."""

    rally_id: int
    start_frame: int
    end_frame: int
    nodes: list[InteractionNode] = field(default_factory=list)

    def chain_summary(self) -> str:
        parts: list[str] = []
        for n in self.nodes:
            if n.stroke_type:
                parts.append(n.stroke_type.value)
            elif n.interaction_type.value.startswith("ball_"):
                parts.append(n.interaction_type.value.replace("ball_", ""))
        return " → ".join(parts) if parts else "empty"

    def to_dict(self) -> dict[str, Any]:
        from backend.intelligence.interaction.rally_graph_builder import RallyGraphBuilder

        return {
            "rally_id": self.rally_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "chain": RallyGraphBuilder.chain_label(self.nodes),
            "nodes": [n.to_dict() for n in self.nodes],
        }


class InteractionGraph:
    """Accumulates interaction nodes and builds per-rally graphs."""

    def __init__(self):
        self.nodes: list[InteractionNode] = []
        self.rallies: list[RallyGraph] = []

    def add(self, node: InteractionNode) -> None:
        self.nodes.append(node)

    def build_rally_graphs(self, rally_segments: list[tuple[int, int, int]]) -> list[RallyGraph]:
        graphs: list[RallyGraph] = []
        for i, (start, end, _shots) in enumerate(rally_segments):
            nodes = [n for n in self.nodes if start <= n.frame_idx <= end]
            g = RallyGraph(rally_id=i, start_frame=start, end_frame=end, nodes=nodes)
            graphs.append(g)
        self.rallies = graphs
        return graphs
