"""Build real rally graphs from world model interactions."""

from __future__ import annotations

from backend.intelligence.interaction.graph import InteractionGraph, InteractionNode, RallyGraph
from backend.intelligence.world.world_model import WorldModel
from backend.utils.types import RallySegment


class RallyGraphBuilder:
    """
    Segment interactions into rallies, dedupe burst detections, build chains.

    Player → Ball → Wall → Player → Net → Ground
    """

    def __init__(self, fps: float, gap_s: float = 2.0, hit_gap_s: float = 0.5, min_shots: int = 2):
        self.fps = fps
        self.gap_frames = int(gap_s * fps)
        self.hit_gap_frames = max(1, int(hit_gap_s * fps))
        self.min_rally_frames = max(1, int(1.0 * fps))
        self.min_shots = min_shots

    def dedupe_hits(self, interactions: list[InteractionNode]) -> list[InteractionNode]:
        """Collapse consecutive same-player hits within one swing window."""
        sorted_nodes = sorted(interactions, key=lambda n: n.frame_idx)
        out: list[InteractionNode] = []
        for n in sorted_nodes:
            if n.interaction_type.value != "player_hit":
                out.append(n)
                continue
            if out:
                prev = out[-1]
                if (
                    prev.interaction_type.value == "player_hit"
                    and prev.actor_id == n.actor_id
                    and n.frame_idx - prev.frame_idx < self.hit_gap_frames
                ):
                    if (n.speed_kmh or 0) > (prev.speed_kmh or 0):
                        out[-1] = n
                    continue
            out.append(n)
        return out

    def segment_rallies(self, interactions: list[InteractionNode]) -> list[tuple[int, int]]:
        if not interactions:
            return []
        sorted_nodes = sorted(interactions, key=lambda n: n.frame_idx)
        segments: list[tuple[int, int]] = []
        start = sorted_nodes[0].frame_idx
        prev = start
        for n in sorted_nodes[1:]:
            gap = n.frame_idx - prev
            ground_break = prev != start and any(
                x.interaction_type.value == "ball_ground"
                for x in sorted_nodes
                if prev <= x.frame_idx <= prev + 2
            )
            if gap > self.gap_frames or n.interaction_type.value == "ball_ground":
                segments.append((start, prev))
                start = n.frame_idx
            prev = n.frame_idx
        segments.append((start, prev))
        return segments

    def build(
        self,
        world: WorldModel,
        analytics_rallies: list[RallySegment] | None = None,
    ) -> list[RallyGraph]:
        interactions = self.dedupe_hits(world.all_interactions)
        graphs: list[RallyGraph] = []

        if analytics_rallies:
            for i, r in enumerate(analytics_rallies):
                nodes = [n for n in interactions if r.start_frame <= n.frame_idx <= r.end_frame]
                nodes.sort(key=lambda n: n.frame_idx)
                if nodes:
                    graphs.append(
                        RallyGraph(
                            rally_id=i,
                            start_frame=r.start_frame,
                            end_frame=r.end_frame,
                            nodes=nodes,
                        )
                    )
            if graphs:
                return graphs

        segments = self.segment_rallies(interactions)
        for i, (start, end) in enumerate(segments):
            if end - start < self.min_rally_frames:
                continue
            nodes = [n for n in interactions if start <= n.frame_idx <= end]
            nodes.sort(key=lambda n: n.frame_idx)
            hit_count = sum(1 for n in nodes if n.interaction_type.value == "player_hit")
            if hit_count < self.min_shots:
                continue
            graphs.append(RallyGraph(rally_id=i, start_frame=start, end_frame=end, nodes=nodes))

        return graphs

    @staticmethod
    def chain_label(nodes: list[InteractionNode]) -> str:
        parts: list[str] = []
        last_part = ""
        for n in nodes:
            part = ""
            if n.interaction_type.value == "player_hit" and n.stroke_type:
                pid = n.actor_id or 0
                part = f"P{pid % 100}:{n.stroke_type.value}"
            elif n.interaction_type.value == "ball_wall_glass":
                part = "glass"
            elif n.interaction_type.value == "ball_wall_fence":
                part = "fence"
            elif n.interaction_type.value == "ball_ground":
                part = "ground"
            elif n.interaction_type.value == "ball_net":
                part = "net"
            elif n.interaction_type.value == "point_end":
                part = "point_over"
            if part and part != last_part:
                parts.append(part)
                last_part = part
        return " → ".join(parts) if parts else "empty"
