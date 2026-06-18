"""Layer 6 — structured knowledge for LLM reasoning."""

from __future__ import annotations

from typing import Any

from backend.intelligence.shot.understanding import ShotUnderstanding
from backend.intelligence.tactical.insights import TacticalSnapshot
from backend.intelligence.world.world_model import WorldModel


class KnowledgeEngine:
    """Convert world model + interactions → thousands of structured facts for LLM."""

    def build_shot_facts(
        self,
        twin: DigitalTwin,
        graph: InteractionGraph,
        target_id: int,
    ) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for node in graph.nodes:
            if node.interaction_type.value != "player_hit" or node.actor_id != target_id:
                continue
            wf = next((f for f in twin.frames if f.frame_idx == node.frame_idx), None)
            opponent_positions = []
            if wf:
                opponent_positions = [
                    {"id": p.track_id, "position": p.position, "zone": p.zone.value}
                    for p in wf.geometry.players
                    if p.track_id != target_id
                ]
            player_zone = "unknown"
            player_pos = node.position
            if wf:
                tp = next((p for p in wf.geometry.players if p.track_id == target_id), None)
                if tp:
                    player_zone = tp.zone.value
                    player_pos = tp.position

            facts.append(
                {
                    "shot": node.stroke_type.value if node.stroke_type else "unknown",
                    "speed_kmh": node.speed_kmh,
                    "landing": node.position,
                    "player_position": player_pos,
                    "player_zone": player_zone,
                    "opponent_positions": opponent_positions,
                    "intent": node.shot_intent.value,
                    "outcome": node.outcome,
                    "frame": node.frame_idx,
                    "match_state": wf.match_state.value if wf else "unknown",
                }
            )
        return facts

    def build_rally_narratives(self, rallies: list[RallyGraph]) -> list[dict[str, Any]]:
        return [
            {
                "rally_id": r.rally_id,
                "duration_frames": r.end_frame - r.start_frame,
                "interaction_chain": r.chain_summary(),
                "shot_count": sum(1 for n in r.nodes if n.stroke_type),
            }
            for r in rallies
        ]

    def package_from_shots(
        self,
        world: WorldModel,
        shots: list[ShotUnderstanding],
        tactical: TacticalSnapshot,
        target_id: int,
        match_meta: dict,
    ) -> dict[str, Any]:
        shot_facts = [s.to_dict() for s in shots if s.player_id == target_id]
        return {
            "match": match_meta,
            "world_summary": world.summary(),
            "shot_facts": shot_facts,
            "shot_understanding": shot_facts,
            "rally_narratives": match_meta.get("rally_graphs", []),
            "tactical": tactical.to_dict(),
            "coach_notes": tactical.coach_notes,
            "patterns": match_meta.get("patterns", {}),
            "opponents": match_meta.get("opponents", {}),
            "self_evaluation": match_meta.get("self_evaluation", {}),
        }
