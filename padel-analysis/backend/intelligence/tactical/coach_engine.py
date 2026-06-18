"""Layer 5 — coach-centric tactical reasoning over the digital twin."""

from __future__ import annotations

from collections import defaultdict

from backend.intelligence.domain import DecisionQuality, Formation
from backend.intelligence.interaction.graph import InteractionGraph, RallyGraph
from backend.intelligence.tactical.insights import (
    CourtControlInsight,
    PositioningInsight,
    TacticalSnapshot,
    TeamShapeInsight,
)
from backend.intelligence.tactical.rules import classify_positioning, evaluate_decision
from backend.intelligence.world.digital_twin import DigitalTwin, WorldFrame
from backend.utils.types import StrokeType


class CoachTacticalEngine:
    """
    Answers coach questions — not CV metrics.

    Positioning, court control, pressure, team shape, decision quality.
    """

    def __init__(self, config: dict, target_id: int):
        self.config = config
        self.target_id = target_id
        self.court_width = config["court"]["court_width_m"]
        self.decisions: list[dict] = []
        self.coach_notes: list[str] = []

    def analyze_match(self, twin: DigitalTwin, graph: InteractionGraph) -> TacticalSnapshot:
        active = twin.active_frames()
        if not active:
            return TacticalSnapshot()

        pos_counts: dict[str, int] = defaultdict(int)
        net_frames = 0
        trapped_frames = 0
        spacing_samples: list[float] = []

        for wf in active:
            player = next((p for p in wf.geometry.players if p.track_id == self.target_id), None)
            if not player:
                continue
            y = player.kinematics.position.y
            x = player.kinematics.position.x
            pos_counts[classify_positioning(y, x, self.court_width)] += 1
            if wf.match_state.value == "net_attack":
                net_frames += 1
            if y < 2.5 and wf.match_state.value in ("rally", "lob_defense"):
                trapped_frames += 1

        target_frames = sum(pos_counts.values()) or 1
        positioning = PositioningInsight(
            too_deep_pct=pos_counts.get("too_deep", 0) / target_frames,
            too_shallow_pct=pos_counts.get("too_shallow", 0) / target_frames,
            wrong_side_pct=pos_counts.get("wrong_lane", 0) / target_frames,
            optimal_pct=(
                pos_counts.get("net_optimal", 0) + pos_counts.get("defensive_optimal", 0)
            )
            / target_frames,
        )

        court_control = CourtControlInsight(
            net_control_pct=net_frames / len(active),
            net_dominance_frames=net_frames,
            backcourt_trapped_pct=trapped_frames / target_frames,
        )

        team_shape = self._team_shape(active)
        self._evaluate_decisions(graph)

        snapshot = TacticalSnapshot(
            positioning=positioning,
            court_control=court_control,
            team_shape=team_shape,
            pressure_score=min(1.0, net_frames / max(len(active), 1) + trapped_frames / target_frames * 0.3),
            coach_notes=self.coach_notes[:20],
        )
        return snapshot

    def _team_shape(self, frames: list[WorldFrame]) -> TeamShapeInsight:
        team_players: dict[int, list] = defaultdict(list)
        for wf in frames:
            for p in wf.geometry.players:
                if p.team_id is not None:
                    team_players[p.team_id].append(p)

        spacings: list[float] = []
        for _tid, players in team_players.items():
            if len(players) >= 2:
                import numpy as np

                p0, p1 = players[-2], players[-1]
                spacings.append(
                    float(
                        np.hypot(
                            p0.kinematics.position.x - p1.kinematics.position.x,
                            p0.kinematics.position.y - p1.kinematics.position.y,
                        )
                    )
                )

        avg_spacing = float(sum(spacings) / len(spacings)) if spacings else 0.0
        formation = Formation.UNKNOWN
        if frames:
            last = frames[-1]
            ys = [p.kinematics.position.y for p in last.geometry.players]
            if ys:
                avg_y = sum(ys) / len(ys)
                if avg_y > 9:
                    formation = Formation.BOTH_AT_NET
                elif avg_y < 5:
                    formation = Formation.BOTH_BACK
                else:
                    formation = Formation.ONE_UP_ONE_BACK

        return TeamShapeInsight(
            formation=formation,
            avg_partner_spacing_m=avg_spacing,
            rotation_quality=max(0, 100 - abs(avg_spacing - 3.0) * 20),
        )

    def _evaluate_decisions(self, graph: InteractionGraph) -> None:
        poor = 0
        excellent = 0
        for node in graph.nodes:
            if node.interaction_type.value != "player_hit" or node.actor_id != self.target_id:
                continue
            if node.stroke_type:
                py = node.position[1] if node.position else None
                quality, note = evaluate_decision(
                    node.stroke_type,
                    None,
                    node.actor_id,
                    node.speed_kmh,
                    player_y=py,
                )
                if note:
                    self.coach_notes.append(note)
                if quality == DecisionQuality.POOR:
                    poor += 1
                elif quality == DecisionQuality.EXCELLENT:
                    excellent += 1
                self.decisions.append(
                    {
                        "frame": node.frame_idx,
                        "stroke": node.stroke_type.value,
                        "intent": node.shot_intent.value,
                        "quality": quality.value,
                    }
                )

        if poor > excellent and poor >= 2:
            self.coach_notes.insert(
                0,
                "Repeated suboptimal shot selection from defensive positions — prioritize wall resets and lobs.",
            )

    def analyze_match_from_shots(
        self, shots: list[ShotUnderstanding], world: WorldModel
    ) -> TacticalSnapshot:
        """Coach analysis from shot understanding + world model."""
        from backend.intelligence.tactical.insights import (
            CourtControlInsight,
            PositioningInsight,
            TacticalSnapshot,
            TeamShapeInsight,
        )
        from backend.intelligence.tactical.rules import classify_positioning

        if not shots:
            return TacticalSnapshot()

        pos_counts: dict[str, int] = {}
        for s in shots:
            if s.player_id != self.target_id:
                continue
            key = classify_positioning(s.position[1], s.position[0])
            pos_counts[key] = pos_counts.get(key, 0) + 1

        total = sum(pos_counts.values()) or 1
        positioning = PositioningInsight(
            too_deep_pct=pos_counts.get("too_deep", 0) / total,
            optimal_pct=(pos_counts.get("net_optimal", 0) + pos_counts.get("defensive_optimal", 0)) / total,
            wrong_side_pct=pos_counts.get("wrong_lane", 0) / total,
        )

        active = world.active_snapshots()
        net_frames = sum(1 for s in active if s.match.state.value == "net_attack")
        court_control = CourtControlInsight(
            net_control_pct=net_frames / max(len(active), 1),
        )

        for s in shots:
            if s.decision_note:
                self.coach_notes.append(s.decision_note)

        return TacticalSnapshot(
            positioning=positioning,
            court_control=court_control,
            team_shape=TeamShapeInsight(),
            coach_notes=self.coach_notes[:20],
        )
