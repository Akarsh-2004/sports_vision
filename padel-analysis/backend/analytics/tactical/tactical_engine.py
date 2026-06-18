"""Phase 12: tactical engine — padel positioning and strategy metrics."""

from __future__ import annotations

import numpy as np

from backend.utils.types import (
    MatchEvent,
    MovementStats,
    PlayerDetection,
    RallySegment,
    ShotQuality,
    StrokeEvent,
    StrokeType,
    TacticalInsights,
    TeamStats,
)


class TacticalEngine:
    """Compute net dominance, wall usage, partner spacing, risk score."""

    def __init__(self, config: dict):
        self.court_length = config["court"]["court_length_m"]
        self.court_width = config["court"]["court_width_m"]
        self.ideal_spacing = config["analytics"].get("partner_spacing_ideal_m", 3.0)

    def compute_team_stats(
        self,
        players_history: dict[int, list[tuple[float, float]]],
        target_ids: set[int],
    ) -> TeamStats:
        if len(target_ids) < 2:
            return TeamStats()

        ids = list(target_ids)[:2]
        spacings: list[float] = []
        for i in range(min(len(players_history.get(ids[0], [])), len(players_history.get(ids[1], [])))):
            p0 = players_history[ids[0]][i]
            p1 = players_history[ids[1]][i]
            spacings.append(float(np.hypot(p0[0] - p1[0], p0[1] - p1[1])))

        if not spacings:
            return TeamStats()

        avg_spacing = float(np.mean(spacings))
        stability = float(1.0 - min(np.std(spacings) / self.ideal_spacing, 1.0))
        overlap = sum(1 for s in spacings if s < 2.0) / len(spacings)

        return TeamStats(
            avg_spacing_m=avg_spacing,
            formation_stability=stability * 100,
            coverage_overlap_pct=overlap * 100,
            rotation_quality=stability * 90,
        )

    def compute_tactical(
        self,
        movement: MovementStats,
        strokes: list[StrokeEvent],
        shots: list[ShotQuality],
        rallies: list[RallySegment],
        events: list[MatchEvent],
    ) -> TacticalInsights:
        total_shots = max(len(strokes), 1)
        lobs = sum(1 for s in strokes if s.stroke_type == StrokeType.LOB)
        smashes = [s for s in strokes if s.stroke_type == StrokeType.SMASH]
        smash_wins = sum(1 for e in events if e.event_type.value == "smash_winner")
        wall_events = sum(1 for e in events if e.event_type.value == "wall_exchange")
        winners = sum(1 for e in events if e.event_type.value in ("winner", "wall_winner", "smash_winner"))
        errors = sum(1 for e in events if "error" in e.event_type.value)

        avg_speed = np.mean([s.power_kmh for s in shots]) if shots else 0
        attack_freq = (winners + len(smashes)) / total_shots
        risk = float(np.clip(errors / max(winners, 1), 0, 2))

        return TacticalInsights(
            net_dominance_pct=movement.net_zone_pct * 100,
            wall_usage_pct=(wall_events / max(len(rallies), 1)) * 100,
            lob_frequency=lobs / total_shots,
            smash_success_pct=(smash_wins / max(len(smashes), 1)) * 100,
            attack_frequency=attack_freq,
            risk_score=risk,
            pressure_zones={
                "net": movement.net_zone_pct,
                "defensive": movement.defensive_zone_pct,
                "transitional": max(0, 1 - movement.net_zone_pct - movement.defensive_zone_pct),
            },
        )

    def snapshot_positions(self, players: list[PlayerDetection]) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for p in players:
            if p.court_xy:
                out[p.track_id] = p.court_xy
        return out
