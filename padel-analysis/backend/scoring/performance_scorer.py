"""Phase 15: padel performance scoring."""

from __future__ import annotations

import numpy as np

from backend.utils.types import MatchEvent, MovementStats, PerformanceScores, ShotQuality, StrokeEvent, StrokeType


class PerformanceScorer:
    def score(
        self,
        movement: MovementStats,
        strokes: list[StrokeEvent],
        shots: list[ShotQuality],
        events: list[MatchEvent],
    ) -> PerformanceScores:
        winners = sum(1 for e in events if e.event_type.value in ("winner", "wall_winner", "smash_winner"))
        errors = sum(1 for e in events if "error" in e.event_type.value)
        total_shots = max(len(strokes), 1)
        volleys = sum(1 for s in strokes if "volley" in s.stroke_type.value)
        smashes = sum(1 for s in strokes if s.stroke_type == StrokeType.SMASH)
        avg_speed = np.mean([s.power_kmh for s in shots]) if shots else 0
        speed_norm = min(avg_speed / 160.0, 1.0)
        in_play = sum(1 for s in shots if s.in_court) / max(len(shots), 1)

        movement_score = self._scale(
            0.4 * min(movement.total_distance_m / 1800, 1)
            + 0.3 * min(movement.sprint_count / 25, 1)
            + 0.3 * movement.net_zone_pct
        )
        consistency = self._scale(1 - errors / total_shots)
        aggression = self._scale(
            0.35 * speed_norm + 0.25 * (winners / total_shots) + 0.2 * (smashes / total_shots) + 0.2 * in_play
        )
        net_play = self._scale(movement.net_zone_pct + volleys / total_shots * 0.5)
        wall_defense = self._scale(
            sum(1 for s in strokes if s.stroke_type == StrokeType.SALIDA) / max(total_shots, 1)
        )
        positioning = self._scale(0.5 * movement.net_zone_pct + 0.5 * (1 - movement.defensive_zone_pct))
        shot_quality = self._scale(np.mean([s.placement_score for s in shots] or [0]))
        stamina = self._scale(0.6 + 0.4 * min(movement.total_distance_m / 2000, 1))

        dims = [movement_score, consistency, aggression, net_play, wall_defense, positioning, shot_quality, stamina]
        overall = float(np.mean(dims))

        return PerformanceScores(
            movement=movement_score,
            consistency=consistency,
            aggression=aggression,
            net_play=net_play,
            wall_defense=wall_defense,
            positioning=positioning,
            shot_quality=shot_quality,
            stamina=stamina,
            overall=overall,
        )

    def _scale(self, x: float) -> float:
        return float(np.clip(x * 100, 0, 100))
