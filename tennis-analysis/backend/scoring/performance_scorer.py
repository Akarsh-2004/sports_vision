from __future__ import annotations

import numpy as np

from backend.utils.types import MatchEvent, MovementStats, PerformanceScores, ShotQuality, StrokeEvent, StrokeType


class PerformanceScorer:
    """Stage 15: multi-dimension performance scoring normalized to 0-100."""

    def score(
        self,
        movement: MovementStats,
        strokes: list[StrokeEvent],
        shots: list[ShotQuality],
        events: list[MatchEvent],
    ) -> PerformanceScores:
        serves = [s for s in strokes if s.stroke_type in (StrokeType.FIRST_SERVE, StrokeType.SECOND_SERVE)]
        returns = [s for s in strokes if s.stroke_type not in (StrokeType.FIRST_SERVE, StrokeType.SECOND_SERVE, StrokeType.UNKNOWN)]
        winners = sum(1 for e in events if e.event_type.value == "winner")
        errors = sum(1 for e in events if "error" in e.event_type.value)
        total_shots = max(len(strokes), 1)

        first_serve_pct = sum(1 for s in serves if s.stroke_type == StrokeType.FIRST_SERVE) / max(len(serves), 1)
        ace_rate = sum(1 for e in events if e.event_type.value == "ace") / max(len(serves), 1)
        avg_speed = np.mean([s.power_kmh for s in shots]) if shots else 0
        speed_norm = min(avg_speed / 180.0, 1.0)

        serve = self._scale(0.4 * first_serve_pct + 0.3 * ace_rate + 0.3 * speed_norm)
        in_play = sum(1 for s in shots if s.in_court) / max(len(shots), 1)
        return_score = self._scale(0.5 * in_play + 0.3 * (winners / total_shots) + 0.2 * np.mean([s.placement_score for s in shots] or [0]))
        movement_score = self._scale(
            0.4 * min(movement.total_distance_m / 2000, 1)
            + 0.3 * min(movement.sprint_count / 30, 1)
            + 0.3 * (1 - abs(movement.lateral_ratio - 0.5))
        )
        consistency = self._scale(1 - errors / total_shots)
        aggression = self._scale(
            0.3 * speed_norm
            + 0.3 * (np.mean([s.placement_score for s in shots]) if shots else 0)
            + 0.2 * min(winners / 20, 1)
            + 0.2 * max(0, 1 - errors / max(winners, 1))
        )
        stamina = self._scale(0.7 + 0.3 * (1 - errors / total_shots))
        coverage = self._scale(min(movement.total_distance_m / 2500, 1) * (0.5 + 0.5 * movement.offensive_zone_pct))

        dims = [serve, return_score, movement_score, consistency, aggression, stamina, coverage]
        overall = float(np.mean(dims))

        return PerformanceScores(
            serve=serve,
            return_score=return_score,
            movement=movement_score,
            consistency=consistency,
            aggression=aggression,
            stamina=stamina,
            court_coverage=coverage,
            overall=overall,
        )

    def _scale(self, x: float) -> float:
        return float(np.clip(x * 100, 0, 100))
