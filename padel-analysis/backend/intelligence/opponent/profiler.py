"""Opponent behavior modeling."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.intelligence.shot.understanding import ShotUnderstanding


class OpponentProfiler:
    def profile(
        self,
        shots: list[ShotUnderstanding],
        target_id: int,
        all_player_ids: set[int],
    ) -> dict[int, dict[str, Any]]:
        opponents = all_player_ids - {target_id}
        profiles: dict[int, dict[str, Any]] = {}

        for opp_id in opponents:
            opp_shots = [s for s in shots if s.player_id == opp_id]
            if not opp_shots:
                profiles[opp_id] = {"note": "insufficient data"}
                continue

            stroke_dist: dict[str, int] = defaultdict(int)
            for s in opp_shots:
                stroke_dist[s.stroke.value] += 1

            total = len(opp_shots)
            lob_pct = stroke_dist.get("lob", 0) / total
            backhand_pct = stroke_dist.get("backhand", 0) / total
            glass_usage = sum(1 for s in opp_shots if s.region.value == "glass_defense") / total

            weaknesses = []
            if backhand_pct > 0.45:
                weaknesses.append("Heavy backhand reliance — attack backhand side")
            if lob_pct > 0.25:
                weaknesses.append("Prefers lobs under pressure — anticipate and attack")
            if glass_usage < 0.05:
                weaknesses.append("Rarely uses glass — may struggle with wall play")

            profiles[opp_id] = {
                "stroke_distribution": dict(stroke_dist),
                "lob_frequency": round(lob_pct, 3),
                "backhand_ratio": round(backhand_pct, 3),
                "glass_usage": round(glass_usage, 3),
                "tactical_recommendations": weaknesses,
            }
        return profiles
