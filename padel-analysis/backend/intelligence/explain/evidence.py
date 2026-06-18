"""Evidence-backed explainability for coaching recommendations."""

from __future__ import annotations

from typing import Any

from backend.intelligence.shot.understanding import ShotUnderstanding
from backend.intelligence.tactical.insights import TacticalSnapshot


class ExplainabilityEngine:
    def build_recommendations(
        self,
        tactical: TacticalSnapshot,
        shots: list[ShotUnderstanding],
        patterns: dict,
        fps: float,
    ) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []

        if tactical.positioning.optimal_pct < 0.35:
            net_shots = [s for s in shots if s.region.value in ("net_front", "attack_zone")]
            recs.append(
                {
                    "advice": "Increase net approaches when partner covers the lane.",
                    "evidence": f"Only {tactical.positioning.optimal_pct*100:.0f}% optimal positioning across match.",
                    "clips": [{"frame": s.frame_idx, "time_s": round(s.frame_idx / fps, 1)} for s in net_shots[:5]],
                    "confidence": 0.75,
                }
            )

        poor = [s for s in shots if s.decision_quality.value in ("poor", "suboptimal")]
        if poor:
            recs.append(
                {
                    "advice": "Reduce aggressive shots from defensive glass zone.",
                    "evidence": f"{len(poor)} suboptimal decisions detected.",
                    "clips": [{"frame": s.frame_idx, "time_s": round(s.frame_idx / fps, 1)} for s in poor[:5]],
                    "confidence": 0.8,
                }
            )

        for insight in patterns.get("insights", [])[:3]:
            recs.append({"advice": insight, "evidence": "Pattern mining", "clips": [], "confidence": 0.65})

        return recs
