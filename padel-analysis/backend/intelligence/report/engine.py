"""Layer 7 — multi-audience report generation."""

from __future__ import annotations

from typing import Any


class ReportEngine:
    """Coach, player, training, and match reports from knowledge package."""

    def generate_all(self, knowledge: dict, intelligence_report: str) -> dict[str, str]:
        return {
            "coach": intelligence_report,
            "player": self._player_report(knowledge),
            "training": self._training_report(knowledge),
            "match": self._match_summary(knowledge),
        }

    def _player_report(self, k: dict) -> str:
        tactical = k.get("tactical", {})
        pos = tactical.get("positioning", {})
        return f"""# Player Report

## Your Game Today
- Net positioning: **{pos.get('optimal_pct', 0) * 100:.0f}%** optimal frames
- Depth issue: standing too deep **{pos.get('too_deep_pct', 0) * 100:.0f}%** of the time

## Focus for Next Session
1. Move forward after defensive wall shots — recover to transition zone, not glass.
2. At net: prioritize bandeja and volleys over groundstrokes.
3. Partner spacing: target ~3 m lateral distance during defense.
"""

    def _training_report(self, k: dict) -> str:
        shots = k.get("shot_understanding") or k.get("shot_facts", [])
        stroke_counts: dict[str, int] = {}
        for s in shots:
            stroke = s.get("stroke", "unknown")
            stroke_counts[stroke] = stroke_counts.get(stroke, 0) + 1
        top = max(stroke_counts.items(), key=lambda x: x[1], default=("none", 0))
        return f"""# Training Plan

## Pattern Identified
Most frequent shot: **{top[0]}** ({top[1]}x)

## Drills
1. Wall reset → transition → net approach (10 min)
2. Bandeja consistency from net (15 min)
3. Defensive lob depth under pressure (10 min)
"""

    def _match_summary(self, k: dict) -> str:
        ws = k.get("world_summary", {})
        return f"""# Match Summary
- Active frames: {ws.get('active_frames', 0)} / {ws.get('total_frames', 0)}
- State distribution: {ws.get('state_distribution', {})}
- Interactions logged: {ws.get('interaction_count', 0)}
"""
