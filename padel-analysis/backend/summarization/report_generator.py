from __future__ import annotations

import json

import httpx

from backend.utils.logging import get_logger
from backend.utils.types import MatchStats

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert padel coach and match analyst. "
    "Write factual, coach-level insights about positioning, wall play, net dominance, and partner spacing."
)


class ReportGenerator:
    def __init__(self, config: dict):
        scfg = config["summarization"]
        self.provider = scfg.get("provider", "template")
        self.ollama_url = scfg.get("ollama_url", "http://localhost:11434")
        self.ollama_model = scfg.get("ollama_model", "qwen2.5:7b")
        self.temperature = scfg.get("temperature", 0.3)

    def generate(self, stats: MatchStats) -> str:
        if self.provider == "ollama":
            report = self._generate_ollama(stats)
            if report:
                return report
        return self._generate_template(stats)

    def _generate_ollama(self, stats: MatchStats) -> str | None:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Generate a ~350 word padel match analysis with tactical insights, "
                        "wall usage, net play, and recommendations.\n\n"
                        f"Stats JSON:\n{json.dumps(stats.to_dict(), indent=2)}"
                    ),
                },
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{self.ollama_url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except Exception as exc:
            logger.warning("Ollama unavailable (%s); using template", exc)
            return None

    def _generate_template(self, stats: MatchStats) -> str:
        m = stats.movement
        s = stats.scores
        t = stats.tactical
        team = stats.team
        stroke_top = max(stats.stroke_distribution.items(), key=lambda x: x[1], default=("N/A", 0))
        winners = sum(1 for e in stats.events if e.event_type.value in ("winner", "wall_winner", "smash_winner"))
        errors = sum(1 for e in stats.events if "error" in e.event_type.value)

        return f"""# Padel Match Analysis Report

## Overview
Match `{stats.match_id}` — {stats.duration_s / 60:.1f} minutes analyzed ({stats.selection_mode} player mode).
Overall performance: **{s.overall:.0f}/100**.

## Tactical Summary
The player spent **{t.net_dominance_pct:.0f}%** of tracked time in the net zone, indicating an
{"aggressive net-oriented" if t.net_dominance_pct > 40 else "balanced transitional"} strategy.
Wall exchanges featured in **{t.wall_usage_pct:.0f}%** of rallies. Smash success rate: **{t.smash_success_pct:.0f}%**.
Attack frequency: **{t.attack_frequency * 100:.0f}%** | Risk score: **{t.risk_score:.2f}**.

## Movement
- Distance covered: **{m.total_distance_m:.0f} m**
- Peak / avg speed: **{m.max_speed_kmh:.1f} / {m.avg_speed_kmh:.1f} km/h**
- Sprints: **{m.sprint_count}**
- Net zone: **{m.net_zone_pct * 100:.0f}%** | Defensive zones: **{m.defensive_zone_pct * 100:.0f}%**

## Team Dynamics
- Partner spacing: **{team.avg_spacing_m:.1f} m** (ideal ~3 m)
- Formation stability: **{team.formation_stability:.0f}/100**
- Coverage overlap: **{team.coverage_overlap_pct:.0f}%**

## Stroke Profile
Most frequent stroke: **{stroke_top[0]}** ({stroke_top[1]} instances).
Winners: **{winners}** | Errors: **{errors}**

## Scores
| Dimension | Score |
|-----------|-------|
| Movement | {s.movement:.0f} |
| Net play | {s.net_play:.0f} |
| Wall defense | {s.wall_defense:.0f} |
| Positioning | {s.positioning:.0f} |
| Consistency | {s.consistency:.0f} |
| Aggression | {s.aggression:.0f} |

## Recommendations
1. {"Maintain net pressure but improve recovery after deep lobs." if t.net_dominance_pct > 45 else "Increase net approaches when partner covers the lane."}
2. {"Reduce unforced errors on wall retrievals." if errors > winners else "Convert more attacking positions into winners."}
3. {"Tighten partner spacing during defensive transitions." if team.avg_spacing_m > 4 else "Keep current formation discipline."}
"""
