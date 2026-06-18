from __future__ import annotations

import json

import httpx

from backend.utils.logging import get_logger
from backend.utils.types import MatchStats

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert tennis coach and match analyst. "
    "Write a factual, concise performance report based only on the provided statistics."
)


class ReportGenerator:
    """Stage 16: structured JSON → NL report (Ollama or template fallback)."""

    def __init__(self, config: dict):
        scfg = config["summarization"]
        self.provider = scfg.get("provider", "template")
        self.ollama_url = scfg.get("ollama_url", "http://localhost:11434")
        self.ollama_model = scfg.get("ollama_model", "llama3.1:8b")
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
                        "Generate a ~300 word tennis match analysis report covering strengths, "
                        "weaknesses, tactical patterns, and recommendations.\n\n"
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
        stroke_top = max(stats.stroke_distribution.items(), key=lambda x: x[1], default=("N/A", 0))
        winners = sum(1 for e in stats.events if e.event_type.value == "winner")
        errors = sum(1 for e in stats.events if "error" in e.event_type.value)

        return f"""# Match Analysis Report

## Overview
Analysis of match `{stats.match_id}` covering {stats.duration_s / 60:.1f} minutes at {stats.fps:.0f} FPS.
Overall performance score: **{s.overall:.0f}/100**.

## Movement & Fitness
- Total distance covered: **{m.total_distance_m:.0f} m**
- Peak speed: **{m.max_speed_kmh:.1f} km/h** | Average: **{m.avg_speed_kmh:.1f} km/h**
- Sprints detected: **{m.sprint_count}**
- Time in offensive zone: **{m.offensive_zone_pct * 100:.0f}%**

## Stroke Patterns
- Most frequent stroke: **{stroke_top[0]}** ({stroke_top[1]} instances)
- Winners: **{winners}** | Errors: **{errors}**

## Performance Dimensions
| Dimension | Score |
|-----------|-------|
| Serve | {s.serve:.0f} |
| Return | {s.return_score:.0f} |
| Movement | {s.movement:.0f} |
| Consistency | {s.consistency:.0f} |
| Aggression | {s.aggression:.0f} |
| Stamina | {s.stamina:.0f} |
| Court Coverage | {s.court_coverage:.0f} |

## Key Strengths
1. Movement score of {s.movement:.0f} indicates solid court coverage.
2. Aggression index at {s.aggression:.0f} reflects active point construction.
3. {len(stats.rallies)} rallies analyzed with structured event detection.

## Areas to Improve
1. Consistency score ({s.consistency:.0f}) — reduce unforced errors in extended rallies.
2. Return game ({s.return_score:.0f}) — deepen returns and target weaker zones.
3. Serve ({s.serve:.0f}) — increase first-serve percentage and placement variety.

## Tactical Recommendations
- Vary shot depth to disrupt opponent rhythm; current lateral ratio is {m.lateral_ratio:.2f}.
- Use the {stroke_top[0]} as a setup shot, then attack to open court.
- Focus recovery speed after wide balls — sprint count suggests room for efficiency gains.
"""
