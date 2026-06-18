"""Layer 6 — LLM reasoning over structured padel knowledge."""

from __future__ import annotations

import json

import httpx

from backend.utils.logging import get_logger

logger = get_logger(__name__)

COACH_SYSTEM = """You are an elite padel coach analyzing structured match data.
You receive JSON with shot facts, interaction chains, positioning, and tactical metrics.
Write coach-level insights: positioning errors, net control, shot selection quality, partner spacing.
Never invent statistics not present in the JSON. Be specific and actionable."""


class KnowledgeReasoner:
    def __init__(self, config: dict):
        scfg = config.get("summarization", {})
        self.provider = scfg.get("provider", "template")
        self.ollama_url = scfg.get("ollama_url", "http://localhost:11434")
        self.ollama_model = scfg.get("ollama_model", "qwen2.5:7b")

    def reason(self, knowledge_package: dict) -> str:
        if self.provider == "ollama":
            text = self._ollama(knowledge_package)
            if text:
                return text
        return self._template(knowledge_package)

    def _ollama(self, package: dict) -> str | None:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": COACH_SYSTEM},
                {
                    "role": "user",
                    "content": f"Analyze this padel match knowledge graph:\n{json.dumps(package, indent=2)[:12000]}",
                },
            ],
            "stream": False,
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{self.ollama_url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except Exception as exc:
            logger.warning("Knowledge reasoner fallback: %s", exc)
            return None

    def _template(self, package: dict) -> str:
        tactical = package.get("tactical", {})
        pos = tactical.get("positioning", {})
        ctrl = tactical.get("court_control", {})
        notes = package.get("coach_notes", [])
        shots = package.get("shot_facts", [])
        rallies = package.get("rally_narratives", [])

        lines = [
            "# Coach Intelligence Report",
            "",
            "## Positioning",
            f"- Too deep: **{pos.get('too_deep_pct', 0) * 100:.0f}%** of active frames",
            f"- Net optimal: **{pos.get('optimal_pct', 0) * 100:.0f}%**",
            f"- Wrong lane: **{pos.get('wrong_side_pct', 0) * 100:.0f}%**",
            "",
            "## Court Control",
            f"- Net control: **{ctrl.get('net_control_pct', 0) * 100:.0f}%**",
            f"- Backcourt trapped: **{ctrl.get('backcourt_trapped_pct', 0) * 100:.0f}%**",
            "",
            "## Interactions",
            f"- Structured shot facts: **{len(shots)}**",
            f"- Rally chains analyzed: **{len(rallies)}**",
        ]
        if rallies:
            lines.append(f"- Example chain: `{rallies[0].get('interaction_chain', '')}`")
        if notes:
            lines.extend(["", "## Coach Notes"] + [f"- {n}" for n in notes[:8]])
        return "\n".join(lines)
