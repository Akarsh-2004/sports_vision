"""Shot understanding — not just stroke labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.intelligence.court.semantic_regions import SemanticRegion, classify_region
from backend.intelligence.domain import DecisionQuality, ShotIntent
from backend.utils.types import StrokeType


@dataclass
class ShotUnderstanding:
    """Rich shot representation for coaching and LLM reasoning."""

    frame_idx: int
    player_id: int
    stroke: StrokeType
    intent: ShotIntent
    pressure: str  # low | medium | high
    risk: str  # low | medium | high
    expected_outcome: str
    region: SemanticRegion
    position: tuple[float, float]
    speed_kmh: float
    confidence: float
    decision_quality: DecisionQuality = DecisionQuality.NEUTRAL
    decision_note: str = ""
    epv_before: float = 0.5
    epv_after: float = 0.5
    opponent_context: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_idx,
            "player_id": self.player_id,
            "stroke": self.stroke.value,
            "intent": self.intent.value,
            "pressure": self.pressure,
            "risk": self.risk,
            "expected_outcome": self.expected_outcome,
            "region": self.region.value,
            "position": list(self.position),
            "speed_kmh": round(self.speed_kmh, 1),
            "confidence": round(self.confidence, 3),
            "decision_quality": self.decision_quality.value,
            "decision_note": self.decision_note,
            "epv_before": round(self.epv_before, 3),
            "epv_after": round(self.epv_after, 3),
            "opponent_context": self.opponent_context,
        }


def infer_pressure(player_y: float, ball_speed: float, at_net: bool) -> str:
    if at_net and ball_speed > 60:
        return "high"
    if ball_speed > 80 or player_y > 12:
        return "medium"
    return "low"


def infer_risk(stroke: StrokeType, region: SemanticRegion) -> str:
    aggressive = {StrokeType.SMASH, StrokeType.VIBORA, StrokeType.BANDEJA}
    if stroke in aggressive and region in (SemanticRegion.DEFENSE_ZONE, SemanticRegion.GLASS_DEFENSE):
        return "high"
    if stroke in aggressive:
        return "medium"
    return "low"


def infer_expected_outcome(stroke: StrokeType, intent: ShotIntent) -> str:
    if intent.value == "finishing":
        return "force_error_or_winner"
    if stroke == StrokeType.LOB:
        return "force_lob_recovery"
    if stroke == StrokeType.DROP_SHOT:
        return "draw_opponent_forward"
    if stroke == StrokeType.SALIDA:
        return "reset_rally"
    return "maintain_rally"
