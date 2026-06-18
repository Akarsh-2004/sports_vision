"""Expected Point Value — probability of winning point before/after shot."""

from __future__ import annotations

from backend.intelligence.court.semantic_regions import SemanticRegion
from backend.intelligence.domain import ShotIntent
from backend.utils.types import StrokeType


def estimate_epv(
    player_y: float,
    stroke: StrokeType | None,
    intent: ShotIntent,
    ball_speed_kmh: float,
    at_net: bool,
    opponents_deep: bool,
) -> tuple[float, float]:
    """
    Heuristic EPV model (0–1). Replace with learned model later.

    Returns (epv_before, epv_after).
    """
    base = 0.45
    if at_net:
        base += 0.12
    if player_y < 3:
        base -= 0.08
    if opponents_deep:
        base += 0.06

    epv_before = max(0.05, min(0.95, base))

    delta = 0.0
    if stroke in (StrokeType.SMASH, StrokeType.VIBORA, StrokeType.BANDEJA):
        delta = 0.18 if at_net else -0.05
    elif stroke == StrokeType.LOB:
        delta = 0.05 if player_y < 4 else -0.03
    elif stroke == StrokeType.SALIDA:
        delta = 0.03
    elif intent == ShotIntent.FINISHING:
        delta = 0.15
    elif intent == ShotIntent.RECOVERY:
        delta = 0.02

    if ball_speed_kmh > 90 and at_net:
        delta += 0.08

    epv_after = max(0.05, min(0.98, epv_before + delta))
    return epv_before, epv_after
