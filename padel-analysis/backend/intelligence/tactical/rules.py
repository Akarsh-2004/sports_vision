"""Padel coaching rules — domain knowledge encoded as reasoning."""

from __future__ import annotations

from backend.intelligence.domain import CourtZone, DecisionQuality, ShotIntent
from backend.intelligence.geometry.entities import GeometryFrame
from backend.utils.types import StrokeType

# Ideal depth bands (m from own baseline, near side y≈0)
IDEAL_DEFENSIVE_Y = (2.5, 5.0)
IDEAL_TRANSITION_Y = (5.0, 8.0)
IDEAL_NET_Y = (8.5, 11.5)


def infer_shot_intent(stroke: StrokeType, geo: GeometryFrame, hitter_id: int) -> ShotIntent:
    player = next((p for p in geo.players if p.track_id == hitter_id), None)
    if not player:
        return ShotIntent.UNKNOWN

    y = player.kinematics.position.y
    aggressive_strokes = {StrokeType.SMASH, StrokeType.VIBORA, StrokeType.BANDEJA}
    defensive_strokes = {StrokeType.LOB, StrokeType.SALIDA, StrokeType.DROP_SHOT}

    if stroke in aggressive_strokes:
        return ShotIntent.FINISHING if player.zone == CourtZone.NET else ShotIntent.AGGRESSIVE
    if stroke in defensive_strokes:
        return ShotIntent.RECOVERY if y < 4 else ShotIntent.DEFENSIVE
    if stroke == StrokeType.CHIQUITA:
        return ShotIntent.SETUP
    return ShotIntent.NEUTRAL


def evaluate_decision(
    stroke: StrokeType,
    geo: GeometryFrame | None,
    hitter_id: int,
    ball_speed_kmh: float,
    player_y: float | None = None,
    player_zone: CourtZone | None = None,
) -> tuple[DecisionQuality, str]:
    """
    Coach judgment: was this the right shot from this position?
    """
    y = player_y
    zone = player_zone
    if geo:
        player = next((p for p in geo.players if p.track_id == hitter_id), None)
        if player:
            y = player.kinematics.position.y
            zone = player.zone

    if y is None:
        return DecisionQuality.NEUTRAL, ""

    if stroke == StrokeType.SMASH and zone not in (CourtZone.NET, CourtZone.TRANSITION):
        return DecisionQuality.POOR, "Smash attempted from too deep — low percentage shot."

    if stroke in (StrokeType.FOREHAND, StrokeType.BACKHAND) and y < 2.0 and ball_speed_kmh > 70:
        return (
            DecisionQuality.SUBOPTIMAL,
            "Aggressive groundstroke from defensive glass zone — consider lob or wall reset.",
        )

    if stroke == StrokeType.LOB and zone == CourtZone.NET:
        return DecisionQuality.SUBOPTIMAL, "Lob from net position — unusual choice; may indicate panic."

    if stroke == StrokeType.BANDEJA and zone == CourtZone.NET:
        return DecisionQuality.EXCELLENT, "Bandeja at net — textbook attacking position."

    if stroke == StrokeType.SALIDA and zone in (CourtZone.DEFENSIVE_NEAR, CourtZone.GLASS_NEAR):
        return DecisionQuality.GOOD, "Salida de pared from back wall — correct defensive response."

    if stroke == StrokeType.VOLLEY_FH and zone == CourtZone.NET:
        return DecisionQuality.GOOD, "Net volley with court position advantage."

    return DecisionQuality.NEUTRAL, ""


def infer_stroke_context(stroke: StrokeType, prev_stroke: StrokeType | None) -> str:
    if prev_stroke == StrokeType.LOB and stroke == StrokeType.SMASH:
        return "lob_defense_punished"
    if prev_stroke and "wall" in prev_stroke.value:
        return "after_wall"
    return "rally"


def classify_positioning(y: float, side_x: float, court_width: float = 10.0) -> str:
    if y < IDEAL_DEFENSIVE_Y[0]:
        return "too_deep"
    if y > IDEAL_NET_Y[1]:
        return "too_shallow"
    if side_x < 1.0 or side_x > court_width - 1.0:
        return "wrong_lane"
    if IDEAL_NET_Y[0] <= y <= IDEAL_NET_Y[1]:
        return "net_optimal"
    if IDEAL_DEFENSIVE_Y[0] <= y <= IDEAL_DEFENSIVE_Y[1]:
        return "defensive_optimal"
    return "transition"
