"""Coach-centric tactical metrics — Layer 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.intelligence.domain import DecisionQuality, Formation


@dataclass
class PositioningInsight:
    too_deep_pct: float = 0.0
    too_shallow_pct: float = 0.0
    wrong_side_pct: float = 0.0
    optimal_pct: float = 0.0


@dataclass
class CourtControlInsight:
    net_control_pct: float = 0.0
    net_dominance_frames: int = 0
    backcourt_trapped_pct: float = 0.0


@dataclass
class TeamShapeInsight:
    formation: Formation = Formation.UNKNOWN
    avg_partner_spacing_m: float = 0.0
    rotation_quality: float = 0.0
    coverage_gap_events: int = 0


@dataclass
class TacticalSnapshot:
    """Per-frame or aggregated coach metrics."""

    frame_idx: int | None = None
    positioning: PositioningInsight = field(default_factory=PositioningInsight)
    court_control: CourtControlInsight = field(default_factory=CourtControlInsight)
    team_shape: TeamShapeInsight = field(default_factory=TeamShapeInsight)
    pressure_score: float = 0.0
    decision_quality: DecisionQuality = DecisionQuality.NEUTRAL
    coach_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "positioning": {
                "too_deep_pct": self.positioning.too_deep_pct,
                "too_shallow_pct": self.positioning.too_shallow_pct,
                "wrong_side_pct": self.positioning.wrong_side_pct,
                "optimal_pct": self.positioning.optimal_pct,
            },
            "court_control": {
                "net_control_pct": self.court_control.net_control_pct,
                "backcourt_trapped_pct": self.court_control.backcourt_trapped_pct,
            },
            "team_shape": {
                "formation": self.team_shape.formation.value,
                "avg_spacing_m": self.team_shape.avg_partner_spacing_m,
                "rotation_quality": self.team_shape.rotation_quality,
            },
            "pressure_score": self.pressure_score,
            "decision_quality": self.decision_quality.value,
            "coach_notes": self.coach_notes,
        }
