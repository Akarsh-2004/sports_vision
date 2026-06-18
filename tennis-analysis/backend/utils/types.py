from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CameraAngle(str, Enum):
    BASELINE = "baseline"
    SIDE_ON = "side_on"
    OVERHEAD = "overhead"
    CLOSE_UP = "close_up"
    UNKNOWN = "unknown"


class StrokeType(str, Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    VOLLEY_FH = "volley_forehand"
    VOLLEY_BH = "volley_backhand"
    SMASH = "smash"
    FIRST_SERVE = "first_serve"
    SECOND_SERVE = "second_serve"
    SLICE_FH = "slice_forehand"
    SLICE_BH = "slice_backhand"
    LOB = "lob"
    DROP_SHOT = "drop_shot"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    WINNER = "winner"
    UNFORCED_ERROR = "unforced_error"
    FORCED_ERROR = "forced_error"
    ACE = "ace"
    DOUBLE_FAULT = "double_fault"
    NET_APPROACH = "net_approach"
    LONG_RALLY = "long_rally"
    PASSING_SHOT = "passing_shot"


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class CourtState:
    frame_idx: int
    homography: list[list[float]] | None = None
    confidence: float = 0.0
    zone: str = "unknown"
    lines_detected: int = 0
    valid_for_analytics: bool = False


@dataclass
class FrameMeta:
    frame_idx: int
    timestamp_s: float
    width: int
    height: int
    camera_angle: CameraAngle = CameraAngle.UNKNOWN
    court_present: bool = False
    quality_score: float = 1.0


@dataclass
class PlayerDetection:
    track_id: int
    bbox: BBox
    court_xy: tuple[float, float] | None = None


@dataclass
class BallDetection:
    x: float
    y: float
    confidence: float
    visible: bool = True


@dataclass
class PoseKeypoints:
    frame_idx: int
    track_id: int
    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class StrokeEvent:
    frame_idx: int
    track_id: int
    stroke_type: StrokeType
    confidence: float


@dataclass
class RallySegment:
    start_frame: int
    end_frame: int
    rally_length_shots: int = 0
    outcome: str = "unknown"
    excitement_score: float = 0.0


@dataclass
class MatchEvent:
    frame_idx: int
    event_type: EventType
    player_track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MovementStats:
    total_distance_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    sprint_count: int = 0
    lateral_ratio: float = 0.5
    offensive_zone_pct: float = 0.0
    defensive_zone_pct: float = 0.0
    heatmap: list[list[float]] = field(default_factory=list)


@dataclass
class ShotQuality:
    frame_idx: int
    power_kmh: float = 0.0
    placement_score: float = 0.0
    in_court: bool = True
    landing_xy: tuple[float, float] | None = None
    zone: str = "unknown"


@dataclass
class PerformanceScores:
    serve: float = 0.0
    return_score: float = 0.0
    movement: float = 0.0
    consistency: float = 0.0
    aggression: float = 0.0
    stamina: float = 0.0
    court_coverage: float = 0.0
    overall: float = 0.0


@dataclass
class MatchStats:
    match_id: str
    target_track_id: int
    total_frames: int
    duration_s: float
    fps: float
    movement: MovementStats = field(default_factory=MovementStats)
    scores: PerformanceScores = field(default_factory=PerformanceScores)
    rallies: list[RallySegment] = field(default_factory=list)
    events: list[MatchEvent] = field(default_factory=list)
    strokes: list[StrokeEvent] = field(default_factory=list)
    shot_qualities: list[ShotQuality] = field(default_factory=list)
    stroke_distribution: dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        def _enum(v: Any) -> Any:
            return v.value if hasattr(v, "value") else v

        return {
            "match_id": self.match_id,
            "target_track_id": self.target_track_id,
            "total_frames": self.total_frames,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "movement": {
                "total_distance_m": self.movement.total_distance_m,
                "max_speed_kmh": self.movement.max_speed_kmh,
                "avg_speed_kmh": self.movement.avg_speed_kmh,
                "sprint_count": self.movement.sprint_count,
                "lateral_ratio": self.movement.lateral_ratio,
                "offensive_zone_pct": self.movement.offensive_zone_pct,
                "defensive_zone_pct": self.movement.defensive_zone_pct,
            },
            "scores": {
                "serve": self.scores.serve,
                "return": self.scores.return_score,
                "movement": self.scores.movement,
                "consistency": self.scores.consistency,
                "aggression": self.scores.aggression,
                "stamina": self.scores.stamina,
                "court_coverage": self.scores.court_coverage,
                "overall": self.scores.overall,
            },
            "rallies": [
                {
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "rally_length_shots": r.rally_length_shots,
                    "outcome": r.outcome,
                    "excitement_score": r.excitement_score,
                }
                for r in self.rallies
            ],
            "events": [
                {
                    "frame_idx": e.frame_idx,
                    "event_type": _enum(e.event_type),
                    "player_track_id": e.player_track_id,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
            "strokes": [
                {
                    "frame_idx": s.frame_idx,
                    "track_id": s.track_id,
                    "stroke_type": _enum(s.stroke_type),
                    "confidence": s.confidence,
                }
                for s in self.strokes
            ],
            "stroke_distribution": self.stroke_distribution,
            "summary": self.summary,
        }
