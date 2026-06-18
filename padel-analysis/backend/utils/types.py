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
    BANDEJA = "bandeja"
    VIBORA = "vibora"
    LOB = "lob"
    DROP_SHOT = "drop_shot"
    CHIQUITA = "chiquita"
    SALIDA = "salida"
    SERVE = "serve"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    WINNER = "winner"
    UNFORCED_ERROR = "unforced_error"
    FORCED_ERROR = "forced_error"
    NET_ERROR = "net_error"
    WALL_WINNER = "wall_winner"
    DOUBLE_BOUNCE = "double_bounce"
    NET_APPROACH = "net_approach"
    LONG_RALLY = "long_rally"
    WALL_EXCHANGE = "wall_exchange"
    SMASH_WINNER = "smash_winner"


class WallHitType(str, Enum):
    GLASS = "glass"
    FENCE = "fence"
    NET = "net"
    GROUND = "ground"
    UNKNOWN = "unknown"


class PlayerSelectionMode(str, Enum):
    SINGLE = "single"
    PAIR = "pair"
    ALL = "all"


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
    team_id: int | None = None


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
class BallContact:
    frame_idx: int
    track_id: int
    court_xy: tuple[float, float]
    ball_speed_kmh: float = 0.0


@dataclass
class WallEvent:
    frame_idx: int
    hit_type: WallHitType
    court_xy: tuple[float, float] | None = None


@dataclass
class RallySegment:
    start_frame: int
    end_frame: int
    rally_length_shots: int = 0
    wall_hits: int = 0
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
    net_zone_pct: float = 0.0
    defensive_zone_pct: float = 0.0
    heatmap: list[list[float]] = field(default_factory=list)


@dataclass
class TeamStats:
    avg_spacing_m: float = 0.0
    formation_stability: float = 0.0
    coverage_overlap_pct: float = 0.0
    rotation_quality: float = 0.0


@dataclass
class TacticalInsights:
    net_dominance_pct: float = 0.0
    wall_usage_pct: float = 0.0
    lob_frequency: float = 0.0
    smash_success_pct: float = 0.0
    attack_frequency: float = 0.0
    risk_score: float = 0.0
    pressure_zones: dict[str, float] = field(default_factory=dict)


@dataclass
class ShotQuality:
    frame_idx: int
    power_kmh: float = 0.0
    placement_score: float = 0.0
    in_court: bool = True
    landing_xy: tuple[float, float] | None = None
    zone: str = "unknown"
    off_wall: bool = False


@dataclass
class PerformanceScores:
    movement: float = 0.0
    consistency: float = 0.0
    aggression: float = 0.0
    net_play: float = 0.0
    wall_defense: float = 0.0
    positioning: float = 0.0
    shot_quality: float = 0.0
    stamina: float = 0.0
    overall: float = 0.0


@dataclass
class MatchStats:
    match_id: str
    target_track_id: int
    selection_mode: str
    total_frames: int
    duration_s: float
    fps: float
    movement: MovementStats = field(default_factory=MovementStats)
    team: TeamStats = field(default_factory=TeamStats)
    tactical: TacticalInsights = field(default_factory=TacticalInsights)
    scores: PerformanceScores = field(default_factory=PerformanceScores)
    rallies: list[RallySegment] = field(default_factory=list)
    events: list[MatchEvent] = field(default_factory=list)
    strokes: list[StrokeEvent] = field(default_factory=list)
    wall_events: list[WallEvent] = field(default_factory=list)
    shot_qualities: list[ShotQuality] = field(default_factory=list)
    stroke_distribution: dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        def _enum(v: Any) -> Any:
            return v.value if hasattr(v, "value") else v

        return {
            "match_id": self.match_id,
            "sport": "padel",
            "target_track_id": self.target_track_id,
            "selection_mode": self.selection_mode,
            "total_frames": self.total_frames,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "movement": {
                "total_distance_m": self.movement.total_distance_m,
                "max_speed_kmh": self.movement.max_speed_kmh,
                "avg_speed_kmh": self.movement.avg_speed_kmh,
                "sprint_count": self.movement.sprint_count,
                "lateral_ratio": self.movement.lateral_ratio,
                "net_zone_pct": self.movement.net_zone_pct,
                "defensive_zone_pct": self.movement.defensive_zone_pct,
            },
            "team": {
                "avg_spacing_m": self.team.avg_spacing_m,
                "formation_stability": self.team.formation_stability,
                "coverage_overlap_pct": self.team.coverage_overlap_pct,
                "rotation_quality": self.team.rotation_quality,
            },
            "tactical": {
                "net_dominance_pct": self.tactical.net_dominance_pct,
                "wall_usage_pct": self.tactical.wall_usage_pct,
                "lob_frequency": self.tactical.lob_frequency,
                "smash_success_pct": self.tactical.smash_success_pct,
                "attack_frequency": self.tactical.attack_frequency,
                "risk_score": self.tactical.risk_score,
                "pressure_zones": self.tactical.pressure_zones,
            },
            "scores": {
                "movement": self.scores.movement,
                "consistency": self.scores.consistency,
                "aggression": self.scores.aggression,
                "net_play": self.scores.net_play,
                "wall_defense": self.scores.wall_defense,
                "positioning": self.scores.positioning,
                "shot_quality": self.scores.shot_quality,
                "stamina": self.scores.stamina,
                "overall": self.scores.overall,
            },
            "rallies": [
                {
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "rally_length_shots": r.rally_length_shots,
                    "wall_hits": r.wall_hits,
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
            "wall_events": [
                {
                    "frame_idx": w.frame_idx,
                    "hit_type": _enum(w.hit_type),
                    "court_xy": w.court_xy,
                }
                for w in self.wall_events
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
