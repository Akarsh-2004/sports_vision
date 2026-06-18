"""Layer 2 — court-coordinate entities with kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.intelligence.domain import CourtZone, SideRole
from backend.utils.types import StrokeType


@dataclass
class Vec2:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Kinematics:
  position: Vec2
  speed_mps: float = 0.0
  speed_kmh: float = 0.0
  direction_deg: float = 0.0
  acceleration_mps2: float = 0.0
  confidence: float = 1.0


@dataclass
class PlayerEntity:
    """Player in court coordinates — geometry layer truth."""

    track_id: int
    team_id: int | None
    kinematics: Kinematics
    zone: CourtZone
    side_role: SideRole = SideRole.UNKNOWN
    facing_deg: float = 0.0
    pose_confidence: float = 0.0
  # pixel bbox retained for viz only
    pixel_bbox: tuple[float, float, float, float] | None = None

    @property
    def position(self) -> tuple[float, float]:
        return self.kinematics.position.as_tuple()


@dataclass
class BallEntity:
    """Ball in court coordinates."""

    kinematics: Kinematics
    visible: bool = True
    height_m: float = 0.0  # estimated from trajectory / pose
    spin_rpm: float | None = None  # future
    bounce_type: str | None = None


@dataclass
class CourtGeometry:
    """Static + dynamic court structure in meters."""

    width_m: float = 10.0
    length_m: float = 20.0
    net_y_m: float = 10.0
    service_depth_m: float = 3.0
    wall_offset_m: float = 3.0
    homography_valid: bool = False
    homography_confidence: float = 0.0

    def zone_at(self, x: float, y: float) -> CourtZone:
        if not (0 <= x <= self.width_m and 0 <= y <= self.length_m):
            if y < 0:
                return CourtZone.GLASS_NEAR
            if y > self.length_m:
                return CourtZone.GLASS_FAR
            return CourtZone.OUT
        if abs(y - self.net_y_m) < 2.0:
            return CourtZone.NET
        if y < self.service_depth_m:
            return CourtZone.DEFENSIVE_NEAR
        if y > self.length_m - self.service_depth_m:
            return CourtZone.DEFENSIVE_FAR
        if y < self.service_depth_m + 1.5 or y > self.length_m - self.service_depth_m - 1.5:
            return CourtZone.TRANSITION
        return CourtZone.TRANSITION

    def side_role_for(self, x: float) -> SideRole:
        if x < self.width_m * 0.45:
            return SideRole.LEFT
        if x > self.width_m * 0.55:
            return SideRole.RIGHT
        return SideRole.UNKNOWN


@dataclass
class StrokeObservation:
    """Vision-layer stroke guess — refined by interaction + tactical layers."""

    stroke_type: StrokeType
    confidence: float
    hitter_id: int | None = None


@dataclass
class GeometryFrame:
    """All spatial truth for one timestamp."""

    frame_idx: int
    timestamp_s: float
    court: CourtGeometry
    players: list[PlayerEntity] = field(default_factory=list)
    ball: BallEntity | None = None
    active_play: bool = True
