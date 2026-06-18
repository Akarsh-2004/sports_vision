"""Padel domain model — coach vocabulary, not CV labels."""

from __future__ import annotations

from enum import Enum


class MatchState(str, Enum):
    """Layer 3 — finite-state machine for what is happening in the match."""

    IDLE = "idle"
    SERVE = "serve"
    RETURN = "return"
    RALLY = "rally"
    LOB_DEFENSE = "lob_defense"
    NET_ATTACK = "net_attack"
    WALL_EXCHANGE = "wall_exchange"
    RESET = "reset"
    POINT_OVER = "point_over"
    DEAD_TIME = "dead_time"


class CourtZone(str, Enum):
    """Padel-specific spatial zones (geometry layer)."""

    NET = "net"
    TRANSITION = "transition"
    DEFENSIVE_NEAR = "defensive_near"
    DEFENSIVE_FAR = "defensive_far"
    SERVICE_NEAR = "service_near"
    SERVICE_FAR = "service_far"
    OUT = "out"
    GLASS_NEAR = "glass_near"
    GLASS_FAR = "glass_far"


class SideRole(str, Enum):
    """Doubles positioning — left (drive) vs right (finisher) side."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class ShotIntent(str, Enum):
    """What the player was trying to do — not just stroke type."""

    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    NEUTRAL = "neutral"
    RECOVERY = "recovery"
    SETUP = "setup"
    FINISHING = "finishing"
    UNKNOWN = "unknown"


class InteractionType(str, Enum):
    """Layer 4 — interaction graph node types."""

    PLAYER_HIT = "player_hit"
    BALL_WALL_GLASS = "ball_wall_glass"
    BALL_WALL_FENCE = "ball_wall_fence"
    BALL_GROUND = "ball_ground"
    BALL_NET = "ball_net"
    BALL_DOUBLE_BOUNCE = "ball_double_bounce"
    POINT_END = "point_end"
    PARTNER_ROTATION = "partner_rotation"


class Formation(str, Enum):
    """Common padel doubles formations."""

    BOTH_BACK = "both_back"
    ONE_UP_ONE_BACK = "one_up_one_back"
    BOTH_AT_NET = "both_at_net"
    UNKNOWN = "unknown"


class DecisionQuality(str, Enum):
    """Coach judgment on shot choice given context."""

    EXCELLENT = "excellent"
    GOOD = "good"
    NEUTRAL = "neutral"
    SUBOPTIMAL = "suboptimal"
    POOR = "poor"
