"""Semantic court regions — spatial intelligence beyond raw zones."""

from __future__ import annotations

from enum import Enum


class SemanticRegion(str, Enum):
    ATTACK_ZONE = "attack_zone"
    TRANSITION_ZONE = "transition_zone"
    DEFENSE_ZONE = "defense_zone"
    GLASS_DEFENSE = "glass_defense"
    LEFT_ALLEY = "left_alley"
    RIGHT_ALLEY = "right_alley"
    MIDDLE_GAP = "middle_gap"
    SMASH_ZONE = "smash_zone"
    LOB_ZONE = "lob_zone"
    NET_FRONT = "net_front"
    SERVICE_NEAR = "service_near"
    SERVICE_FAR = "service_far"
    OUT = "out"


# Court partitions (10m x 20m, y=0 near baseline)
REGION_BOUNDS: dict[SemanticRegion, tuple[float, float, float, float]] = {
    # x_min, x_max, y_min, y_max
    SemanticRegion.LEFT_ALLEY: (0.0, 2.5, 0.0, 20.0),
    SemanticRegion.RIGHT_ALLEY: (7.5, 10.0, 0.0, 20.0),
    SemanticRegion.MIDDLE_GAP: (3.5, 6.5, 0.0, 20.0),
    SemanticRegion.NET_FRONT: (0.0, 10.0, 8.5, 11.5),
    SemanticRegion.SMASH_ZONE: (2.0, 8.0, 9.0, 12.0),
    SemanticRegion.ATTACK_ZONE: (0.0, 10.0, 7.0, 13.0),
    SemanticRegion.TRANSITION_ZONE: (0.0, 10.0, 4.0, 16.0),
    SemanticRegion.DEFENSE_ZONE: (0.0, 10.0, 0.0, 5.0),
    SemanticRegion.GLASS_DEFENSE: (0.0, 10.0, -3.0, 2.5),
    SemanticRegion.LOB_ZONE: (0.0, 10.0, 0.0, 4.0),
    SemanticRegion.SERVICE_NEAR: (0.0, 10.0, 0.0, 3.0),
    SemanticRegion.SERVICE_FAR: (0.0, 10.0, 17.0, 20.0),
}


def classify_region(x: float, y: float) -> SemanticRegion:
    """Return primary semantic region for a court position."""
    if not (-3 <= x <= 13 and -3 <= y <= 23):
        return SemanticRegion.OUT
    if y < -0.5:
        return SemanticRegion.GLASS_DEFENSE
    if 8.5 <= y <= 11.5 and 2.0 <= x <= 8.0:
        return SemanticRegion.SMASH_ZONE
    if 8.5 <= y <= 11.5:
        return SemanticRegion.NET_FRONT
    if x < 2.5:
        return SemanticRegion.LEFT_ALLEY
    if x > 7.5:
        return SemanticRegion.RIGHT_ALLEY
    if 3.5 <= x <= 6.5:
        return SemanticRegion.MIDDLE_GAP
    if y < 4.0:
        return SemanticRegion.LOB_ZONE if y < 3 else SemanticRegion.DEFENSE_ZONE
    if y > 16.0:
        return SemanticRegion.DEFENSE_ZONE
    if 7.0 <= y <= 13.0:
        return SemanticRegion.ATTACK_ZONE
    return SemanticRegion.TRANSITION_ZONE
