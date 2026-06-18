"""Padel court geometry and world-coordinate landmark definitions."""

from __future__ import annotations

from enum import Enum

import numpy as np

# Playable court: 10 m wide × 20 m long (baseline to baseline).
COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = COURT_LENGTH_M / 2.0  # 10.0
SERVICE_DEPTH_M = 3.0
WALL_OFFSET_M = 3.0  # back glass ~3 m behind baseline (for future wall analytics)


class LandmarkId(str, Enum):
    """Manual / auto calibration landmarks (10–14 points)."""

    NEAR_LEFT = "near_left"
    NEAR_RIGHT = "near_right"
    FAR_LEFT = "far_left"
    FAR_RIGHT = "far_right"
    NEAR_SERVICE_LEFT = "near_service_left"
    NEAR_SERVICE_RIGHT = "near_service_right"
    FAR_SERVICE_LEFT = "far_service_left"
    FAR_SERVICE_RIGHT = "far_service_right"
    NET_LEFT = "net_left"
    NET_RIGHT = "net_right"
    CENTER_NEAR = "center_near"
    CENTER_FAR = "center_far"
    NET_CENTER = "net_center"
    CENTER_LINE_NEAR = "center_line_near"
    CENTER_LINE_FAR = "center_line_far"


# Outer corners used for primary homography (same order as tennis: near-L, near-R, far-L, far-R).
CANONICAL_COURT = np.float32(
    [
        [0, 0],
        [COURT_WIDTH_M, 0],
        [0, COURT_LENGTH_M],
        [COURT_WIDTH_M, COURT_LENGTH_M],
    ]
)

# Full landmark set in court meters (x along width, y along length from near baseline).
COURT_LANDMARKS: dict[LandmarkId, tuple[float, float]] = {
    LandmarkId.NEAR_LEFT: (0.0, 0.0),
    LandmarkId.NEAR_RIGHT: (COURT_WIDTH_M, 0.0),
    LandmarkId.FAR_LEFT: (0.0, COURT_LENGTH_M),
    LandmarkId.FAR_RIGHT: (COURT_WIDTH_M, COURT_LENGTH_M),
    LandmarkId.NEAR_SERVICE_LEFT: (0.0, SERVICE_DEPTH_M),
    LandmarkId.NEAR_SERVICE_RIGHT: (COURT_WIDTH_M, SERVICE_DEPTH_M),
    LandmarkId.FAR_SERVICE_LEFT: (0.0, COURT_LENGTH_M - SERVICE_DEPTH_M),
    LandmarkId.FAR_SERVICE_RIGHT: (COURT_WIDTH_M, COURT_LENGTH_M - SERVICE_DEPTH_M),
    LandmarkId.NET_LEFT: (0.0, NET_Y_M),
    LandmarkId.NET_RIGHT: (COURT_WIDTH_M, NET_Y_M),
    LandmarkId.NET_CENTER: (COURT_WIDTH_M / 2, NET_Y_M),
    LandmarkId.CENTER_NEAR: (COURT_WIDTH_M / 2, 0.0),
    LandmarkId.CENTER_FAR: (COURT_WIDTH_M / 2, COURT_LENGTH_M),
    LandmarkId.CENTER_LINE_NEAR: (COURT_WIDTH_M / 2, 0.0),
    LandmarkId.CENTER_LINE_FAR: (COURT_WIDTH_M / 2, COURT_LENGTH_M),
}

# Minimum landmarks for interactive calibration UI (corners + net posts + service intersections).
CALIBRATION_SEQUENCE: list[LandmarkId] = [
    LandmarkId.NEAR_LEFT,
    LandmarkId.NEAR_RIGHT,
    LandmarkId.FAR_LEFT,
    LandmarkId.FAR_RIGHT,
    LandmarkId.NEAR_SERVICE_LEFT,
    LandmarkId.NEAR_SERVICE_RIGHT,
    LandmarkId.FAR_SERVICE_LEFT,
    LandmarkId.FAR_SERVICE_RIGHT,
    LandmarkId.NET_LEFT,
    LandmarkId.NET_RIGHT,
    LandmarkId.NET_CENTER,
    LandmarkId.CENTER_LINE_NEAR,
    LandmarkId.CENTER_LINE_FAR,
]

# Homography refinement uses corners only; extra landmarks validate fit.
HOMOGRAPHY_CORNER_IDS = (
    LandmarkId.NEAR_LEFT,
    LandmarkId.NEAR_RIGHT,
    LandmarkId.FAR_LEFT,
    LandmarkId.FAR_RIGHT,
)
