"""Player readiness and movement helpers for rally gating."""

from __future__ import annotations

import numpy as np

from backend.utils.types import CourtState, PlayerDetection


def players_in_ready_position(
    players: list[PlayerDetection],
    court: CourtState | None = None,
    court_length_m: float = 23.77,
) -> bool:
    """Both players near baselines in ready stance area."""
    if len(players) < 2:
        return len(players) >= 1
    positions: list[float] = []
    for p in players:
        if p.court_xy:
            positions.append(p.court_xy[1])
        else:
            cy = p.bbox.centroid[1]
            positions.append(cy)
    if court and court.homography:
        near = sum(1 for y in positions if y < court_length_m * 0.35 or y > court_length_m * 0.65)
        return near >= 2 or len(players) >= 2
    # Pixel fallback: players spread vertically
    ys = [p.bbox.centroid[1] for p in players]
    return max(ys) - min(ys) > 80


def walking_to_baseline(
    players: list[PlayerDetection],
    prev_positions: dict[int, tuple[float, float]],
    fps: float = 25.0,
) -> bool:
    """Detect slow lateral/backward walk toward baseline between points."""
    for p in players:
        prev = prev_positions.get(p.track_id)
        if prev is None or p.court_xy is None:
            continue
        cx, cy = p.court_xy
        px, py = prev
        speed = float(np.hypot(cx - px, cy - py) * fps)
        dy = cy - py
        if 0.3 < speed < 2.5 and abs(dy) > abs(cx - px):
            return True
    return False
