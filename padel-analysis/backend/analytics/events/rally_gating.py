"""Rally gating for 4-player padel."""

from __future__ import annotations

import numpy as np

from backend.utils.types import CourtState, PlayerDetection


def players_in_ready_position(
    players: list[PlayerDetection],
    court: CourtState | None = None,
    court_length_m: float = 20.0,
    min_players: int = 2,
) -> bool:
    """At least two players spread across court halves (doubles)."""
    if len(players) < min_players:
        return len(players) >= 1

    positions_y: list[float] = []
    for p in players:
        if p.court_xy:
            positions_y.append(p.court_xy[1])
        else:
            positions_y.append(p.bbox.centroid[1])

    if court and court.homography:
        spread = max(positions_y) - min(positions_y)
        return spread > court_length_m * 0.25 or len(players) >= 3

    ys = [p.bbox.centroid[1] for p in players]
    return max(ys) - min(ys) > 60


def walking_to_baseline(
    players: list[PlayerDetection],
    prev_positions: dict[int, tuple[float, float]],
    fps: float = 25.0,
) -> bool:
    for p in players:
        prev = prev_positions.get(p.track_id)
        if prev is None or p.court_xy is None:
            continue
        cx, cy = p.court_xy
        px, py = prev
        speed = float(np.hypot(cx - px, cy - py) * fps)
        if 0.25 < speed < 2.8:
            return True
    return False


def detect_baseline_reset(
    prev_players: dict[int, tuple[float, float]],
    curr_players: dict[int, tuple[float, float]],
    court_length_m: float,
    fps: float = 25.0,
) -> bool:
    """
    After a rally, players retreat toward their baselines (both halves moving apart).
    """
    if len(prev_players) < 2 or len(curr_players) < 2:
        return False

    net_y = court_length_m / 2.0
    retreat_top = 0
    retreat_bottom = 0
    count_top = 0
    count_bottom = 0

    for pid, (cx, cy) in curr_players.items():
        prev = prev_players.get(pid)
        if not prev:
            continue
        dy = (cy - prev[1]) * fps
        if cy < net_y:
            count_top += 1
            if dy < -0.2:
                retreat_top += 1
        else:
            count_bottom += 1
            if dy > 0.2:
                retreat_bottom += 1

    if count_top == 0 or count_bottom == 0:
        return False
    return retreat_top >= max(1, count_top // 2) and retreat_bottom >= max(1, count_bottom // 2)
