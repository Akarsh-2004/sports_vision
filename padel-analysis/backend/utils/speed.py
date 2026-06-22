"""Ball speed sanity checks — damp Kalman noise before highlights/reports."""

from __future__ import annotations

MAX_REALISTIC_PADEL_SPEED_KMH = 120.0


def clamp_ball_speed_kmh(speed: float, cap: float = MAX_REALISTIC_PADEL_SPEED_KMH) -> float:
    """Hard cap unrealistic tracker spikes (amateur padel rarely exceeds ~80 km/h)."""
    if speed <= 0:
        return 0.0
    return min(float(speed), cap)


def smooth_speed_series(speeds: list[float], window: int = 3) -> list[float]:
    """Median-of-window smoothing for a speed signal."""
    if not speeds:
        return []
    w = max(1, window)
    out: list[float] = []
    for i in range(len(speeds)):
        chunk = speeds[max(0, i - w + 1) : i + 1]
        chunk_sorted = sorted(chunk)
        out.append(chunk_sorted[len(chunk_sorted) // 2])
    return out


def peak_speed_in_range(
    speeds_by_frame: dict[int, float],
    start_frame: int,
    end_frame: int,
) -> float:
    """Peak clamped ball speed within a rally window."""
    if not speeds_by_frame:
        return 0.0
    vals = [
        clamp_ball_speed_kmh(v)
        for f, v in speeds_by_frame.items()
        if start_frame <= f <= end_frame
    ]
    return max(vals) if vals else 0.0
