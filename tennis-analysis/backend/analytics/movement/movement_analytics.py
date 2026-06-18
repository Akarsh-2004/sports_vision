from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from backend.utils.types import MovementStats


class MovementAnalytics:
    """Stage 10: distance, speed, sprints, heatmap."""

    def __init__(self, config: dict):
        acfg = config["analytics"]
        self.sprint_threshold = acfg["sprint_threshold_mps"]
        self.sprint_min_frames = int(acfg["sprint_min_duration_s"] * config["pipeline"]["target_fps"])
        self.court_length = config["court"]["court_length_m"]
        self.singles_width = config["court"]["singles_width_m"]
        self.positions: list[tuple[int, float, float]] = []
        self.speeds: list[float] = []
        self.fps = config["pipeline"]["target_fps"]

    def add_position(self, frame_idx: int, court_x: float, court_y: float) -> None:
        self.positions.append((frame_idx, court_x, court_y))

    def compute(self) -> MovementStats:
        if len(self.positions) < 2:
            return MovementStats()

        total_dist = 0.0
        lateral = 0.0
        longitudinal = 0.0
        speeds_mps: list[float] = []
        offensive = 0
        defensive = 0

        for i in range(1, len(self.positions)):
            _, x0, y0 = self.positions[i - 1]
            f1, x1, y1 = self.positions[i]
            dx, dy = x1 - x0, y1 - y0
            dist = float(np.hypot(dx, dy))
            total_dist += dist
            lateral += abs(dx)
            longitudinal += abs(dy)
            dt = max((f1 - self.positions[i - 1][0]) / self.fps, 1e-6)
            speeds_mps.append(dist / dt)
            if y1 < self.court_length * 0.33:
                offensive += 1
            elif y1 > self.court_length * 0.66:
                defensive += 1

        if speeds_mps:
            from scipy.signal import medfilt

            smoothed = medfilt(speeds_mps, kernel_size=5) if len(speeds_mps) >= 5 else speeds_mps
            max_speed = float(np.max(smoothed)) * 3.6
            avg_speed = float(np.mean(smoothed)) * 3.6
        else:
            max_speed = avg_speed = 0.0

        sprint_count = self._count_sprints(speeds_mps)
        heatmap = self._build_heatmap()
        total_frames = max(len(self.positions), 1)
        lat_ratio = lateral / max(lateral + longitudinal, 1e-6)

        return MovementStats(
            total_distance_m=total_dist,
            max_speed_kmh=max_speed,
            avg_speed_kmh=avg_speed,
            sprint_count=sprint_count,
            lateral_ratio=float(lat_ratio),
            offensive_zone_pct=offensive / total_frames,
            defensive_zone_pct=defensive / total_frames,
            heatmap=heatmap,
        )

    def _count_sprints(self, speeds_mps: list[float]) -> int:
        count = 0
        run = 0
        for s in speeds_mps:
            if s >= self.sprint_threshold:
                run += 1
            else:
                if run >= self.sprint_min_frames:
                    count += 1
                run = 0
        if run >= self.sprint_min_frames:
            count += 1
        return count

    def _build_heatmap(self, bins: int = 20) -> list[list[float]]:
        grid = np.zeros((bins, bins), dtype=np.float32)
        for _, x, y in self.positions:
            xi = int(np.clip(x / self.singles_width * (bins - 1), 0, bins - 1))
            yi = int(np.clip(y / self.court_length * (bins - 1), 0, bins - 1))
            grid[yi, xi] += 1
        if grid.sum() > 0:
            grid = gaussian_filter(grid, sigma=0.8)
            grid /= grid.max()
        return grid.tolist()
