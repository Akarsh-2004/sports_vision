"""
Ball shot frame detection — adapted from abdullahtarek/tennis_analysis.

Uses trajectory inflection (direction change) rather than pose heuristics.
More reliable for rally segmentation when ball tracking is noisy.
https://github.com/abdullahtarek/tennis_analysis

Padel improvements over tennis version:
  - Uses BOTH Y and X direction changes (padel has lateral wall redirections)
  - Confidence-weighted smoothing (ignores low-conf frames for inflection)
  - Adaptive min_change_frames based on fps
  - Speed-based hit detection: sharp speed increase = likely hit
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BallShotDetector:
    """Detect racket/ball contact frames from ball trajectory inflections."""

    def __init__(self, fps: float = 25.0, min_change_frames: int | None = None):
        self.fps = fps
        # Tennis repo uses 25 frames at 24fps (~1s); scale for our fps
        self.min_change_frames = min_change_frames or max(12, int(0.45 * fps))

    def detect_from_trajectory(
        self,
        trajectory: list[tuple[int, float, float, float]],
        min_conf: float = 0.15,
    ) -> list[int]:
        """
        Return frame indices where ball direction changes sharply (hits).

        trajectory: list of (frame_idx, x, y, confidence)

        Uses both Y-axis inflections (classic tennis approach) and X-axis
        inflections (padel wall redirections) then merges and deduplicates.
        """
        if len(trajectory) < 8:
            return []

        # High-confidence detections only for inflection
        hq = [(t[0], t[1], t[2], t[3]) for t in trajectory if t[3] >= min_conf]
        if len(hq) < 8:
            hq = list(trajectory)  # fall back to all

        frames = [t[0] for t in hq]
        xs = [t[1] for t in hq]
        ys = [t[2] for t in hq]
        confs = [t[3] for t in hq]

        df = pd.DataFrame({"frame": frames, "x": xs, "y": ys, "conf": confs})
        df = df.sort_values("frame").reset_index(drop=True)

        # Smoothed position (rolling average)
        df["y_roll"] = df["y"].rolling(window=5, min_periods=1).mean()
        df["x_roll"] = df["x"].rolling(window=5, min_periods=1).mean()
        df["delta_y"] = df["y_roll"].diff()
        df["delta_x"] = df["x_roll"].diff()

        # Magnitude of direction change
        df["speed"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
        df["speed_roll"] = df["speed"].rolling(window=3, min_periods=1).mean()

        # Y-axis hits (classic: ball going up then down / down then up)
        y_hits = self._find_inflections(df, "delta_y")

        # X-axis hits (padel wall redirections — ball changes lateral direction)
        x_hits = self._find_inflections(df, "delta_x", sensitivity=0.7)

        # Speed spike hits (sharp acceleration = racket strike)
        speed_hits = self._find_speed_spikes(df)

        # Merge all hit candidates, map back to frame indices
        all_hits = sorted(set(y_hits) | set(x_hits) | set(speed_hits))

        return self._dedupe(all_hits)

    def _find_inflections(
        self,
        df: pd.DataFrame,
        delta_col: str,
        sensitivity: float = 1.0,
    ) -> list[int]:
        """Find frames where ball reverses direction along given axis."""
        hit_frames: list[int] = []
        window = int(self.min_change_frames * 1.2)
        n = len(df)

        for i in range(1, n - window):
            neg_change = df[delta_col].iloc[i] > 0 and df[delta_col].iloc[i + 1] < 0
            pos_change = df[delta_col].iloc[i] < 0 and df[delta_col].iloc[i + 1] > 0
            if not (neg_change or pos_change):
                continue

            change_count = 0
            for j in range(i + 1, min(i + window + 1, n)):
                if neg_change and df[delta_col].iloc[i] > 0 and df[delta_col].iloc[j] < 0:
                    change_count += 1
                elif pos_change and df[delta_col].iloc[i] < 0 and df[delta_col].iloc[j] > 0:
                    change_count += 1

            threshold = (self.min_change_frames - 1) * sensitivity
            if change_count >= threshold:
                hit_frames.append(int(df["frame"].iloc[i]))

        return hit_frames

    def _find_speed_spikes(self, df: pd.DataFrame) -> list[int]:
        """
        Detect sharp speed increases (ball being hit causes sudden acceleration).
        Only trigger if speed increase is ≥2x rolling average.
        """
        hit_frames: list[int] = []
        n = len(df)
        for i in range(2, n - 1):
            prev_speed = df["speed_roll"].iloc[i - 1]
            curr_speed = df["speed"].iloc[i]
            if prev_speed > 0 and curr_speed > prev_speed * 2.0 and curr_speed > 5.0:
                hit_frames.append(int(df["frame"].iloc[i]))
        return hit_frames

    def assign_hitter(
        self,
        shot_frame: int,
        players_by_frame: dict[int, dict[int, tuple[float, float]]],
        ball_by_frame: dict[int, tuple[float, float]],
    ) -> int | None:
        """Closest player to ball at shot frame (tennis_analysis approach)."""
        if shot_frame not in players_by_frame or shot_frame not in ball_by_frame:
            return None
        ball = ball_by_frame[shot_frame]
        positions = players_by_frame[shot_frame]
        if not positions:
            return None
        return min(
            positions.keys(),
            key=lambda pid: float(np.hypot(positions[pid][0] - ball[0], positions[pid][1] - ball[1])),
        )

    def _dedupe(self, frames: list[int], min_gap: int | None = None) -> list[int]:
        gap = min_gap or max(1, int(0.4 * self.fps))
        if not frames:
            return []
        out = [frames[0]]
        for f in sorted(frames)[1:]:
            if f - out[-1] >= gap:
                out.append(f)
        return out
