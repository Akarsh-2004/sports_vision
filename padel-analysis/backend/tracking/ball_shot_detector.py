"""
Ball shot frame detection — adapted from abdullahtarek/tennis_analysis.

Uses trajectory inflection (direction change) rather than pose heuristics.
More reliable for rally segmentation when ball tracking is noisy.
https://github.com/abdullahtarek/tennis_analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BallShotDetector:
    """Detect racket/ball contact frames from ball Y trajectory inflections."""

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
        """
        if len(trajectory) < 8:
            return []

        frames = [t[0] for t in trajectory]
        ys = [t[2] for t in trajectory]
        confs = [t[3] for t in trajectory]

        df = pd.DataFrame({"frame": frames, "y": ys, "conf": confs})
        df = df[df["conf"] >= min_conf]
        if len(df) < 8:
            df = pd.DataFrame({"frame": frames, "y": ys, "conf": confs})

        df = df.sort_values("frame").reset_index(drop=True)
        df["mid_y"] = df["y"]
        df["mid_y_roll"] = df["mid_y"].rolling(window=5, min_periods=1).mean()
        df["delta_y"] = df["mid_y_roll"].diff()

        hit_frames: list[int] = []
        window = int(self.min_change_frames * 1.2)
        n = len(df)

        for i in range(1, n - window):
            neg_change = df["delta_y"].iloc[i] > 0 and df["delta_y"].iloc[i + 1] < 0
            pos_change = df["delta_y"].iloc[i] < 0 and df["delta_y"].iloc[i + 1] > 0
            if not (neg_change or pos_change):
                continue

            change_count = 0
            for j in range(i + 1, min(i + window + 1, n)):
                if neg_change and df["delta_y"].iloc[i] > 0 and df["delta_y"].iloc[j] < 0:
                    change_count += 1
                elif pos_change and df["delta_y"].iloc[i] < 0 and df["delta_y"].iloc[j] > 0:
                    change_count += 1

            if change_count > self.min_change_frames - 1:
                hit_frames.append(int(df["frame"].iloc[i]))

        return self._dedupe(hit_frames)

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
