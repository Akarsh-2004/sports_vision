from __future__ import annotations

import numpy as np

from backend.utils.types import PoseKeypoints, StrokeEvent, StrokeType


class StrokeClassifier:
    """Stage 8: pose-based rule classifier (MVP)."""

    WRIST_KEYS = ("left_wrist", "right_wrist", "LEFT_WRIST", "RIGHT_WRIST")
    SHOULDER_KEYS = ("left_shoulder", "right_shoulder", "LEFT_SHOULDER", "RIGHT_SHOULDER")

    def classify(
        self,
        frame_idx: int,
        track_id: int,
        pose: PoseKeypoints | None,
        ball_speed: float = 0.0,
        at_baseline: bool = False,
    ) -> StrokeEvent:
        if pose is None or not pose.keypoints:
            return StrokeEvent(frame_idx, track_id, StrokeType.UNKNOWN, 0.3)

        lw = self._get_kp(pose, "left_wrist", "LEFT_WRIST")
        rw = self._get_kp(pose, "right_wrist", "RIGHT_WRIST")
        ls = self._get_kp(pose, "left_shoulder", "LEFT_SHOULDER")
        rs = self._get_kp(pose, "right_shoulder", "RIGHT_SHOULDER")

        if lw is None or rw is None or ls is None or rs is None:
            return StrokeEvent(frame_idx, track_id, StrokeType.UNKNOWN, 0.3)

        shoulder_mid_x = (ls[0] + rs[0]) / 2
        active_wrist = lw if abs(lw[0] - shoulder_mid_x) > abs(rw[0] - shoulder_mid_x) else rw
        is_forehand = active_wrist[0] > shoulder_mid_x

        if at_baseline and ball_speed > 80:
            stroke = StrokeType.FIRST_SERVE if ball_speed > 120 else StrokeType.SECOND_SERVE
            conf = 0.7
        elif active_wrist[1] < min(ls[1], rs[1]) - 30:
            stroke = StrokeType.SMASH
            conf = 0.65
        elif abs(active_wrist[1] - ls[1]) < 40:
            stroke = StrokeType.VOLLEY_FH if is_forehand else StrokeType.VOLLEY_BH
            conf = 0.6
        else:
            stroke = StrokeType.FOREHAND if is_forehand else StrokeType.BACKHAND
            conf = 0.75

        return StrokeEvent(frame_idx, track_id, stroke, conf)

    def _get_kp(self, pose: PoseKeypoints, *names: str) -> tuple[float, float, float] | None:
        for n in names:
            if n in pose.keypoints:
                return pose.keypoints[n]
        return None

    def distribution(self, strokes: list[StrokeEvent]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for s in strokes:
            key = s.stroke_type.value
            dist[key] = dist.get(key, 0) + 1
        return dist
