"""Phase 8: padel stroke recognition (rule-based MVP; VideoMAE hook for future)."""

from __future__ import annotations

from backend.utils.types import PoseKeypoints, StrokeEvent, StrokeType


class StrokeClassifier:
    """Pose + ball-speed rules for padel strokes."""

    def classify(
        self,
        frame_idx: int,
        track_id: int,
        pose: PoseKeypoints | None,
        ball_speed: float = 0.0,
        at_net: bool = False,
        at_back_wall: bool = False,
    ) -> StrokeEvent:
        if pose is None or not pose.keypoints:
            return StrokeEvent(frame_idx, track_id, StrokeType.UNKNOWN, 0.3)

        lw = self._get_kp(pose, "left_wrist", "LEFT_WRIST")
        rw = self._get_kp(pose, "right_wrist", "RIGHT_WRIST")
        ls = self._get_kp(pose, "left_shoulder", "LEFT_SHOULDER")
        rs = self._get_kp(pose, "right_shoulder", "RIGHT_SHOULDER")
        if not all([lw, rw, ls, rs]):
            return StrokeEvent(frame_idx, track_id, StrokeType.UNKNOWN, 0.3)

        shoulder_mid_x = (ls[0] + rs[0]) / 2
        active = lw if abs(lw[0] - shoulder_mid_x) > abs(rw[0] - shoulder_mid_x) else rw
        is_forehand = active[0] > shoulder_mid_x

        if at_back_wall and ball_speed > 60:
            stroke = StrokeType.SALIDA
            conf = 0.6
        elif active[1] < min(ls[1], rs[1]) - 25 and ball_speed > 90:
            stroke = StrokeType.SMASH
            conf = 0.7
        elif at_net and ball_speed < 50 and active[1] > ls[1]:
            stroke = StrokeType.CHIQUITA
            conf = 0.55
        elif at_net and 40 < ball_speed < 80 and active[1] < ls[1] - 10:
            stroke = StrokeType.BANDEJA
            conf = 0.65
        elif at_net and ball_speed > 70 and not is_forehand:
            stroke = StrokeType.VIBORA
            conf = 0.6
        elif at_net and abs(active[1] - ls[1]) < 35:
            stroke = StrokeType.VOLLEY_FH if is_forehand else StrokeType.VOLLEY_BH
            conf = 0.65
        elif ball_speed < 35 and active[1] > ls[1]:
            stroke = StrokeType.DROP_SHOT
            conf = 0.5
        elif ball_speed > 50 and active[1] < ls[1] - 50:
            stroke = StrokeType.LOB
            conf = 0.55
        else:
            stroke = StrokeType.FOREHAND if is_forehand else StrokeType.BACKHAND
            conf = 0.72

        return StrokeEvent(frame_idx, track_id, stroke, conf)

    def _get_kp(self, pose: PoseKeypoints, *names: str):
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
