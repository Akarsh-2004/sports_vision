from __future__ import annotations

import cv2
import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import BBox, PoseKeypoints

logger = get_logger(__name__)

MP_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class PoseEstimator:
    """Stage 9: MediaPipe Pose (MVP)."""

    def __init__(self, config: dict):
        self.keyframe_interval = config["pipeline"].get("pose_keyframe_interval", 3)
        self._mp_pose = None
        self._pose = None
        self._load()

    def _load(self) -> None:
        try:
            import mediapipe as mp

            self._mp_pose = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose loaded")
        except Exception as exc:
            logger.warning("MediaPipe unavailable: %s", exc)

    def estimate(
        self, frame: np.ndarray, frame_idx: int, track_id: int, bbox: BBox | None = None
    ) -> PoseKeypoints | None:
        if frame_idx % self.keyframe_interval != 0:
            return None
        if self._pose is None:
            return self._fallback_pose(frame_idx, track_id, bbox)

        crop = frame
        offset = (0, 0)
        if bbox is not None:
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = int(bbox.x2), int(bbox.y2)
            crop = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        kps: dict[str, tuple[float, float, float]] = {}
        h, w = crop.shape[:2]
        for i, name in enumerate(MP_LANDMARKS):
            if i >= len(result.pose_landmarks.landmark):
                break
            lm = result.pose_landmarks.landmark[i]
            kps[name] = (lm.x * w + offset[0], lm.y * h + offset[1], lm.visibility)

        return PoseKeypoints(frame_idx=frame_idx, track_id=track_id, keypoints=kps)

    def _fallback_pose(
        self, frame_idx: int, track_id: int, bbox: BBox | None
    ) -> PoseKeypoints | None:
        if bbox is None:
            return None
        cx, cy = bbox.centroid
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        return PoseKeypoints(
            frame_idx=frame_idx,
            track_id=track_id,
            keypoints={
                "left_shoulder": (cx - bw * 0.15, cy - bh * 0.2, 1.0),
                "right_shoulder": (cx + bw * 0.15, cy - bh * 0.2, 1.0),
                "left_wrist": (cx - bw * 0.25, cy, 1.0),
                "right_wrist": (cx + bw * 0.25, cy, 1.0),
                "left_hip": (cx - bw * 0.1, cy + bh * 0.15, 1.0),
                "right_hip": (cx + bw * 0.1, cy + bh * 0.15, 1.0),
            },
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
