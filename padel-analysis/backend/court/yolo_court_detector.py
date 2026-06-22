"""YOLOv8-pose court keypoint detector for padel homography."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.court.geometry import COURT_LENGTH_M, COURT_WIDTH_M
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Category order from Roboflow export (ids 1..26).
KEYPOINT_TO_COURT_M = {
    8: (0.0, 0.0),  # court_bottom_left_close  -> near left
    9: (COURT_WIDTH_M, 0.0),  # court_bottom_right_close -> near right
    12: (0.0, COURT_LENGTH_M),  # court_top_left_far -> far left
    14: (COURT_WIDTH_M, COURT_LENGTH_M),  # court_top_right_far -> far right
    16: (0.0, COURT_LENGTH_M / 2),  # net_bottom_left
    17: (COURT_WIDTH_M, COURT_LENGTH_M / 2),  # net_bottom_right
    18: (0.0, COURT_LENGTH_M / 2),  # net_top_left
    19: (COURT_WIDTH_M, COURT_LENGTH_M / 2),  # net_top_right
}

HOMOGRAPHY_KPT_IDX = (8, 9, 12, 14)  # 0-based indices into 26-keypoint vector


class YoloCourtKeypointDetector:
    """Runs fine-tuned YOLOv8-pose and estimates court homography."""

    def __init__(self, weights_path: str | Path, conf: float = 0.25, device: str = "auto"):
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.conf = conf
        self.model = YOLO(str(self.weights_path))
        self.device = device
        logger.info("Loaded court keypoint model: %s", self.weights_path)

    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return (26,3) keypoints array [x,y,conf] and mean confidence."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            verbose=False,
            device=self.device,
        )
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None, 0.0

        kpts = results[0].keypoints.data
        if kpts is None or len(kpts) == 0:
            return None, 0.0

        best = kpts[0].cpu().numpy().reshape(-1, 3)
        if best.shape[0] < 26:
            return None, 0.0

        conf = float(best[:, 2].mean())
        return best[:26], conf

    def homography_from_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        keypoints, model_conf = self.detect_keypoints(frame)
        if keypoints is None:
            return None, 0.0

        src_pts: list[list[float]] = []
        dst_pts: list[list[float]] = []
        for idx in HOMOGRAPHY_KPT_IDX:
            x, y, c = keypoints[idx]
            if c < 0.25 or idx not in KEYPOINT_TO_COURT_M:
                continue
            src_pts.append([float(x), float(y)])
            dst_pts.append(list(KEYPOINT_TO_COURT_M[idx]))

        if len(src_pts) < 4:
            return None, model_conf * 0.5

        src = np.float32(src_pts[:4])
        dst = np.float32(dst_pts[:4])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inlier_ratio = float(mask.sum() / 4.0) if mask is not None else 0.5
        conf = model_conf * inlier_ratio
        return H, conf
