from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from backend.utils.types import BallDetection


class BallTracker:
    """Stage 7: Kalman filter + physics interpolation for ball trajectory."""

    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.kf = self._make_kalman()
        self.missed = 0
        self.max_missed = 5
        self.trajectory: list[tuple[int, float, float, float]] = []
        self.initialized = False

    def _make_kalman(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.R *= 10
        kf.Q *= 0.1
        kf.P *= 10
        return kf

    def update(self, frame_idx: int, det: BallDetection) -> tuple[float, float, float]:
        if det.visible and det.confidence > 0.2:
            if not self.initialized:
                self.kf.x = np.array([det.x, det.y, 0, 0], dtype=float)
                self.initialized = True
            else:
                self.kf.predict()
                self.kf.update(np.array([det.x, det.y]))
            self.missed = 0
            x, y = float(self.kf.x[0]), float(self.kf.x[1])
            conf = det.confidence
        else:
            self.missed += 1
            if self.initialized and self.missed <= self.max_missed:
                self.kf.predict()
                x, y = float(self.kf.x[0]), float(self.kf.x[1])
                conf = max(0.1, 0.5 - self.missed * 0.08)
            else:
                x, y, conf = 0.0, 0.0, 0.0
                self.initialized = False

        if conf > 0:
            self.trajectory.append((frame_idx, x, y, conf))
        return x, y, conf

    def get_speed_kmh(self, court_scale_m_per_px: float = 0.02) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        f1, x1, y1, _ = self.trajectory[-2]
        f2, x2, y2, _ = self.trajectory[-1]
        dt = max((f2 - f1) / self.fps, 1e-6)
        dist_m = np.hypot(x2 - x1, y2 - y1) * court_scale_m_per_px
        return dist_m / dt * 3.6

    def reset(self) -> None:
        self.kf = self._make_kalman()
        self.missed = 0
        self.initialized = False
