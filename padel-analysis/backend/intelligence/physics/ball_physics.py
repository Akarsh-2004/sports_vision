"""Ball physics engine — trajectory, bounce, glass reflection, repair."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend.intelligence.geometry.entities import BallEntity, Vec2


@dataclass
class PhysicsState:
    position: Vec2
    velocity: Vec2  # m/s
    visible: bool = True
    confidence: float = 0.0
    bounce_predicted: bool = False
    wall_reflection: bool = False
    repaired: bool = False


@dataclass
class BallPhysicsEngine:
    """
    Detection → trajectory prediction → bounce physics → glass reflection
    → Kalman blend → repair missing frames.
    """

    court_width: float = 10.0
    court_length: float = 20.0
    wall_offset: float = 3.0
    gravity: float = 9.81
    restitution_ground: float = 0.75
    restitution_glass: float = 0.55
    max_miss_frames: int = 8
    trajectory: list[tuple[int, PhysicsState]] = field(default_factory=list)
    _kalman_pos: np.ndarray | None = None
    _kalman_vel: np.ndarray | None = None
    _miss_count: int = 0

    def update(
        self,
        frame_idx: int,
        detected: BallEntity | None,
        fps: float,
    ) -> PhysicsState:
        dt = 1.0 / fps

        if detected and detected.visible and detected.kinematics.confidence > 0.1:
            meas = np.array([detected.kinematics.position.x, detected.kinematics.position.y])
            meas_vel = np.array(
                [
                    detected.kinematics.speed_mps * math.cos(math.radians(detected.kinematics.direction_deg)),
                    detected.kinematics.speed_mps * math.sin(math.radians(detected.kinematics.direction_deg)),
                ]
            )
            if self._kalman_pos is None:
                self._kalman_pos = meas.copy()
                self._kalman_vel = meas_vel.copy()
            else:
                alpha = 0.35
                self._kalman_pos = alpha * meas + (1 - alpha) * (self._kalman_pos + self._kalman_vel * dt)
                self._kalman_vel = alpha * meas_vel + (1 - alpha) * self._kalman_vel
            self._miss_count = 0
            conf = float(detected.kinematics.confidence)
            repaired = False
        else:
            self._miss_count += 1
            if self._kalman_pos is not None and self._miss_count <= self.max_miss_frames:
                self._kalman_pos = self._kalman_pos + self._kalman_vel * dt
                self._apply_physics_constraints()
                conf = max(0.15, 0.7 - self._miss_count * 0.08)
                repaired = True
            else:
                conf = 0.0
                repaired = False

        if self._kalman_pos is None:
            state = PhysicsState(Vec2(0, 0), Vec2(0, 0), visible=False, confidence=0)
        else:
            speed = float(np.linalg.norm(self._kalman_vel))
            bounce = self._detect_bounce()
            wall = self._detect_wall_reflection()
            state = PhysicsState(
                position=Vec2(float(self._kalman_pos[0]), float(self._kalman_pos[1])),
                velocity=Vec2(float(self._kalman_vel[0]), float(self._kalman_vel[1])),
                visible=conf > 0.2,
                confidence=conf,
                bounce_predicted=bounce,
                wall_reflection=wall,
                repaired=repaired,
            )

        self.trajectory.append((frame_idx, state))
        return state

    def _apply_physics_constraints(self) -> None:
        if self._kalman_pos is None or self._kalman_vel is None:
            return
        x, y = self._kalman_pos
        if y < 0:
            self._kalman_pos[1] = -y * self.restitution_ground
            self._kalman_vel[1] = abs(self._kalman_vel[1]) * self.restitution_ground
        if y > self.court_length:
            self._kalman_pos[1] = 2 * self.court_length - y
            self._kalman_vel[1] = -abs(self._kalman_vel[1]) * self.restitution_ground
        if x < 0 or x > self.court_width:
            self._kalman_vel[0] *= -self.restitution_glass

    def _detect_bounce(self) -> bool:
        if len(self.trajectory) < 3:
            return False
        _, s1 = self.trajectory[-2]
        _, s2 = self.trajectory[-1]
        return s1.velocity.y * s2.velocity.y < 0 and s2.position.y < 1.5

    def _detect_wall_reflection(self) -> bool:
        if len(self.trajectory) < 3:
            return False
        _, s1 = self.trajectory[-2]
        _, s2 = self.trajectory[-1]
        v1 = np.array([s1.velocity.x, s1.velocity.y])
        v2 = np.array([s2.velocity.x, s2.velocity.y])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 0.5 or n2 < 0.5:
            return False
        angle = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
        return angle > 50

    def court_trajectory(self) -> list[tuple[int, float, float, float]]:
        return [
            (f, s.position.x, s.position.y, s.confidence)
            for f, s in self.trajectory
            if s.confidence > 0.15
        ]
