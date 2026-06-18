"""
Padel mini-court overlay — adapted from abdullahtarek/tennis_analysis mini_court.

Renders 10m × 20m top-down court in corner of broadcast frame.
https://github.com/abdullahtarek/tennis_analysis
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.court.geometry import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, SERVICE_DEPTH_M


class PadelMiniCourt:
    """Draw padel court diagram + player/ball positions on video frames."""

    def __init__(self, frame: np.ndarray, width_px: int = 220, height_px: int = 440):
        self.court_w_m = COURT_WIDTH_M
        self.court_l_m = COURT_LENGTH_M
        self.buffer = 40
        self.width_px = width_px
        self.height_px = height_px
        self._set_canvas_position(frame)
        self._set_court_rect()

    def _set_canvas_position(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.end_x = w - self.buffer
        self.end_y = self.buffer + self.height_px
        self.start_x = self.end_x - self.width_px
        self.start_y = self.end_y - self.height_px

    def _set_court_rect(self) -> None:
        pad = 16
        self.cx0 = self.start_x + pad
        self.cy0 = self.start_y + pad
        self.cx1 = self.end_x - pad
        self.cy1 = self.end_y - pad
        self.draw_w = self.cx1 - self.cx0
        self.draw_h = self.cy1 - self.cy0

    def court_to_mini(self, x_m: float, y_m: float) -> tuple[int, int]:
        px = int(self.cx0 + (x_m / self.court_w_m) * self.draw_w)
        py = int(self.cy0 + (y_m / self.court_l_m) * self.draw_h)
        return px, py

    def draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        overlay = np.zeros_like(out)
        cv2.rectangle(overlay, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)

        # Outer court
        p00 = self.court_to_mini(0, 0)
        p11 = self.court_to_mini(self.court_w_m, self.court_l_m)
        cv2.rectangle(out, p00, p11, (40, 40, 40), 2)

        # Net
        nl = self.court_to_mini(0, NET_Y_M)
        nr = self.court_to_mini(self.court_w_m, NET_Y_M)
        cv2.line(out, nl, nr, (0, 120, 255), 2)

        # Service lines
        for y in (SERVICE_DEPTH_M, self.court_l_m - SERVICE_DEPTH_M):
            a = self.court_to_mini(0, y)
            b = self.court_to_mini(self.court_w_m, y)
            cv2.line(out, a, b, (80, 80, 80), 1)

        # Center line
        c0 = self.court_to_mini(self.court_w_m / 2, 0)
        c1 = self.court_to_mini(self.court_w_m / 2, self.court_l_m)
        cv2.line(out, c0, c1, (80, 80, 80), 1)

        # Glass zones (dashed feel — thin outer bands)
        for y in (0.5, self.court_l_m - 0.5):
            a = self.court_to_mini(0.3, y)
            b = self.court_to_mini(self.court_w_m - 0.3, y)
            cv2.line(out, a, b, (180, 180, 180), 1)

        cv2.putText(
            out,
            "TOP VIEW",
            (self.start_x + 8, self.start_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        return out

    def draw_points(
        self,
        frame: np.ndarray,
        positions: list[tuple[float, float]],
        color: tuple[int, int, int] = (0, 255, 100),
        radius: int = 5,
    ) -> np.ndarray:
        out = frame.copy()
        for x_m, y_m in positions:
            if 0 <= x_m <= self.court_w_m and 0 <= y_m <= self.court_l_m:
                px, py = self.court_to_mini(x_m, y_m)
                cv2.circle(out, (px, py), radius, color, -1)
        return out

    def draw_frame(
        self,
        frame: np.ndarray,
        players: list[tuple[float, float, tuple[int, int, int]]] | None = None,
        ball: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """
        players: list of (x_m, y_m, bgr_color)
        ball: (x_m, y_m) court coordinates
        """
        out = self.draw_court_lines(frame)
        if players:
            for x_m, y_m, color in players:
                if 0 <= x_m <= self.court_w_m and 0 <= y_m <= self.court_l_m:
                    px, py = self.court_to_mini(x_m, y_m)
                    cv2.circle(out, (px, py), 6, color, -1)
        if ball and 0 <= ball[0] <= self.court_w_m and 0 <= ball[1] <= self.court_l_m:
            px, py = self.court_to_mini(ball[0], ball[1])
            cv2.circle(out, (px, py), 4, (0, 255, 255), -1)
        return out
