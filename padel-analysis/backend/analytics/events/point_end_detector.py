"""Classify whether a rally segment ended with a completed point."""

from __future__ import annotations

from backend.analytics.events.rally_gating import detect_baseline_reset
from backend.utils.types import RallySegment


class PointEndDetector:
    """
    After a rally closes, inspect post-rally frames to decide if a point finished.

    Returns: point_complete | interrupted | ambiguous
    """

    def __init__(self, config: dict, court_length_m: float, court_width_m: float):
        self.fps = config["pipeline"]["target_fps"]
        self.court_length = court_length_m
        self.court_width = court_width_m
        self.net_y = court_length_m / 2.0
        rcfg = config.get("rally", {})
        self.post_window_frames = int(rcfg.get("point_end_post_window_s", 2.0) * self.fps)
        self.ball_idle_frames = int(rcfg.get("ball_stationary_s", 1.2) * self.fps)
        self.ball_lost_frames = int(rcfg.get("point_end_ball_lost_s", 3.0) * self.fps)
        self.min_point_duration_s = rcfg.get("min_active_point_s", 4.0)

    def build_frame_index(
        self,
        point_frames: list[dict],
        ball_court_positions: dict[int, tuple[float, float]],
        player_court_positions: dict[int, dict[int, tuple[float, float]]],
    ) -> dict[int, dict]:
        index: dict[int, dict] = {}
        for f in point_frames:
            idx = f["frame"]
            index[idx] = {
                **f,
                "ball_court": ball_court_positions.get(idx),
                "players": player_court_positions.get(idx, {}),
            }
        for idx, pos in ball_court_positions.items():
            if idx not in index:
                index[idx] = {
                    "frame": idx,
                    "ball_conf": 0.0,
                    "ball_speed": 0.0,
                    "active_play": False,
                    "walking": False,
                    "ball_court": pos,
                    "players": player_court_positions.get(idx, {}),
                }
        return index

    def classify_end(
        self,
        rally: RallySegment,
        frame_index: dict[int, dict],
    ) -> str:
        duration_s = (rally.end_frame - rally.start_frame) / self.fps
        if duration_s < 1.0:
            return "interrupted"

        post_frames = self._window(frame_index, rally.end_frame, rally.end_frame + self.post_window_frames)
        if self._players_walking_to_baseline(post_frames):
            return "point_complete"
        if self._ball_left_court(post_frames):
            return "point_complete"
        if self._ball_stationary_long(rally, frame_index):
            return "point_complete"
        if self._ball_lost_extended(rally, frame_index):
            return "point_complete"
        if duration_s >= self.min_point_duration_s:
            # Active-play bounded segments without strong post-signals are still real points
            return "point_complete"
        return "interrupted"

    def _window(self, frame_index: dict[int, dict], start: int, end: int) -> list[dict]:
        return [
            frame_index[i]
            for i in sorted(frame_index)
            if start <= i <= end
        ]

    def _players_walking_to_baseline(self, post_frames: list[dict]) -> bool:
        if len(post_frames) < 3:
            return False
        for i in range(1, len(post_frames)):
            prev = post_frames[i - 1]
            curr = post_frames[i]
            if detect_baseline_reset(prev.get("players", {}), curr.get("players", {}), self.court_length):
                return True
            if curr.get("walking"):
                return True
        return False

    def _ball_left_court(self, post_frames: list[dict]) -> bool:
        margin = 0.35
        for f in post_frames:
            pos = f.get("ball_court")
            if not pos:
                continue
            x, y = pos
            if x < -margin or x > self.court_width + margin or y < -margin or y > self.court_length + margin:
                return True
        return False

    def _ball_stationary_long(self, rally: RallySegment, frame_index: dict[int, dict]) -> bool:
        idle = 0
        for idx in sorted(frame_index):
            if idx < rally.end_frame:
                continue
            f = frame_index[idx]
            if f.get("ball_conf", 0) > 0.15 or f.get("ball_speed", 0) > 1.5:
                idle = 0
            else:
                idle += 1
            if idle >= self.ball_idle_frames:
                return True
        return False

    def _ball_lost_extended(self, rally: RallySegment, frame_index: dict[int, dict]) -> bool:
        lost = 0
        for idx in sorted(frame_index):
            if idx < rally.start_frame or idx > rally.end_frame + self.ball_lost_frames:
                continue
            f = frame_index[idx]
            if f.get("ball_conf", 0) < 0.12 and f.get("ball_speed", 0) < 1.0:
                lost += 1
            else:
                lost = 0
            if lost >= self.ball_lost_frames and idx >= rally.end_frame - int(self.fps * 0.5):
                return True
        return False

    @staticmethod
    def infer_winner(
        rally: RallySegment,
        frame_index: dict[int, dict],
        court_length_m: float,
    ) -> str | None:
        """Best-effort: ball dead in team half → other team wins."""
        net_y = court_length_m / 2.0
        last_pos = None
        for idx in sorted(frame_index):
            if rally.start_frame <= idx <= rally.end_frame + 30:
                pos = frame_index[idx].get("ball_court")
                if pos and frame_index[idx].get("ball_conf", 0) > 0.1:
                    last_pos = pos
        if not last_pos:
            return None
        _, cy = last_pos
        if cy < net_y:
            return "B"
        return "A"
