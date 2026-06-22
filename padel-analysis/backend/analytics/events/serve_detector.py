"""Detect serve frame at the start of a point."""

from __future__ import annotations

from backend.utils.types import RallySegment


class ServeDetector:
    """
    Heuristic serve detection after dead time:
    - ball near baseline
    - low player motion at back court
    - upward then forward ball motion
    """

    def __init__(self, config: dict, court_length_m: float):
        self.fps = config["pipeline"]["target_fps"]
        self.court_length = court_length_m
        self.net_y = court_length_m / 2.0
        self.search_frames = int(3.0 * self.fps)
        self.dead_gap_frames = int(3.0 * self.fps)
        self.baseline_band_m = 3.5

    def detect(
        self,
        rally: RallySegment,
        frame_index: dict[int, dict],
        dead_segment_end: int | None = None,
    ) -> int | None:
        search_start = dead_segment_end if dead_segment_end is not None else max(0, rally.start_frame - self.search_frames)
        search_end = min(rally.end_frame, rally.start_frame + self.search_frames)

        best_frame: int | None = None
        best_score = 0.0

        sorted_frames = sorted(
            i for i in frame_index if search_start <= i <= search_end
        )
        for i, frame_idx in enumerate(sorted_frames):
            f = frame_index[frame_idx]
            score = 0.0
            if self._ball_at_baseline(f):
                score += 0.4
            if self._single_player_stationary(f, frame_index, frame_idx):
                score += 0.35
            if self._upward_launch(frame_index, sorted_frames, i):
                score += 0.25
            if score > best_score:
                best_score = score
                best_frame = frame_idx

        return best_frame if best_score >= 0.55 else None

    def dead_gap_before(self, rally: RallySegment, active_segments: list[dict]) -> int | None:
        """Frame index where dead time ended before this rally's active segment."""
        for seg in active_segments:
            if seg["start_frame"] <= rally.start_frame <= seg["end_frame"]:
                return max(0, seg["start_frame"] - 1)
        return max(0, rally.start_frame - self.dead_gap_frames)

    def _ball_at_baseline(self, frame: dict) -> bool:
        pos = frame.get("ball_court")
        if not pos:
            return False
        _, cy = pos
        near_top = cy <= self.baseline_band_m
        near_bottom = cy >= self.court_length - self.baseline_band_m
        return near_top or near_bottom

    def _single_player_stationary(
        self,
        frame: dict,
        frame_index: dict[int, dict],
        frame_idx: int,
    ) -> bool:
        players = frame.get("players") or {}
        if len(players) < 1:
            return False
        back_court = 0
        stationary = 0
        for _pid, (x, y) in players.items():
            at_back = y <= self.baseline_band_m or y >= self.court_length - self.baseline_band_m
            if at_back:
                back_court += 1
                prev = frame_index.get(frame_idx - 2, {}).get("players", {}).get(_pid)
                if prev:
                    speed = abs(y - prev[1]) * self.fps
                    if speed < 1.2:
                        stationary += 1
        return back_court >= 1 and stationary >= 1

    def _upward_launch(
        self,
        frame_index: dict[int, dict],
        sorted_frames: list[int],
        idx: int,
    ) -> bool:
        if idx < 2:
            return False
        f0 = frame_index.get(sorted_frames[idx - 2], {})
        f1 = frame_index.get(sorted_frames[idx - 1], {})
        f2 = frame_index.get(sorted_frames[idx], {})
        p0, p1, p2 = f0.get("ball_court"), f1.get("ball_court"), f2.get("ball_court")
        if not (p0 and p1 and p2):
            return False
        # court y increases toward bottom; upward toss = y decreases then increases forward
        dy1 = p1[1] - p0[1]
        dy2 = p2[1] - p1[1]
        speed_rise = f2.get("ball_speed", 0) > f0.get("ball_speed", 0)
        return dy1 < -0.15 and dy2 > dy1 and speed_rise
