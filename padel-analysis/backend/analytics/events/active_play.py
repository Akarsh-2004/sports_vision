"""Filter dead time, audience close-ups, and between-point waiting from broadcast footage."""

from __future__ import annotations

import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import CourtState, PlayerDetection

logger = get_logger(__name__)


class ActivePlayGate:
    """
    Marks frames where real match play occurs.

    Ignores:
    - Audience / player face close-ups (oversized bboxes)
    - Between-point walking and wait time (no ball + slow players)
    - Court cutaways with no on-court players
    """

    def __init__(self, config: dict, fps: float, audio_hits: set[int] | None = None):
        ap = config.get("active_play", {})
        self.enabled = ap.get("enabled", True)
        self.fps = fps
        self.min_players = ap.get("min_players_on_court", 2)
        self.min_ball_speed_kmh = ap.get("min_ball_speed_kmh", 4.0)
        self.min_ball_conf = ap.get("min_ball_conf", 0.18)
        self.min_player_motion_mps = ap.get("min_player_motion_mps", 0.9)
        self.dead_ball_idle_s = ap.get("dead_time_ball_idle_s", 2.5)
        self.ball_activity_tail_s = ap.get("ball_activity_tail_s", 1.2)
        self.audio_window_s = ap.get("audio_hit_window_s", 1.0)
        self.closeup_bbox_ratio = ap.get("closeup_bbox_area_ratio", 0.28)
        self.min_segment_s = ap.get("min_active_segment_s", 1.5)
        self.merge_gap_s = ap.get("merge_gap_s", 2.0)
        self.ignore_closeup = ap.get("ignore_audience_closeup", True)

        self.audio_hits = audio_hits or set()
        self.audio_window_frames = int(self.audio_window_s * fps)
        self.dead_idle_frames = int(self.dead_ball_idle_s * fps)
        self.ball_tail_frames = int(self.ball_activity_tail_s * fps)
        self._last_ball_active_frame = -10_000
        self._ball_idle_frames = 0
        self._flags: list[tuple[int, bool]] = []
        self._skipped_closeup = 0
        self._skipped_dead = 0

    def evaluate(
        self,
        frame_idx: int,
        players: list[PlayerDetection],
        ball_conf: float,
        ball_speed_kmh: float,
        court_state: CourtState,
        frame_shape: tuple[int, ...],
        prev_positions: dict[int, tuple[float, float]],
        court_length_m: float = 20.0,
    ) -> bool:
        if not self.enabled:
            self._flags.append((frame_idx, True))
            return True

        h, w = frame_shape[:2]
        frame_area = float(h * w)

        if self.ignore_closeup and self._is_audience_closeup(players, frame_area):
            self._skipped_closeup += 1
            self._flags.append((frame_idx, False))
            return False

        on_court = self._players_on_court(players, court_state, court_length_m, frame_area)
        if on_court < self.min_players:
            self._flags.append((frame_idx, False))
            return False

        ball_active = ball_conf >= self.min_ball_conf or ball_speed_kmh >= self.min_ball_speed_kmh
        if ball_active:
            self._last_ball_active_frame = frame_idx
            self._ball_idle_frames = 0
        else:
            self._ball_idle_frames += 1

        audio_near = self._audio_near(frame_idx)
        recent_ball = (frame_idx - self._last_ball_active_frame) <= self.ball_tail_frames
        player_speed = self._max_player_speed_mps(players, prev_positions)
        players_moving = player_speed >= self.min_player_motion_mps

        dead_time = (
            self._ball_idle_frames >= self.dead_idle_frames
            and not players_moving
            and not audio_near
            and not recent_ball
        )
        if dead_time:
            self._skipped_dead += 1
            self._flags.append((frame_idx, False))
            return False

        active = (
            ball_active
            or audio_near
            or recent_ball
            or (players_moving and on_court >= self.min_players)
        )
        self._flags.append((frame_idx, active))
        return active

    def build_segments(self) -> list[dict]:
        """Merge active frames into contiguous play segments."""
        if not self._flags:
            return []

        min_frames = int(self.min_segment_s * self.fps)
        gap_frames = int(self.merge_gap_s * self.fps)
        segments: list[dict] = []
        start: int | None = None
        last_active = -1

        for frame_idx, active in self._flags:
            if active:
                if start is None:
                    start = frame_idx
                last_active = frame_idx
            elif start is not None and (frame_idx - last_active) > gap_frames:
                if last_active - start + 1 >= min_frames:
                    segments.append(
                        {
                            "start_frame": start,
                            "end_frame": last_active,
                            "duration_s": (last_active - start + 1) / self.fps,
                        }
                    )
                start = None

        if start is not None and last_active >= start:
            if last_active - start + 1 >= min_frames:
                segments.append(
                    {
                        "start_frame": start,
                        "end_frame": last_active,
                        "duration_s": (last_active - start + 1) / self.fps,
                    }
                )

        logger.info(
            "Active play: %d segments, skipped %d close-up frames, %d dead-time frames",
            len(segments),
            self._skipped_closeup,
            self._skipped_dead,
        )
        return segments

    def active_frame_count(self) -> int:
        return sum(1 for _, a in self._flags if a)

    def _is_audience_closeup(self, players: list[PlayerDetection], frame_area: float) -> bool:
        if not players:
            return True
        ratios = [p.bbox.area / max(frame_area, 1.0) for p in players]
        if max(ratios) >= self.closeup_bbox_ratio:
            return True
        if len(players) == 1 and ratios[0] > 0.12:
            return True
        return False

    def _players_on_court(
        self,
        players: list[PlayerDetection],
        court_state: CourtState,
        court_length_m: float,
        frame_area: float,
    ) -> int:
        count = 0
        for p in players:
            if p.court_xy and court_state.valid_for_analytics:
                cx, cy = p.court_xy
                if -1.0 <= cx <= 11.0 and -1.0 <= cy <= court_length_m + 1.0:
                    count += 1
                    continue
            ratio = p.bbox.area / max(frame_area, 1.0)
            if 0.005 < ratio < 0.22:
                count += 1
        return count

    def _max_player_speed_mps(
        self,
        players: list[PlayerDetection],
        prev_positions: dict[int, tuple[float, float]],
    ) -> float:
        max_speed = 0.0
        for p in players:
            prev = prev_positions.get(p.track_id)
            if prev is None or p.court_xy is None:
                continue
            cx, cy = p.court_xy
            px, py = prev
            max_speed = max(max_speed, float(np.hypot(cx - px, cy - py) * self.fps))
        return max_speed

    def _audio_near(self, frame_idx: int) -> bool:
        if not self.audio_hits:
            return False
        for hit in self.audio_hits:
            if abs(frame_idx - hit) <= self.audio_window_frames:
                return True
        return False

    def is_frame_active(self, frame_idx: int) -> bool:
        for f, active in self._flags:
            if f == frame_idx:
                return active
        return False

    def filter_frames(self, frame_indices: list[int]) -> list[int]:
        active_set = {f for f, a in self._flags if a}
        return [f for f in frame_indices if f in active_set]
