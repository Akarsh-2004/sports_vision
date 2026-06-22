"""Phase 11: padel rally segmentation (wall bounces count as activity)."""

from __future__ import annotations

from backend.utils.types import RallySegment


class PointSegmentation:
    """Doubles-aware rally detection with shorter minimum duration."""

    def __init__(self, config: dict):
        self.config = config
        self.fps = config["pipeline"]["target_fps"]
        rcfg = config.get("rally", {})
        self.min_duration_s = rcfg.get("min_duration_s", 2.0)
        self.min_bounces = rcfg.get("min_bounces", 1)
        self.min_shots = rcfg.get("min_shots", 2)
        self.ball_stationary_s = rcfg.get("ball_stationary_s", 1.2)
        self.point_gap_s = rcfg.get("point_gap_s", 5.0)
        self.point_min_shots = rcfg.get("point_min_shots", 1)
        self.min_active_point_s = rcfg.get("min_active_point_s", 4.0)
        self.min_active_point_shots = rcfg.get("min_active_point_shots", 2)
        self._frames: list[dict] = []

    def add_frame_state(
        self,
        frame_idx: int,
        ball_conf: float,
        ball_speed: float,
        players_ready: bool,
        ball_bounce: bool = False,
        walking_to_baseline: bool = False,
        wall_hit: bool = False,
        active_play: bool = True,
    ) -> None:
        self._frames.append(
            {
                "frame": frame_idx,
                "ball_conf": ball_conf,
                "ball_speed": ball_speed,
                "players_ready": players_ready,
                "ball_bounce": ball_bounce or wall_hit,
                "walking": walking_to_baseline,
                "wall_hit": wall_hit,
                "active_play": active_play,
            }
        )

    def segment(self, shot_frames: list[int] | None = None) -> list[RallySegment]:
        if not self._frames:
            return []

        min_frames = int(self.min_duration_s * self.fps)
        stationary_frames = int(self.ball_stationary_s * self.fps)
        rallies: list[RallySegment] = []
        in_rally = False
        start = 0
        bounce_count = 0
        wall_hits = 0
        still_count = 0

        for f in self._frames:
            if not f.get("active_play", True):
                if in_rally:
                    still_count += 1
                    if still_count >= stationary_frames // 2:
                        in_rally = False
                        bounce_count = 0
                        wall_hits = 0
                        still_count = 0
                continue

            ball_active = f["ball_conf"] > 0.2 or f["ball_speed"] > 1.5
            rally_signal = ball_active and f.get("active_play", True)

            if not in_rally:
                if rally_signal:
                    in_rally = True
                    start = f["frame"]
                    bounce_count = 1 if f["ball_bounce"] else 0
                    wall_hits = 1 if f["wall_hit"] else 0
                    still_count = 0
            else:
                if f["ball_bounce"]:
                    bounce_count += 1
                if f["wall_hit"]:
                    wall_hits += 1
                if not ball_active:
                    still_count += 1
                else:
                    still_count = 0

                end_rally = (
                    still_count >= stationary_frames
                    or f["walking"]
                    or (not rally_signal and still_count > self.fps * 0.4)
                )
                if end_rally:
                    end = f["frame"]
                    shots = self._count_shots(start, end, shot_frames or [])
                    if (end - start) >= min_frames and (
                        bounce_count >= self.min_bounces or shots >= self.min_shots
                    ):
                        rallies.append(
                            RallySegment(
                                start_frame=start,
                                end_frame=end,
                                rally_length_shots=shots,
                                wall_hits=wall_hits,
                            )
                        )
                    in_rally = False
                    bounce_count = 0
                    wall_hits = 0
                    still_count = 0

        if in_rally and self._frames:
            end = self._frames[-1]["frame"]
            shots = self._count_shots(start, end, shot_frames or [])
            if (end - start) >= min_frames and shots >= self.min_shots:
                rallies.append(
                    RallySegment(
                        start_frame=start,
                        end_frame=end,
                        rally_length_shots=shots,
                        wall_hits=wall_hits,
                    )
                )

        return rallies

    def segment_from_active_play(
        self,
        active_segments: list[dict],
        shot_frames: list[int],
    ) -> list[RallySegment]:
        """
        Use active-play segments as point boundaries (best for phone / screen recordings).

        Filters out short tails between points (e.g. 3s segment with 2 false hits).
        """
        if not active_segments:
            return []

        preroll = int(0.5 * self.fps)
        postroll = int(1.0 * self.fps)
        rallies: list[RallySegment] = []

        for seg in active_segments:
            duration_s = seg.get("duration_s", 0)
            if duration_s < self.min_active_point_s:
                continue
            start_f = int(seg["start_frame"])
            end_f = int(seg["end_frame"])
            shots_in = [f for f in shot_frames if start_f <= f <= end_f]
            if len(shots_in) < self.min_active_point_shots:
                continue
            rallies.append(
                RallySegment(
                    start_frame=max(0, start_f - preroll),
                    end_frame=end_f + postroll,
                    rally_length_shots=len(shots_in),
                    wall_hits=0,
                )
            )
        return rallies

    def segment_from_shot_clusters(self, shot_frames: list[int]) -> list[RallySegment]:
        """
        Fallback point detection when ball tracking is weak (phone footage).

        Clusters debounced shot timestamps; a gap >= point_gap_s starts a new point.
        """
        if not shot_frames:
            return []

        gap_frames = max(1, int(self.point_gap_s * self.fps))
        preroll = int(1.0 * self.fps)
        postroll = int(1.5 * self.fps)
        min_frames = max(int(0.8 * self.fps), int(self.min_duration_s * self.fps * 0.5))

        sorted_shots = sorted(set(shot_frames))
        clusters: list[list[int]] = [[sorted_shots[0]]]
        for frame in sorted_shots[1:]:
            if frame - clusters[-1][-1] > gap_frames:
                clusters.append([frame])
            else:
                clusters[-1].append(frame)

        rallies: list[RallySegment] = []
        for cluster in clusters:
            if len(cluster) < self.point_min_shots:
                continue
            start = max(0, cluster[0] - preroll)
            end = cluster[-1] + postroll
            duration_s = (end - start) / self.fps
            if duration_s < self.min_active_point_s:
                continue
            if (end - start) < min_frames:
                continue
            rallies.append(
                RallySegment(
                    start_frame=start,
                    end_frame=end,
                    rally_length_shots=len(cluster),
                    wall_hits=0,
                )
            )
        return rallies

    def best_segments(
        self,
        shot_frames: list[int],
        ball_rallies: list[RallySegment] | None = None,
        active_segments: list[dict] | None = None,
    ) -> list[RallySegment]:
        """Pick the most plausible point list — active-play > shot-cluster > ball-based."""
        ball_rallies = ball_rallies or []
        active_rallies = self.segment_from_active_play(active_segments or [], shot_frames)
        shot_rallies = self.segment_from_shot_clusters(shot_frames)

        if len(active_rallies) >= 2:
            return active_rallies
        if len(shot_rallies) >= 2:
            return shot_rallies
        if ball_rallies:
            return ball_rallies
        return active_rallies or shot_rallies

    def _count_shots(self, start: int, end: int, shot_frames: list[int]) -> int:
        return sum(1 for f in shot_frames if start <= f <= end)
