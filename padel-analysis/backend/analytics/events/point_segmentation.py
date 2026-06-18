"""Phase 11: padel rally segmentation (wall bounces count as activity)."""

from __future__ import annotations

from backend.utils.types import RallySegment


class PointSegmentation:
    """Doubles-aware rally detection with shorter minimum duration."""

    def __init__(self, config: dict):
        self.fps = config["pipeline"]["target_fps"]
        rcfg = config.get("rally", {})
        self.min_duration_s = rcfg.get("min_duration_s", 2.0)
        self.min_bounces = rcfg.get("min_bounces", 1)
        self.min_shots = rcfg.get("min_shots", 2)
        self.ball_stationary_s = rcfg.get("ball_stationary_s", 1.2)
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

    def _count_shots(self, start: int, end: int, shot_frames: list[int]) -> int:
        return sum(1 for f in shot_frames if start <= f <= end)
