from __future__ import annotations

import subprocess
from pathlib import Path

from backend.highlights.interval_merge import HighlightInterval, merge_overlapping_intervals
from backend.utils.ffmpeg import get_ffmpeg
from backend.utils.logging import get_logger
from backend.utils.types import EventType, MatchEvent, RallySegment

logger = get_logger(__name__)


class HighlightGenerator:
    """Stage 14: excitement-scored clips with dedup merge pass."""

    def __init__(self, config: dict):
        hcfg = config["highlights"]
        self.top_k = hcfg["top_k"]
        self.preroll = hcfg["preroll_frames"]
        self.postroll = hcfg["postroll_frames"]
        self.min_excitement = hcfg.get("min_excitement", 45.0)
        self.merge_overlap_s = hcfg.get("merge_overlap_s", 2.0)
        self.fps = config["pipeline"]["target_fps"]
        self.output_dir = Path(config["paths"]["data_reports"]) / "highlights"

    def _excitement_raw(
        self,
        r: RallySegment,
        rally_events: list[MatchEvent],
        max_ball_speed: float,
    ) -> float:
        ace_bonus = sum(1 for e in rally_events if e.event_type == EventType.ACE)
        winner_bonus = sum(1 for e in rally_events if e.event_type == EventType.WINNER)
        net_approach = sum(1 for e in rally_events if e.event_type == EventType.NET_APPROACH)
        long_rally_bonus = 20.0 if r.rally_length_shots > 15 else 0.0
        return (
            r.rally_length_shots * 2.5
            + min(max_ball_speed / 4.0, 30.0)
            + ace_bonus * 15.0
            + winner_bonus * 10.0
            + min(net_approach, 3) * 3.0
            + long_rally_bonus
        )

    def score_rallies(
        self,
        rallies: list[RallySegment],
        events: list[MatchEvent],
        max_ball_speed: float,
        max_distance: float,
    ) -> list[RallySegment]:
        if not rallies:
            return []

        scored: list[RallySegment] = []
        for r in rallies:
            rally_events = [e for e in events if r.start_frame <= e.frame_idx <= r.end_frame]
            r.excitement_score = float(min(100.0, self._excitement_raw(r, rally_events, max_ball_speed)))
            if r.excitement_score >= self.min_excitement:
                scored.append(r)

        if not scored and rallies:
            # Keep best rally when none clear the bar (short phone clips)
            best = max(rallies, key=lambda x: x.excitement_score)
            if best.excitement_score >= self.min_excitement * 0.5:
                scored = [best]

        return sorted(scored, key=lambda r: r.excitement_score, reverse=True)

    def build_intervals(self, rallies: list[RallySegment]) -> list[HighlightInterval]:
        intervals = [
            HighlightInterval(
                start_frame=max(0, r.start_frame - self.preroll),
                end_frame=r.end_frame + self.postroll,
                excitement=r.excitement_score,
                label=f"rally_{r.rally_length_shots}shots",
            )
            for r in rallies
        ]
        return merge_overlapping_intervals(intervals, self.fps, self.merge_overlap_s)

    def extract_clips(self, video_path: str, rallies: list[RallySegment]) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        intervals = self.build_intervals(rallies[: self.top_k * 2])
        paths: list[str] = []
        for i, iv in enumerate(intervals[: self.top_k]):
            start_t = iv.start_frame / self.fps
            end_t = iv.end_frame / self.fps
            out = self.output_dir / f"highlight_{i:02d}.mp4"
            cmd = [
                get_ffmpeg(), "-y",
                "-ss", str(start_t),
                "-to", str(end_t),
                "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart",
                str(out),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                paths.append(str(out))
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("FFmpeg clip extraction failed for highlight %d", i)
        return paths
