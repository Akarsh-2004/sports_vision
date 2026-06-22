"""
Broadcast-style annotated highlight video exporter for padel analysis.

Draws per-frame overlays:
  • Ball trajectory trail (fading yellow dots, last N frames)
  • Player bounding boxes with team color + ID label
  • Mini court diagram (top-down 10×20m) in top-right corner
  • Ball speed meter (km/h gauge, bottom-left)
  • Excitement bar (bottom center)
  • Rally info banner (top center)

Usage:
    from backend.visualization.annotated_exporter import AnnotatedExporter

    exporter = AnnotatedExporter(config, fps=25.0)
    out_path = exporter.export(
        video_path="clip.mp4",
        out_path="clip_annotated.mp4",
        frame_data=frame_data,   # dict[int, FrameAnnotation]
        highlight=highlight,     # CoachingHighlight metadata
        start_frame=0,
        end_frame=125,
    )
"""

from __future__ import annotations

import math
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.utils.logging import get_logger
from backend.visualization.mini_court import PadelMiniCourt

logger = get_logger(__name__)

# ── team colours (BGR) ────────────────────────────────────────────────────────
TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (50, 220, 255),    # Team 0 → cyan-gold
    1: (255, 80, 50),     # Team 1 → coral-red
    2: (80, 255, 160),    # extras
    3: (220, 80, 255),
}
BALL_COLOR = (0, 230, 255)        # vivid yellow
TRAIL_BASE = (0, 180, 220)        # trail fades toward this
ACCENT = (255, 200, 0)            # gold accent


# ── per-frame annotation data ─────────────────────────────────────────────────

@dataclass
class PlayerAnnotation:
    track_id: int
    team: int
    x1: float
    y1: float
    x2: float
    y2: float
    court_x: float | None = None
    court_y: float | None = None
    label: str = ""


@dataclass
class FrameAnnotation:
    frame_idx: int
    ball_px: tuple[float, float] | None = None
    ball_court: tuple[float, float] | None = None
    ball_speed_kmh: float = 0.0
    ball_conf: float = 0.0
    players: list[PlayerAnnotation] = field(default_factory=list)
    excitement: float = 0.0
    rally_length: int = 0
    state: str = ""


# ── main exporter ─────────────────────────────────────────────────────────────

class AnnotatedExporter:
    """
    Render broadcast-style annotated MP4 for a single highlight clip.

    If frame_data is empty or None, falls back to copying the raw clip.
    """

    def __init__(self, config: dict, fps: float = 25.0):
        self.config = config
        self.fps = fps
        acfg = config.get("coach_highlights", {})
        self.style = acfg.get("annotation_style", "broadcast")
        self.trail_len = acfg.get("ball_trail_frames", 12)
        self.show_mini_court = acfg.get("show_mini_court", True)
        self.show_speed_meter = acfg.get("show_speed_meter", True)
        self.show_excitement = acfg.get("show_excitement_bar", True)
        self.show_player_ids = acfg.get("show_player_ids", True)

    # ── public API ─────────────────────────────────────────────────────────────

    def export(
        self,
        video_path: str,
        out_path: str | Path,
        frame_data: dict[int, FrameAnnotation],
        highlight: Any | None = None,   # CoachingHighlight or dict
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> str:
        """
        Read frames from video_path[start_frame:end_frame], draw overlays,
        write annotated MP4 to out_path. Returns str(out_path).
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return ""

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        end_frame = min(end_frame or total, total)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp_path = out_path.with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, self.fps, (w, h))

        if not writer.isOpened():
            logger.warning("VideoWriter failed for %s", tmp_path)
            cap.release()
            return ""

        # Initialise mini court on a blank frame (will re-create per frame for sizing)
        mini: PadelMiniCourt | None = None
        ball_trail: deque[tuple[float, float, float]] = deque(maxlen=self.trail_len)

        # Commentary text from highlight
        commentary = ""
        if highlight:
            if hasattr(highlight, "commentary"):
                commentary = highlight.commentary
            elif isinstance(highlight, dict):
                commentary = highlight.get("commentary", "")

        try:
            for idx in range(start_frame, end_frame):
                ok, frame = cap.read()
                if not ok:
                    break

                ann = frame_data.get(idx)
                if mini is None and ann is not None:
                    mini = PadelMiniCourt(frame, width_px=200, height_px=400)

                frame = self._draw_frame(
                    frame, idx, ann, ball_trail, mini, commentary, highlight
                )
                writer.write(frame)
        finally:
            cap.release()
            writer.release()

        # Re-encode with ffmpeg for proper H.264 + AAC
        self._reencode(video_path, tmp_path, out_path, start_frame, end_frame)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

        logger.debug("Annotated clip → %s", out_path)
        return str(out_path)

    # ── frame drawing ──────────────────────────────────────────────────────────

    def _draw_frame(
        self,
        frame: np.ndarray,
        idx: int,
        ann: FrameAnnotation | None,
        ball_trail: deque,
        mini: PadelMiniCourt | None,
        commentary: str,
        highlight: Any,
    ) -> np.ndarray:
        if ann is None:
            return frame

        # Update ball trail
        if ann.ball_px and ann.ball_conf > 0.15:
            ball_trail.append((*ann.ball_px, ann.ball_conf))

        # Draw layers in order (back to front)
        frame = self._draw_ball_trail(frame, ball_trail)
        if self.show_player_ids:
            frame = self._draw_players(frame, ann.players)
        frame = self._draw_ball(frame, ann.ball_px, ann.ball_conf)
        if self.show_mini_court and mini is not None:
            frame = self._draw_mini_court(frame, ann, mini)
        if self.show_speed_meter:
            frame = self._draw_speed_meter(frame, ann.ball_speed_kmh)
        if self.show_excitement:
            frame = self._draw_excitement_bar(frame, ann.excitement)
        frame = self._draw_rally_banner(frame, ann, commentary)

        return frame

    # ── overlay primitives ─────────────────────────────────────────────────────

    def _draw_ball_trail(
        self, frame: np.ndarray, trail: deque
    ) -> np.ndarray:
        """Draw fading ball trajectory trail."""
        if not trail:
            return frame
        pts = list(trail)
        n = len(pts)
        out = frame.copy()
        for i, (x, y, conf) in enumerate(pts):
            alpha = (i + 1) / n          # 0→1 (oldest→newest)
            radius = max(2, int(4 * alpha))
            # Interpolate colour: faint → vivid yellow
            r = int(TRAIL_BASE[2] * (1 - alpha) + BALL_COLOR[2] * alpha)
            g = int(TRAIL_BASE[1] * (1 - alpha) + BALL_COLOR[1] * alpha)
            b = int(TRAIL_BASE[0] * (1 - alpha) + BALL_COLOR[0] * alpha)
            cv2.circle(out, (int(x), int(y)), radius + 2, (0, 0, 0), -1)
            cv2.circle(out, (int(x), int(y)), radius, (b, g, r), -1)
        return out

    def _draw_ball(
        self,
        frame: np.ndarray,
        ball_px: tuple[float, float] | None,
        conf: float,
    ) -> np.ndarray:
        if not ball_px or conf < 0.15:
            return frame
        x, y = int(ball_px[0]), int(ball_px[1])
        cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
        cv2.circle(frame, (x, y), 8, BALL_COLOR, -1)
        # Glow ring
        cv2.circle(frame, (x, y), 13, (*BALL_COLOR[:2], 120), 1)
        return frame

    def _draw_players(
        self, frame: np.ndarray, players: list[PlayerAnnotation]
    ) -> np.ndarray:
        for p in players:
            color = TEAM_COLORS.get(p.team, (200, 200, 200))
            x1, y1, x2, y2 = int(p.x1), int(p.y1), int(p.x2), int(p.y2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = p.label or f"P{p.track_id}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )

            # Team colour dot under feet
            cx = (x1 + x2) // 2
            cv2.circle(frame, (cx, y2), 5, color, -1)

        return frame

    def _draw_mini_court(
        self, frame: np.ndarray, ann: FrameAnnotation, mini: PadelMiniCourt
    ) -> np.ndarray:
        """Draw padel top-down court in top-right corner."""
        player_dots = []
        for p in ann.players:
            if p.court_x is not None and p.court_y is not None:
                color = TEAM_COLORS.get(p.team, (200, 200, 200))
                player_dots.append((p.court_x, p.court_y, color))

        ball = ann.ball_court if ann.ball_court else None
        return mini.draw_frame(frame, players=player_dots, ball=ball)

    def _draw_speed_meter(self, frame: np.ndarray, speed_kmh: float) -> np.ndarray:
        """Draw ball speed gauge — bottom-left."""
        h, w = frame.shape[:2]
        x0, y0 = 20, h - 20
        meter_w, meter_h = 180, 50

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x0, y0 - meter_h),
            (x0 + meter_w, y0),
            (20, 20, 20), -1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (x0, y0 - meter_h), (x0 + meter_w, y0), ACCENT, 1)

        # Speed bar
        max_speed = 180.0
        fill = min(1.0, speed_kmh / max_speed)
        bar_color = (
            (0, 220, 80) if speed_kmh < 80 else
            (0, 160, 255) if speed_kmh < 130 else
            (0, 80, 255)
        )
        bar_w = int((meter_w - 10) * fill)
        if bar_w > 0:
            cv2.rectangle(
                frame,
                (x0 + 5, y0 - 18),
                (x0 + 5 + bar_w, y0 - 6),
                bar_color, -1,
            )

        # Text
        cv2.putText(
            frame, "BALL SPEED",
            (x0 + 5, y0 - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"{speed_kmh:.0f} km/h",
            (x0 + 5, y0 - 4),
            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return frame

    def _draw_excitement_bar(self, frame: np.ndarray, excitement: float) -> np.ndarray:
        """Draw excitement level bar — bottom center."""
        h, w = frame.shape[:2]
        bar_w = 300
        bar_h = 12
        x0 = (w - bar_w) // 2
        y0 = h - 16

        # Track
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - bar_h - 20), (x0 + bar_w + 4, y0 + 4), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, "EXCITEMENT",
            (x0, y0 - bar_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA,
        )

        fill = min(1.0, max(0.0, excitement / 100.0))
        n_segs = 20
        seg_w = bar_w // n_segs
        filled = int(fill * n_segs)
        for i in range(n_segs):
            sx = x0 + i * seg_w
            if i < filled:
                # Gradient: green → orange → red
                t = i / n_segs
                r = int(min(255, 510 * t))
                g = int(min(255, 510 * (1 - t)))
                b = 40
                cv2.rectangle(frame, (sx + 1, y0 - bar_h), (sx + seg_w - 1, y0), (b, g, r), -1)
            else:
                cv2.rectangle(frame, (sx + 1, y0 - bar_h), (sx + seg_w - 1, y0), (40, 40, 40), -1)

        cv2.putText(
            frame, f"{excitement:.0f}/100",
            (x0 + bar_w + 8, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT, 1, cv2.LINE_AA,
        )
        return frame

    def _draw_rally_banner(
        self,
        frame: np.ndarray,
        ann: FrameAnnotation,
        commentary: str,
    ) -> np.ndarray:
        """Draw rally info banner at top-center."""
        h, w = frame.shape[:2]
        banner_h = 38
        banner_w = min(w - 40, 700)
        x0 = (w - banner_w) // 2
        y0 = 12

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + banner_w, y0 + banner_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + banner_h), ACCENT, 1)

        # Left: rally length
        if ann.rally_length > 0:
            cv2.putText(
                frame, f"RALLY  {ann.rally_length} shots",
                (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Right: state
        if ann.state:
            state_label = ann.state.replace("_", " ").upper()
            tw = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            cv2.putText(
                frame, state_label,
                (x0 + banner_w - tw - 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT, 1, cv2.LINE_AA,
            )

        return frame

    # ── ffmpeg re-encode ───────────────────────────────────────────────────────

    def _reencode(
        self,
        src_video: str,
        annotated_tmp: Path,
        out_path: Path,
        start_frame: int,
        end_frame: int,
    ) -> None:
        """
        Re-encode with ffmpeg: merge annotated video track with original audio.
        If ffmpeg not available, just rename tmp to output.
        """
        try:
            from backend.utils.ffmpeg import get_ffmpeg
            ffmpeg = get_ffmpeg()
        except Exception:
            annotated_tmp.rename(out_path)
            return

        start_t = start_frame / self.fps
        duration = (end_frame - start_frame) / self.fps

        cmd = [
            ffmpeg, "-y",
            "-i", str(annotated_tmp),        # annotated video (no audio)
            "-ss", str(start_t),
            "-t", str(duration),
            "-i", str(src_video),            # original audio
            "-map", "0:v:0",                 # video from annotated
            "-map", "1:a:0?",                # audio from original (optional)
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "21",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("ffmpeg re-encode failed (%s); using raw annotated clip.", exc)
            if annotated_tmp.exists():
                annotated_tmp.rename(out_path)


# ── convenience builder ────────────────────────────────────────────────────────

def build_frame_annotations(
    player_court_positions: dict[int, dict[int, tuple[float, float]]],
    ball_court_positions: dict[int, tuple[float, float]],
    ball_trajectory_px: list[tuple[int, float, float, float]],
    excitement_by_rally: list,           # list[RallySegment]
    fps: float,
) -> dict[int, FrameAnnotation]:
    """
    Build FrameAnnotation dict from orchestrator data structures.
    Called from the orchestrator after pipeline run.
    """
    ann_dict: dict[int, FrameAnnotation] = {}

    # Ball pixel positions indexed by frame
    ball_px_by_frame: dict[int, tuple[float, float, float]] = {
        t[0]: (t[1], t[2], t[3]) for t in ball_trajectory_px
    }

    # Excitement by frame (from rally segments)
    exc_by_frame: dict[int, float] = {}
    rally_len_by_frame: dict[int, int] = {}
    for r in excitement_by_rally:
        exc = getattr(r, "excitement_score", 0.0)
        rlen = getattr(r, "rally_length_shots", 0)
        for f in range(r.start_frame, r.end_frame + 1):
            exc_by_frame[f] = exc
            rally_len_by_frame[f] = rlen

    # Union of all frames
    all_frames = (
        set(player_court_positions.keys())
        | set(ball_court_positions.keys())
        | set(ball_px_by_frame.keys())
    )

    for fidx in sorted(all_frames):
        ball_px_data = ball_px_by_frame.get(fidx)
        ball_px = (ball_px_data[0], ball_px_data[1]) if ball_px_data else None
        ball_conf = ball_px_data[2] if ball_px_data else 0.0
        ball_court = ball_court_positions.get(fidx)

        players = []
        for tid, court_pos in (player_court_positions.get(fidx) or {}).items():
            # We don't store pixel boxes per-frame in the orchestrator currently
            # — FrameAnnotation.players will be populated by court positions only
            players.append(PlayerAnnotation(
                track_id=tid,
                team=tid % 2,          # naive team assignment (0 or 1)
                x1=0, y1=0, x2=0, y2=0,  # pixel boxes not stored per-frame
                court_x=court_pos[0],
                court_y=court_pos[1],
                label=f"P{tid}",
            ))

        ann_dict[fidx] = FrameAnnotation(
            frame_idx=fidx,
            ball_px=ball_px,
            ball_court=ball_court,
            ball_conf=ball_conf,
            players=players,
            excitement=exc_by_frame.get(fidx, 0.0),
            rally_length=rally_len_by_frame.get(fidx, 0),
        )

    return ann_dict
