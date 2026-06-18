from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import CameraAngle, FrameMeta

logger = get_logger(__name__)


class VideoIngestion:
    """Stage 1: decode, normalize FPS/resolution, optional stabilize, court tagging."""

    def __init__(self, config: dict):
        self.cfg = config["pipeline"]
        self.target_fps = self.cfg["target_fps"]
        self.target_w = self.cfg["target_width"]
        self.target_h = self.cfg["target_height"]
        self.stabilize = self.cfg.get("stabilize", True)
        self.court_gate = self.cfg.get("court_gate", True)

    def ingest(self, video_path: str | Path) -> tuple[str, list[FrameMeta]]:
        video_path = Path(video_path)
        out_dir = Path(self.cfg.get("output_dir", "data/processed"))
        out_dir.mkdir(parents=True, exist_ok=True)
        normalized = out_dir / f"{video_path.stem}_normalized.mp4"

        playable = str(video_path)
        if self._ffmpeg_available():
            self._normalize_with_ffmpeg(video_path, normalized)
            if self._verify_video(normalized):
                playable = str(normalized)
            else:
                logger.warning("FFmpeg output unreadable; using source video directly")
        else:
            logger.info("FFmpeg unavailable; processing source video directly (on-the-fly resize)")

        meta = self._sample_metadata(playable)
        if self.court_gate:
            usable = [m for m in meta if m.court_present or m.camera_angle == CameraAngle.CLOSE_UP]
            if usable and len(usable) >= len(meta) * 0.05:
                meta = usable
            elif self.court_gate and not usable:
                logger.warning("Court gate matched no frames; processing all frames anyway")
        logger.info("Ingested %s: %d frames (%d tagged usable)", video_path.name, self._count_output_frames(playable), len(meta))
        return playable, meta

    def _ffmpeg_available(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _verify_video(self, path: Path) -> bool:
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return bool(ok)

    def _count_output_frames(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        step = max(1, int(round(src_fps / self.target_fps)))
        return total // step if total > 0 else 0

    def _sample_metadata(self, video_path: str, sample_every: int = 50) -> list[FrameMeta]:
        """Lightweight metadata from sparse frame samples (no full-video stabilize pass)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        step = max(1, int(round(src_fps / self.target_fps)))
        meta: list[FrameMeta] = []
        raw_idx = 0
        out_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % step == 0:
                if out_idx % sample_every == 0:
                    frame = self._resize_pad(frame)
                    h, w = frame.shape[:2]
                    meta.append(
                        FrameMeta(
                            frame_idx=out_idx,
                            timestamp_s=out_idx / self.target_fps,
                            width=w,
                            height=h,
                            camera_angle=self._classify_camera_angle(frame),
                            court_present=self._detect_court_presence(frame),
                            quality_score=self._frame_quality(frame),
                        )
                    )
                out_idx += 1
            raw_idx += 1
        cap.release()
        return meta

    def _normalize_with_ffmpeg(self, src: Path, dst: Path) -> None:
        vf = (
            f"fps={self.target_fps},scale={self.target_w}:{self.target_h}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={self.target_w}:{self.target_h}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an", str(dst),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def frame_generator(self, video_path: str | Path) -> Generator[tuple[int, np.ndarray], None, None]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        step = max(1, int(round(src_fps / self.target_fps)))
        prev_gray: np.ndarray | None = None
        raw_idx = 0
        out_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % step == 0:
                frame = self._resize_pad(frame)
                if self.stabilize and prev_gray is not None:
                    frame = self._stabilize_frame(frame, prev_gray)
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield out_idx, frame
                out_idx += 1
            raw_idx += 1
        cap.release()

    def _resize_pad(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        y0 = (self.target_h - nh) // 2
        x0 = (self.target_w - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas

    def _stabilize_frame(self, frame: np.ndarray, prev_gray: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return frame
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            return frame
        matches = sorted(matches, key=lambda m: m.distance)[:50]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
        if M is None:
            return frame
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    def _detect_court_presence(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad range for hard courts (blue/green) and clay
        masks = [
            cv2.inRange(hsv, np.array([35, 30, 30]), np.array([90, 255, 255])),
            cv2.inRange(hsv, np.array([90, 30, 30]), np.array([130, 255, 255])),
        ]
        court_ratio = max(m.sum() for m in masks) / (masks[0].size * 255)
        return court_ratio > 0.08

    def _classify_camera_angle(self, frame: np.ndarray) -> CameraAngle:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        row_sums = green.sum(axis=1) / 255
        peak_row = int(np.argmax(row_sums))
        band_width = (row_sums > row_sums.max() * 0.5).sum()
        if band_width > h * 0.35 and 0.25 * h < peak_row < 0.75 * h:
            return CameraAngle.BASELINE
        if band_width < h * 0.15:
            return CameraAngle.CLOSE_UP
        if peak_row < 0.3 * h:
            return CameraAngle.OVERHEAD
        return CameraAngle.SIDE_ON

    def _frame_quality(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(1.0, lap_var / 500.0))
