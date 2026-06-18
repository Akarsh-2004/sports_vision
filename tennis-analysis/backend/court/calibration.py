"""Manual court corner calibration for partial-court phone footage."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.court.geometry import CANONICAL_COURT

_CALIB_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration"


def calibration_path(video_stem: str) -> Path:
    _CALIB_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_stem)
    return _CALIB_DIR / f"{safe}_homography.json"


def save_calibration(video_stem: str, corners_px: list[tuple[float, float]]) -> np.ndarray:
    """corners_px: near-left, near-right, far-left, far-right in image coordinates."""
    src = np.float32(corners_px)
    H, _ = cv2.findHomography(src, CANONICAL_COURT, cv2.RANSAC, 5.0)
    data = {"corners": corners_px, "homography": H.tolist()}
    calibration_path(video_stem).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return H


def load_calibration(video_stem: str) -> np.ndarray | None:
    path = calibration_path(video_stem)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["homography"], dtype=np.float64)


def calibrate_from_clicks(video_path: str, corners: list[tuple[float, float]]) -> np.ndarray:
    stem = Path(video_path).stem
    return save_calibration(stem, corners)
