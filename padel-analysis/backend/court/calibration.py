"""Manual padel court calibration with 10–14 landmarks and homography."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.court.geometry import (
    CALIBRATION_SEQUENCE,
    COURT_LANDMARKS,
    CANONICAL_COURT,
    HOMOGRAPHY_CORNER_IDS,
    LandmarkId,
)

_CALIB_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration"


def calibration_path(video_stem: str) -> Path:
    _CALIB_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_stem)
    return _CALIB_DIR / f"{safe}_homography.json"


def _landmarks_to_arrays(
    landmarks_px: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build src/dst point arrays from labeled landmarks."""
    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for key, px in landmarks_px.items():
        try:
            lid = LandmarkId(key)
        except ValueError:
            continue
        if lid not in COURT_LANDMARKS:
            continue
        src_pts.append([px[0], px[1]])
        dst_pts.append(list(COURT_LANDMARKS[lid]))
    return np.float32(src_pts), np.float32(dst_pts)


def compute_homography(
    landmarks_px: dict[str, tuple[float, float]],
) -> tuple[np.ndarray | None, float]:
    """
    Compute homography from pixel landmarks to court meters.

    Uses all provided landmarks when >= 4; corners alone define the primary transform.
    Returns (H, reprojection_rmse_m).
    """
    corner_src = []
    for lid in HOMOGRAPHY_CORNER_IDS:
        key = lid.value
        if key not in landmarks_px:
            return None, 0.0
        corner_src.append(landmarks_px[key])

    H, mask = cv2.findHomography(np.float32(corner_src), CANONICAL_COURT, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0.0

    src, dst = _landmarks_to_arrays(landmarks_px)
    if len(src) < 4:
        conf = float(mask.sum() / 4.0) if mask is not None else 0.5
        return H, conf

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    # Lower RMSE → higher confidence (pad at ~0.5 m RMSE).
    conf = float(max(0.0, min(1.0, 1.0 - rmse / 0.5)))
    return H, conf


def save_calibration(
    video_stem: str,
    landmarks_px: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Persist landmarks + homography for a video stem."""
    H, confidence = compute_homography(landmarks_px)
    if H is None:
        raise ValueError("Need at least four corner landmarks for homography")

    corners = [landmarks_px[lid.value] for lid in HOMOGRAPHY_CORNER_IDS]
    data = {
        "sport": "padel",
        "corners": corners,
        "landmarks": {k: list(v) for k, v in landmarks_px.items()},
        "homography": H.tolist(),
        "confidence": confidence,
        "court_width_m": 10.0,
        "court_length_m": 20.0,
    }
    calibration_path(video_stem).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return H


def load_calibration(video_stem: str) -> np.ndarray | None:
    path = calibration_path(video_stem)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["homography"], dtype=np.float64)


def load_calibration_data(video_stem: str) -> dict | None:
    path = calibration_path(video_stem)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def calibrate_from_clicks(
    video_path: str,
    landmarks_px: dict[str, tuple[float, float]],
) -> np.ndarray:
    stem = Path(video_path).stem
    return save_calibration(stem, landmarks_px)


def calibrate_corners_only(
    video_stem: str,
    corners_px: list[tuple[float, float]],
) -> np.ndarray:
    """Backward-compatible 4-corner calibration."""
    keys = [lid.value for lid in HOMOGRAPHY_CORNER_IDS]
    landmarks = dict(zip(keys, corners_px))
    return save_calibration(video_stem, landmarks)


def default_calibration_sequence() -> list[LandmarkId]:
    return list(CALIBRATION_SEQUENCE)
