"""Tests for padel court calibration and geometry."""

from __future__ import annotations

import numpy as np

from backend.court.calibration import compute_homography, save_calibration, load_calibration
from backend.court.geometry import COURT_LANDMARKS, LandmarkId, COURT_WIDTH_M, COURT_LENGTH_M


def test_canonical_court_dimensions():
    from backend.court.geometry import CANONICAL_COURT

    assert CANONICAL_COURT.shape == (4, 2)
    assert float(CANONICAL_COURT[1, 0]) == COURT_WIDTH_M
    assert float(CANONICAL_COURT[2, 1]) == COURT_LENGTH_M


def test_landmark_count():
    assert len(COURT_LANDMARKS) >= 10


def test_homography_identity_corners(tmp_path, monkeypatch):
    import backend.court.calibration as cal_mod

    monkeypatch.setattr(cal_mod, "_CALIB_DIR", tmp_path)

    # Synthetic top-down: pixel coords == court meters scaled × 50
    scale = 50.0
    landmarks = {
        lid.value: (COURT_LANDMARKS[lid][0] * scale, COURT_LANDMARKS[lid][1] * scale)
        for lid in (
            LandmarkId.NEAR_LEFT,
            LandmarkId.NEAR_RIGHT,
            LandmarkId.FAR_LEFT,
            LandmarkId.FAR_RIGHT,
        )
    }
    H, conf = compute_homography(landmarks)
    assert H is not None
    assert conf > 0.5

    save_calibration("test_match", landmarks)
    loaded = load_calibration("test_match")
    assert loaded is not None
    assert loaded.shape == (3, 3)

    pt = np.array([[[5.0 * scale, 10.0 * scale]]], dtype=np.float32)
    import cv2

    mapped = cv2.perspectiveTransform(pt, loaded)
    assert abs(mapped[0, 0, 0] - 5.0) < 0.2
    assert abs(mapped[0, 0, 1] - 10.0) < 0.2
