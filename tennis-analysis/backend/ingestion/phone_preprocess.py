"""Phone/courtside footage preprocessing (CLAHE + denoise)."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_phone_frame(frame: np.ndarray, denoise: bool = True) -> np.ndarray:
    """Enhance contrast and reduce noise for phone-recorded tennis footage."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 6, 6, 7, 21)
    return enhanced


def is_phone_footage(width: int, height: int) -> bool:
    """Heuristic: portrait-ish or sub-1080p courtside phone capture."""
    return height < 900 or (width / max(height, 1)) < 1.6
