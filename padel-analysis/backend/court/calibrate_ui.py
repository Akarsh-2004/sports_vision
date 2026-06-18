"""Interactive OpenCV UI for padel court landmark calibration."""

from __future__ import annotations

from pathlib import Path

import cv2

from backend.court.calibration import save_calibration
from backend.court.geometry import CALIBRATION_SEQUENCE, LandmarkId


def _label_for(lid: LandmarkId) -> str:
    return lid.value.replace("_", " ").title()


def run_calibration_ui(video_path: str, frame_index: int = 0) -> dict[str, tuple[float, float]]:
    """
    Click landmarks in order on a representative frame.

    Keys: u = undo, s = save, q = quit without save, n/p = next/prev frame.
  """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = max(0, min(frame_index, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read calibration frame")

    sequence = list(CALIBRATION_SEQUENCE)
    clicks: dict[str, tuple[float, float]] = {}
    step = 0
    win = "Padel Court Calibration"

    def redraw() -> None:
        display = frame.copy()
        for name, (x, y) in clicks.items():
            cv2.circle(display, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(display, name[:12], (int(x) + 8, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if step < len(sequence):
            prompt = f"Click: {_label_for(sequence[step])} ({step + 1}/{len(sequence)})"
        else:
            prompt = "All landmarks set — press S to save"
        cv2.putText(display, prompt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow(win, display)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal step
        if event != cv2.EVENT_LBUTTONDOWN or step >= len(sequence):
            return
        lid = sequence[step]
        clicks[lid.value] = (float(x), float(y))
        step += 1
        redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    saved = False
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and step > 0:
            step -= 1
            del clicks[sequence[step].value]
            redraw()
        if key == ord("s") and len(clicks) >= 4:
            stem = Path(video_path).stem
            save_calibration(stem, clicks)
            saved = True
            break
        if key == ord("n"):
            idx = min(idx + 1, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, new_frame = cap.read()
            if ok:
                frame[:] = new_frame
                redraw()
        if key == ord("p"):
            idx = max(idx - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, new_frame = cap.read()
            if ok:
                frame[:] = new_frame
                redraw()

    cap.release()
    cv2.destroyAllWindows()
    if not saved:
        raise SystemExit("Calibration cancelled")
    return clicks
