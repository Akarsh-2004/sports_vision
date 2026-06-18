from __future__ import annotations

import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import BBox, PlayerDetection

logger = get_logger(__name__)


class PlayerTracker:
    """Stage 4: BoT-SORT via Ultralytics or IoU-based fallback."""

    def __init__(self, config: dict):
        self.cfg = config["models"]
        self.tracker_name = self.cfg.get("player_tracker", "botsort.yaml")
        self.model = None
        self._iou_tracker: dict[int, tuple[float, float]] = {}
        self._next_id = 1
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.cfg.get("player_detector", "yolo11n.pt"))
            logger.info("Player tracker using Ultralytics track mode: %s", self.tracker_name)
        except Exception as exc:
            logger.warning("Ultralytics tracker unavailable: %s", exc)

    def update(self, frame: np.ndarray, detections: list[PlayerDetection] | None = None) -> list[PlayerDetection]:
        if self.model is not None:
            return self._track_ultralytics(frame)
        return self._track_iou(detections or [])

    def _track_ultralytics(self, frame: np.ndarray) -> list[PlayerDetection]:
        results = self.model.track(
            frame,
            classes=[0],
            persist=True,
            tracker=self.tracker_name,
            conf=self.cfg.get("person_conf", 0.35),
            verbose=False,
        )
        tracked: list[PlayerDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                tid = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tracked.append(
                    PlayerDetection(
                        track_id=tid,
                        bbox=BBox(x1, y1, x2, y2, float(box.conf[0])),
                    )
                )
        return tracked

    def _track_iou(self, detections: list[PlayerDetection]) -> list[PlayerDetection]:
        if not detections:
            return []
        assigned: list[PlayerDetection] = []
        used_ids: set[int] = set()
        for det in detections:
            cx, cy = det.bbox.centroid
            best_id, best_iou = None, 0.0
            for tid, (px, py) in self._iou_tracker.items():
                if tid in used_ids:
                    continue
                dist = np.hypot(cx - px, cy - py)
                if dist < 80:
                    iou = 1.0 / (1.0 + dist / 50)
                    if iou > best_iou:
                        best_iou, best_id = iou, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used_ids.add(best_id)
            self._iou_tracker[best_id] = det.bbox.centroid
            assigned.append(
                PlayerDetection(track_id=best_id, bbox=det.bbox, court_xy=det.court_xy)
            )
        return assigned
