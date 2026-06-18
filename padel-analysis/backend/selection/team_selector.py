"""Player / team selection for doubles padel."""

from __future__ import annotations

import cv2
import numpy as np

from backend.utils.types import PlayerDetection, PlayerSelectionMode


class TeamSelector:
    """Select one player, a pair, or track all four."""

    def __init__(self, mode: str = "single"):
        self.mode = PlayerSelectionMode(mode)
        self._jersey_hist: dict[int, np.ndarray] = {}
        self._target_ids: set[int] = set()

    def select_by_click(
        self,
        click_xy: tuple[float, float],
        players: list[PlayerDetection],
        frame: np.ndarray,
    ) -> int | None:
        if not players:
            return None
        cx, cy = click_xy
        best = min(
            players,
            key=lambda p: float(np.hypot(p.bbox.centroid[0] - cx, p.bbox.centroid[1] - cy)),
        )
        self._store_jersey(best.track_id, frame, best.bbox)
        self._target_ids = {best.track_id}
        if self.mode == PlayerSelectionMode.PAIR:
            partner = self._nearest_partner(best, players)
            if partner:
                self._target_ids.add(partner.track_id)
        elif self.mode == PlayerSelectionMode.ALL:
            self._target_ids = {p.track_id for p in players}
        return best.track_id

    def select_auto(self, players: list[PlayerDetection], court_length: float) -> int | None:
        """Pick near-side player with largest court Y (closest to camera baseline)."""
        with_court = [p for p in players if p.court_xy]
        if not with_court:
            return players[0].track_id if players else None
        best = min(with_court, key=lambda p: p.court_xy[1])  # type: ignore[index]
        self._target_ids = {best.track_id}
        if self.mode == PlayerSelectionMode.PAIR:
            partner = self._nearest_partner(best, with_court)
            if partner:
                self._target_ids.add(partner.track_id)
        elif self.mode == PlayerSelectionMode.ALL:
            self._target_ids = {p.track_id for p in players}
        return best.track_id

    def confirm_track(self, players: list[PlayerDetection], frame: np.ndarray) -> int | None:
        if not self._target_ids or not players:
            return None
        primary = min(self._target_ids)
        ids = {p.track_id for p in players}
        if primary in ids:
            return primary
        if primary not in self._jersey_hist:
            return players[0].track_id
        ref = self._jersey_hist[primary]
        best_id = primary
        best_dist = float("inf")
        for p in players:
            hist = self._jersey_hist.get(p.track_id) or self._extract_jersey(frame, p.bbox)
            d = float(cv2.compareHist(ref, hist, cv2.HISTCMP_BHATTACHARYYA))
            if d < best_dist:
                best_dist = d
                best_id = p.track_id
        self._target_ids = {best_id}
        return best_id

    def target_ids(self) -> set[int]:
        return set(self._target_ids)

    def assign_teams(self, players: list[PlayerDetection], court_width: float) -> None:
        """Split players into team 0 (left half) and team 1 (right half) by average X."""
        if len(players) < 2:
            return
        for p in players:
            if p.court_xy:
                p.team_id = 0 if p.court_xy[0] < court_width / 2 else 1

    def _nearest_partner(
        self, anchor: PlayerDetection, players: list[PlayerDetection]
    ) -> PlayerDetection | None:
        others = [p for p in players if p.track_id != anchor.track_id and p.court_xy and anchor.court_xy]
        if not others:
            return None
        ax, ay = anchor.court_xy  # type: ignore[misc]
        return min(others, key=lambda p: float(np.hypot(p.court_xy[0] - ax, p.court_xy[1] - ay)))  # type: ignore[index]

    def _store_jersey(self, track_id: int, frame: np.ndarray, bbox) -> None:
        self._jersey_hist[track_id] = self._extract_jersey(frame, bbox)

    def _extract_jersey(self, frame: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        h = max(1, y2 - y1)
        patch = frame[y1 : y1 + h // 3, x1:x2]
        if patch.size == 0:
            return np.zeros((16, 1, 3), dtype=np.float32)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
