"""Debounce shot detections — one contact per swing, not per frame."""

from __future__ import annotations


class ShotDebouncer:
    """Padel swings last ~0.4–0.7s; reject duplicate hits in that window."""

    def __init__(self, fps: float, min_gap_s: float = 0.55, global_gap_s: float = 0.22):
        self.min_gap_frames = max(1, int(min_gap_s * fps))
        self.global_gap_frames = max(1, int(global_gap_s * fps))
        self._last_by_player: dict[int, int] = {}
        self._last_any = -10_000

    def allow(self, player_id: int, frame_idx: int) -> bool:
        last_player = self._last_by_player.get(player_id, -10_000)
        if frame_idx - last_player < self.min_gap_frames:
            return False
        if frame_idx - self._last_any < self.global_gap_frames:
            return False
        self._last_by_player[player_id] = frame_idx
        self._last_any = frame_idx
        return True

    def dedupe_frames(self, frames: list[int], fps: float | None = None) -> list[int]:
        """Collapse a legacy shot frame list to one frame per burst."""
        if not frames:
            return []
        gap = self.min_gap_frames if fps is None else max(1, int(0.55 * fps))
        sorted_f = sorted(frames)
        out = [sorted_f[0]]
        for f in sorted_f[1:]:
            if f - out[-1] >= gap:
                out.append(f)
        return out
