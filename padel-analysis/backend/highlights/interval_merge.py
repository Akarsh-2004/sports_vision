"""Merge overlapping highlight intervals (Spintip-style dedup)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HighlightInterval:
    start_frame: int
    end_frame: int
    excitement: float
    label: str = ""


def merge_overlapping_intervals(
    intervals: list[HighlightInterval],
    fps: float,
    min_overlap_s: float = 2.0,
) -> list[HighlightInterval]:
    """Merge intervals where overlap exceeds min_overlap_s; keep max excitement."""
    if not intervals:
        return []
    min_overlap_f = int(min_overlap_s * fps)
    sorted_iv = sorted(intervals, key=lambda x: x.start_frame)
    merged: list[HighlightInterval] = [sorted_iv[0]]
    for iv in sorted_iv[1:]:
        prev = merged[-1]
        overlap = prev.end_frame - iv.start_frame
        if overlap > min_overlap_f:
            prev.end_frame = max(prev.end_frame, iv.end_frame)
            if iv.excitement > prev.excitement:
                prev.excitement = iv.excitement
                prev.label = iv.label or prev.label
        else:
            merged.append(iv)
    return merged
