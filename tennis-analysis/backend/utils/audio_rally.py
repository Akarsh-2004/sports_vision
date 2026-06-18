"""Audio transient detection for rally boundary hints."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.utils.logging import get_logger

logger = get_logger(__name__)


def detect_hit_frames(video_path: str, fps: float, threshold_percentile: float = 92.0) -> list[int]:
    """Return frame indices where sharp audio transients (ball hits) likely occur."""
    try:
        import subprocess
        import tempfile

        from backend.utils.ffmpeg import get_ffmpeg

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav = tmp.name
        subprocess.run(
            [get_ffmpeg(), "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "8000", wav],
            check=True,
            capture_output=True,
        )
        import wave

        with wave.open(wav, "rb") as wf:
            rate = wf.getframerate()
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
        Path(wav).unlink(missing_ok=True)
        if len(raw) < rate:
            return []

        win = int(rate * 0.02)
        hop = win // 2
        energy = []
        for i in range(0, len(raw) - win, hop):
            chunk = raw[i : i + win]
            energy.append(float(np.sum(chunk * chunk) / win))
        energy = np.array(energy)
        if len(energy) < 3:
            return []
        diff = np.abs(np.diff(energy, prepend=energy[0]))
        thresh = np.percentile(diff, threshold_percentile)
        hit_times = np.where(diff >= thresh)[0] * hop / rate
        return [int(t * fps) for t in hit_times]
    except Exception as exc:
        logger.warning("Audio rally hints unavailable: %s", exc)
        return []
