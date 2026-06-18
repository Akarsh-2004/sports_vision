"""Resolve ffmpeg executable (PATH, imageio-ffmpeg bundle, or winget install)."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = list(Path(local).glob("Microsoft/WinGet/Packages/*FFmpeg*/**/ffmpeg.exe"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        "ffmpeg not found. Install via: pip install imageio-ffmpeg  OR  winget install Gyan.FFmpeg"
    )


def run_ffmpeg(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    cmd = [get_ffmpeg(), *args]
    return subprocess.run(cmd, check=True, capture_output=True, **kwargs)
