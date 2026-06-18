# Tennis Analysis — Environment Setup

## Quick setup (Windows)

```powershell
cd tennis-analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-min.txt
pip install imageio-ffmpeg
```

Or run: `scripts\setup_env.bat`

## FFmpeg

The project uses **imageio-ffmpeg** (bundled ~30MB binary via pip) — no separate install needed:

```powershell
pip install imageio-ffmpeg
python -c "from backend.utils.ffmpeg import get_ffmpeg; print(get_ffmpeg())"
```

Optional system install (needs ~250MB free disk):

```powershell
winget install Gyan.FFmpeg
```

## Generate highlights (first 1 minute)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/make_highlights.py ^
  --video "..\videoplayback.webm" ^
  --stats "data\reports\videoplayback_bdf89c84\match_stats.json" ^
  --max-seconds 60
```

Outputs:
- `data/reports/<match_id>/highlights/first_minute_reel.mp4`
- `highlight_00.mp4`, `highlight_01.mp4` (top rallies in that window)

## Disk space note

Full `requirements.txt` includes PyTorch (~2GB). Use `requirements-min.txt` if disk is limited; YOLO/ultralytics needs torch installed separately when you run the full pipeline.
