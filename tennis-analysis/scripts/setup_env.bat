@echo off
REM Setup virtual environment for tennis-analysis (lightweight, no torch)
cd /d "%~dp0.."
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-min.txt
pip install ultralytics --no-deps 2>nul
pip install torch --index-url https://download.pytorch.org/whl/cpu 2>nul
echo.
echo Done. Activate with:  .venv\Scripts\activate
echo FFmpeg via:  python -c "from backend.utils.ffmpeg import get_ffmpeg; print(get_ffmpeg())"
