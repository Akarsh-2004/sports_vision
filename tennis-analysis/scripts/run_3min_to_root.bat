@echo off
REM Analyze first 3 minutes and save everything to ..\output_3min\
cd /d "%~dp0.."
set ROOT=%~dp0..\output_3min
set VIDEO=%~dp0..\..\videoplayback.webm
mkdir "%ROOT%" 2>nul

echo === Step 1: Pipeline (3 min / 4500 frames) ===
python run.py "%VIDEO%" --max-frames 4500 2>&1 | tee "%ROOT%\pipeline.log"
if errorlevel 1 exit /b 1

echo === Step 2: Find latest report ===
for /f "delims=" %%i in ('dir /b /od data\reports\videoplayback_* 2^>nul') do set MATCH=%%i
echo Match folder: %MATCH%

echo === Step 3: Highlights ===
.\.venv\Scripts\python.exe scripts\make_highlights.py ^
  --video "%VIDEO%" ^
  --stats "data\reports\%MATCH%\match_stats.json" ^
  --max-seconds 180 ^
  --out-dir "%ROOT%\highlights"

echo === Step 4: Copy reports to root output ===
copy /Y "data\reports\%MATCH%\match_stats.json" "%ROOT%\"
copy /Y "data\reports\%MATCH%\report.md" "%ROOT%\"
copy /Y "data\reports\%MATCH%\full_output.json" "%ROOT%\"
xcopy /Y /E "data\reports\viz\*" "%ROOT%\viz\" 2>nul

echo Done. Open folder: %ROOT%
pause
