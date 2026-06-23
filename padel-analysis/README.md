# Padel Match Analysis

AI-powered padel match analysis from raw video. Tracks four players and the ball, segments points, scores rallies (Team A vs B), generates coach reports, highlight clips, and an interactive dashboard.

Works with **broadcast footage** and **phone recordings** (auto-detected).

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Web UI](#web-ui)
- [Command-line usage](#command-line-usage)
- [Court calibration](#court-calibration)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Point scoring](#point-scoring)
- [Model weights](#model-weights)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)

---

## Prerequisites

Install these **before** running the pipeline:

| Tool | Version | Required for |
|------|---------|--------------|
| [Python](https://www.python.org/downloads/) | **3.11+** | Pipeline, API |
| [FFmpeg](https://ffmpeg.org/download.html) | any recent | Video decode, highlight clips |
| [Node.js](https://nodejs.org/) | **18+** | Web UI only |
| NVIDIA GPU / Apple Silicon | optional | Faster YOLO inference |

### FFmpeg install

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg
```

Verify both:

```bash
python --version    # Python 3.11+
ffmpeg -version
node --version      # 18+ (web UI only)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akarsh-2004/sports_vision.git
cd sports_vision/padel-analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Activate it:**

```bash
# Windows (PowerShell or CMD)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> First run downloads PyTorch and YOLO weights (~1–2 GB). This can take several minutes.

### 4. Set your compute device

Open `configs/padel.yaml` and set `models.device` for your machine:

```yaml
models:
  device: auto    # recommended — picks cuda / mps / cpu automatically
```

| Your machine | Use |
|--------------|-----|
| Windows + NVIDIA GPU | `cuda` |
| Mac (M1/M2/M3/M4) | `mps` |
| No GPU | `cpu` |
| Not sure | `auto` |

> Default in the repo may be `mps`. **Change this on Windows/Linux if you do not have Apple Silicon.**

---

## Quick start

Analyze a match video from the command line:

```bash
python run.py run path/to/match.mp4
```

When finished you will see:

```
Done. Match ID: match_abc12345
Overall score: 72/100
Report saved to data/reports/match_abc12345/
```

Open the report:

```bash
# Windows
start data\reports\<match_id>\report.md

# macOS
open data/reports/<match_id>/report.md
```

For a **fast smoke test** (first 500 frames only):

```bash
python run.py run path/to/match.mp4 --max-frames 500
```

---

## Web UI

A minimal React app lets you upload a video, watch progress, and view scores + coach notes in the browser.

### Development mode (two terminals)

**Terminal 1 — start the API** (from `padel-analysis/`):

```bash
.venv\Scripts\activate          # Windows
python run.py serve
```

API runs at **http://localhost:8001**  
API docs: **http://localhost:8001/docs**

**Terminal 2 — start the frontend** (from `padel-analysis/frontend/`):

```bash
npm install
npm run dev
```

Open **http://localhost:5174** → select a video → click **Analyze**.

The UI shows:
- Pipeline progress bar
- Team A vs B point score
- Performance metrics (overall, net play, wall defense, etc.)
- Per-point breakdown
- Coach report text
- Link to the full `dashboard.html`

### Production mode (single server)

Build the frontend once, then serve everything from the API:

```bash
cd frontend
npm install
npm run build
cd ..
python run.py serve
```

Open **http://localhost:8001**

---

## Command-line usage

```bash
python run.py <command> [options]
```

| Command | Description |
|---------|-------------|
| `run <video>` | Full analysis pipeline |
| `calibrate <video>` | Interactive court landmark picker |
| `serve` | Start FastAPI + web UI server |

### `run` options

```bash
python run.py run match.mp4
python run.py run match.mp4 --max-frames 500
python run.py run match.mp4 --click-x 640 --click-y 400
python run.py run match.mp4 --selection single    # single | pair | all
python run.py run match.mp4 --config configs/padel.yaml

# Shorthand (video as first arg)
python run.py match.mp4
```

| Flag | Purpose |
|------|---------|
| `--max-frames N` | Process only first N frames (quick test) |
| `--click-x`, `--click-y` | Pixel coords to select target player |
| `--selection` | Analyze one player, a pair, or all four |
| `--config` | Path to alternate YAML config |

### `serve` options

```bash
python run.py serve
python run.py serve --host 0.0.0.0 --port 8001
```

---

## Court calibration

If automatic court detection is poor, calibrate manually by clicking 13 landmarks on a video frame:

```bash
python run.py calibrate path/to/match.mp4
```

**Landmarks:** 4 outer corners, 4 service-line intersections, net left/center/right, center line near/far.

**Controls:**

| Key | Action |
|-----|--------|
| Click | Place landmark |
| `u` | Undo last point |
| `s` | Save calibration |
| `n` / `p` | Next / previous frame |

Saved to `data/calibration/<video_stem>_homography.json`.

Enable in config:

```yaml
court:
  use_manual_calibration: true
```

---

## Configuration

All settings live in **`configs/padel.yaml`**.

Key sections:

| Section | What it controls |
|---------|------------------|
| `pipeline` | FPS, resolution, detection stride |
| `models` | YOLO weights, device, ball detector mode |
| `court` | Dimensions, YOLO court model, calibration |
| `rally` | Point segmentation thresholds |
| `highlights` | Clip count, excitement weights |
| `coach_highlights` | Annotated clip overlays |
| `selection` | Which players to analyze |
| `paths` | Data directories |

Example — switch ball detector to heuristic (no custom weights):

```yaml
models:
  ball_detector: heuristic
```

Example — disable intelligence layer for faster runs:

```yaml
intelligence:
  enabled: false
```

---

## Outputs

Each run creates `data/reports/<match_id>/`:

| File / folder | Contents |
|---------------|----------|
| `report.md` | Coach-style tactical summary |
| `match_stats.json` | Performance scores, movement, rallies |
| `full_output.json` | Everything (scores, points, highlights metadata) |
| `dashboard.html` | Interactive charts (open in browser) |
| `player_report.md` | Per-player intelligence report |
| `highlights/` | Top rally clips (`.mp4`) |
| `highlights/annotated/` | Clips with ball trail, mini-court overlay |

Global visualizations (shared across runs): `data/reports/viz/`

---

## Point scoring

The pipeline estimates a **Team A vs Team B** score from video (not official 15-30-40 padel rules).

1. **Segment points** — active-play periods, shot clusters, or ball-based rallies
2. **Detect point end** — ball stops, leaves court, players walk to baseline
3. **Infer winner** — last ball position relative to the net (heuristic)
4. **Update score** — `{"A": 3, "B": 2}` displayed as `3-2`

- **Team A** = top half of court (near side in standard view)
- **Team B** = bottom half

Points marked `interrupted` do not count toward the score.

---

## Model weights

Weights are gitignored (large files). The pipeline runs without them using fallbacks.

| Weight file | Purpose | Fallback |
|-------------|---------|----------|
| `weights/padel_ball_yolo11n.pt` | Ball detection | Heuristic color/motion |
| `weights/padel_court_keypoints.pt` | Court keypoints | Classical line detection |
| `yolo11n.pt` | Player detection | Auto-downloaded by Ultralytics |

### Train custom weights (optional)

```bash
# Ball detector (requires local dataset — edit path in script)
python scripts/train_ball_detector.py

# Court keypoints
python scripts/train_court_keypoints.py

# Court YOLO segmentation
python scripts/train_court_yolo.py
```

### Validate installation

```bash
python scripts/validate_pipeline.py
```

---

## Project structure

```
padel-analysis/
├── run.py                      # CLI entry point
├── configs/padel.yaml          # Configuration
├── requirements.txt
├── backend/
│   ├── api/                    # FastAPI (web UI backend)
│   ├── pipeline/               # Main orchestrator
│   ├── detection/              # Ball & player detectors
│   ├── tracking/               # Ball & player trackers
│   ├── court/                  # Homography & calibration
│   ├── analytics/              # Events, rallies, scoring
│   ├── highlights/             # Clip extraction
│   ├── intelligence/           # World model, coach engine
│   └── visualization/          # Charts, annotated export
├── frontend/                   # React web UI
├── scripts/                    # Training & dashboard tools
├── tests/
└── data/
    ├── raw/                    # Uploaded / input videos
    ├── processed/              # Normalized video frames
    └── reports/                # Analysis outputs
```

---

## Troubleshooting

### `ffmpeg not found`

Install FFmpeg and restart your terminal. The pipeline cannot decode video without it.

### `CUDA out of memory` / `MPS error`

Set `models.device: cpu` in `configs/padel.yaml`, or reduce resolution:

```yaml
pipeline:
  target_width: 960
  target_height: 540
```

### Pipeline is very slow

- Use a GPU (`cuda` or `mps`)
- Test with `--max-frames 500` first
- Increase `pipeline.detection_stride` (e.g. `6` or `8`)

### Ball YOLO weights not found

Either train weights (see above) or switch to heuristic mode:

```yaml
models:
  ball_detector: heuristic
```

### Web UI cannot reach API

- Ensure `python run.py serve` is running on port **8001**
- Frontend dev server proxies `/api` → `localhost:8001` (see `frontend/vite.config.js`)
- Check firewall is not blocking local ports

### Upload fails in web UI

Default max upload is 2048 MB (`api.max_upload_mb` in config). Large videos may take a long time to process on CPU.

### No points detected

Common with very short clips or poor ball visibility. Try:
- Longer video with clear rallies
- Lower `rally.min_active_point_s` in config
- Train or add ball YOLO weights

---

## Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
Video → Ingest (FFmpeg) → Court detection → Player tracking (YOLO + BoT-SORT)
  → Ball tracking → Pose → Strokes → Wall events → Movement analytics
  → Point segmentation → Score tracking → Highlights → Coach report → Dashboard
```

Court geometry uses a **10 m × 20 m** homography — all positions are projected into real court meters.

---

## Related

- [Root README](../README.md) — monorepo overview
- [Tennis analysis](../tennis-analysis/README.md) — sister pipeline
- [context.md](context.md) — developer & agent reference
