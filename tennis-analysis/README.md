# Tennis Match Analysis System

AI-powered computer vision pipeline for deep tennis player performance analysis from raw match footage. Implements all **17 pipeline stages** from the engineering design document.

## Architecture

```
Video → Ingestion → Court → Player/Ball Detection → Tracking → Target Selection
  → Analytics (strokes, pose, movement, quality, points, events)
  → Output (highlights, scoring, NL report, visualizations)
```

## Quick start

### Prerequisites

- Python 3.11+
- FFmpeg (recommended)
- Optional: NVIDIA GPU + CUDA for YOLO acceleration
- Optional: [Ollama](https://ollama.ai) for LLM reports

### Install

```bash
cd tennis-analysis
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run CLI analysis

```bash
python run.py path/to/match.mp4
python run.py path/to/match.mp4 --click-x 640 --click-y 400 --max-frames 500
```

### Run API server

```bash
python run.py --serve
# Open http://localhost:8000/docs for OpenAPI
```

### Frontend dashboard

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

### Docker

```bash
docker compose up --build
```

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `backend/ingestion/` | FFmpeg decode, FPS normalize, stabilization, court gate |
| 2 | `backend/court/` | Hough lines, homography, CourtState |
| 3 | `backend/detection/` | YOLOv11 player detection |
| 4 | `backend/tracking/` | BoT-SORT player tracking |
| 5 | `backend/selection/` | Click-to-select target player |
| 6 | `backend/detection/` | Ball detection (TrackNet/heuristic) |
| 7 | `backend/tracking/` | Kalman ball tracker |
| 8 | `backend/analytics/strokes/` | Pose-based stroke classifier |
| 9 | `backend/analytics/pose/` | MediaPipe pose estimation |
| 10 | `backend/analytics/movement/` | Distance, speed, heatmaps |
| 11 | `backend/analytics/quality/` | Shot power, placement, aggression |
| 12 | `backend/analytics/events/` | Point/rally segmentation |
| 13 | `backend/analytics/events/` | Winner, error, ace detection |
| 14 | `backend/highlights/` | FFmpeg highlight clips |
| 15 | `backend/scoring/` | 8-dimension performance scores |
| 16 | `backend/summarization/` | Template/Ollama NL report |
| 17 | `backend/visualization/` | Heatmap, radar, shot chart, timeline |

## Configuration

Edit `configs/default.yaml` for FPS, models, thresholds, and paths.

## Outputs

Results are written to `data/reports/<match_id>/`:

- `match_stats.json` — structured analytics
- `report.md` — natural language summary
- `full_output.json` — complete pipeline output
- `data/reports/viz/` — PNG visualizations
- `data/reports/highlights/` — top rally clips

## Tests

```bash
pytest tests/ -v
```

## MVP vs production

This implementation follows **Phase 1–3** of the design doc:

- **MVP**: YOLOv11n, BoT-SORT, MediaPipe, heuristic ball detection, rule-based strokes
- **Upgrade path**: TrackNet v3, YOLOv11x, ViTPose-B, VideoMAE, TensorRT, Ollama LLM

Set `models.ball_detector: tracknet` and `summarization.provider: ollama` when weights/services are available.

## License

Open source — see design document for dataset and model attribution requirements.
