# Padel Match Analysis Pipeline

AI-powered padel match analysis alongside the existing tennis pipeline in `tennis-analysis/`.

## Architecture

```
Video → Frame Extraction → Court Calibration (OpenCV homography)
  → Player Detection/Tracking (4 players) + Ball Tracking
  → Pose → Stroke Recognition → Ball Contact → Wall Events
  → Tactical Engine → Rally Segmentation → Highlights → LLM Report → Dashboard
```

OpenCV homography is the geometric backbone: every player, ball bounce, and event is projected into a **10 m × 20 m** court coordinate system.

## Quick Start

```bash
cd padel-analysis
pip install -r requirements.txt

# Phase 1: calibrate court (click 13 landmarks)
python run.py calibrate path/to/match.mp4

# Run full pipeline
python run.py run path/to/match.mp4
python run.py run match.mp4 --click-x 640 --click-y 400 --selection single
python run.py path/to/match.mp4   # shorthand (same as run)
```

## Pipeline Phases

| Phase | Module | Status |
|-------|--------|--------|
| 1 | `backend/court/` — geometry, calibration, homography | **Implemented** |
| 2 | `backend/detection/player_detector.py` | Reused (YOLO11) |
| 3 | `backend/tracking/player_tracker.py` | Reused (BoT-SORT) |
| 4 | `backend/detection/ball_detector.py` | Reused (heuristic) |
| 5 | `backend/tracking/ball_tracker.py` | Reused (Kalman) |
| 6 | `backend/analytics/events/wall_detector.py` | MVP heuristics |
| 7 | `backend/analytics/pose/` | Reused (MediaPipe) |
| 8 | `backend/analytics/strokes/` | Padel stroke taxonomy |
| 9 | `backend/analytics/contact/` | Racket-ball contact |
| 10 | `backend/analytics/movement/` | Net/defensive zones |
| 11 | `backend/analytics/quality/` | Shot placement |
| 12 | `backend/analytics/tactical/` | Net dominance, spacing |
| 13 | `backend/analytics/events/` | Wall winners, net errors |
| 14 | `backend/selection/team_selector.py` | Single / pair / all |
| 15 | `backend/scoring/` | Padel performance radar |
| 16 | `backend/summarization/` | Coach-level report |
| 17 | `backend/highlights/` | Rally clips |
| 18 | `backend/visualization/` | Heatmaps, shot chart |

## Calibration (Phase 1)

Manual calibration collects **13 landmarks**:

- 4 outer corners
- 4 service-line intersections
- Net left, net right, net center
- Center line near / far

Saved to `data/calibration/{video_stem}_homography.json`. Keys: `u` undo, `s` save, `n`/`p` change frame.

Future: YOLOv11 / RT-DETR landmark model via Roboflow keypoint detection.

## Configuration

`configs/padel.yaml` — court dimensions, 4-player cap, rally thresholds, highlight weights, LLM provider.

## Output

Per run in `data/reports/<match_id>/`:

- `match_stats.json` — structured metrics
- `report.md` — tactical summary
- `full_output.json` — complete bundle
- `data/reports/viz/` — heatmap, radar, shot chart, timeline
- `data/reports/highlights/` — top rally clips

## Tests

```bash
pytest tests/ -v
```

## Next Steps

1. Fine-tune YOLO on padel broadcast footage (4 players, glass walls)
2. Train Roboflow landmark model for auto-calibration
3. Replace heuristic ball detector with YOLO + optical flow recovery
4. Integrate VideoMAE / SlowFast for stroke classification
5. React dashboard (fork `tennis-analysis/frontend` with padel court SVG)
