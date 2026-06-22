# Padel Analysis Project Context

This file serves as a reference for developers and agents to understand the repository structure, fine-tuned models, execution commands, and future plans.

---

## 🎾 Project Overview
This repository implements a **Padel Intelligence Engine** designed to analyze match videos. It tracks players and the ball, detects court geometry, segments rallies, and generates broadcast-style annotated video highlights with overlays.

---

## ⚙️ Environment Setup
- **Directory:** `/Users/akarshsaklani/Desktop/krateasy_vision/sports_vision/padel-analysis`
- **Virtual Environment:** `.venv/` (Python 3.14.3)
- **Activate Env Command:**
  ```bash
  source .venv/bin/activate
  ```
- **Accelerated Device:** Apple Silicon M4 GPU (`mps`) is used for model training and pipeline execution.

---

## 🧠 Fine-Tuned Models
Both models are stored in the `weights/` directory:
1. **Padel Ball Detector:** Fine-tuned YOLOv11n model using the 8,229-image Padel Ball Detection v4 dataset.
   * Path: `weights/padel_ball_yolo11n.pt`
2. **Court Keypoints Detector:** Fine-tuned YOLOv8-pose model (using COCO-pose base) on Padel.v1i.yolov8.
   * Path: `weights/padel_court_keypoints.pt`

---

## 🚀 Common Commands

### 1. Run Pipeline Analysis
Runs detection, tracking, rally segmentation, and highlight generation on a video file.
```bash
python3 run.py run "/Users/akarshsaklani/Desktop/krateasy_vision/Screen Recording 2026-06-19 at 4.34.59 PM.mov"
```

### 2. Verify Pipeline Integrity
Smoke tests individual engine components to check imports and basic functionality.
```bash
python3 scripts/validate_pipeline.py
```

### 3. Fine-Tune Ball Detector
```bash
python3 scripts/train_ball_detector.py
```

### 4. Fine-Tune Court Keypoints Detector
```bash
python3 scripts/train_court_keypoints.py
```

---

## 📁 Directory Structure
* `configs/padel.yaml` — Central configuration (device mappings, thresholds, and options).
* `backend/`
  * `court/` — Court detection and line keypoint mappings.
  * `detection/` — YOLO ball and player detectors.
  * `tracking/` — Kalman filter tracking and racket-contact physics.
  * `highlights/` — AI Coach highlight scorer and highlight manifest generator.
  * `visualization/` — Video annotation engine (rendering trails, mini-courts, speed meters).
* `data/reports/` — Output reports containing `dashboard.html` and extracted highlight clips (`highlights/annotated/`).

---

## 🔮 Next Steps & Approved Features
To address highlight quality and improve physical modeling, the following 3D physical modeling steps are approved for implementation next:
1. **3D Homography Projection:** Use the court keypoint model outputs to project 2D pixel coordinates to actual 3D court meters ($X, Y, Z$).
2. **Upgraded Physics Engine:** Track vertical height $Z$ to detect true ground bounces ($Z \le 0$), wall rebounds (boundary collisions), and net crossings.
3. **Advanced Highlight Triggers:** Implement logic for padel-specific events like *Por Tres* smashes (ball exiting over side glass), *Salidas* (recovering ball outside the court doors), and double bounces.
