# Sports Vision — Project Context

> **Repo:** [github.com/Akarsh-2004/sports_vision](https://github.com/Akarsh-2004/sports_vision)  
> **Purpose:** AI coaching platform for racket sports — computer vision + sports intelligence, not just reports.

This file gives humans and AI agents enough context to work on the codebase without reading the full history.

---

## What This Project Is

**Sports Vision** is a monorepo for AI-powered match analysis:

| Package | Sport | Status |
|---------|-------|--------|
| `tennis-analysis/` | Tennis (singles/doubles) | Production pipeline + React frontend + API |
| `padel-analysis/` | Padel (doubles, walls, glass) | Intelligence engine + AI Coach dashboard v2 |

**Product vision:** Move from *"run analysis → PDF report"* to an **interactive AI coach** — clickable timeline, evidence-backed highlights, tactical reasoning, multi-match learning.

**Design principle:**
```
Computer vision = sensor layer
World Model + Intelligence = reasoning layer
Dashboard + clips = coaching experience
```

---

## Repository Layout

```
sports_vision/
├── context.md              ← you are here
├── README.md
├── tennis-analysis/        ← tennis CV pipeline (17 stages)
├── padel-analysis/         ← padel intelligence engine (7 layers)
├── output_padel_sample/    ← packaged demo output (gitignored)
├── padel_sample.mp4        ← test video (gitignored)
└── output_*/               ← local run artifacts (gitignored)
```

---

## Tennis Pipeline (`tennis-analysis/`)

Classic end-to-end match analyzer inspired by [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) but extended:

- YOLO11 player detection + BoT-SORT tracking
- Ball tracking (heuristic / optional YOLO)
- Court homography, pose, stroke classification
- Rally segmentation, highlights, performance radar
- FastAPI backend + Vite React frontend
- LLM reports (Ollama / template)

**Run:**
```bash
cd tennis-analysis
pip install -r requirements.txt
python run.py run path/to/video.mp4
```

---

## Padel Pipeline (`padel-analysis/`)

Padel-specific fork with a **reasoning-layer architecture** (see `padel-analysis/ARCHITECTURE.md`).

### Architecture (7 layers)

```
Layer 1  Vision          YOLO, tracking, pose, ball
Layer 2  Geometry        court projection (10m × 20m)
Layer 3  Match State FSM  RALLY, NET_ATTACK, WALL_EXCHANGE, DEAD_TIME
Layer 4  Interaction      Player → Ball → Wall → Ground graphs
Layer 5  Tactical         positioning, patterns, opponent profiles
Layer 6  Knowledge        structured facts for LLM
Layer 7  Reports          coach / player / training + dashboard
```

**World Model** (`backend/intelligence/world/world_model.py`) is the single source of truth per frame — Court, Players, Ball, Match, Events, Confidence.

### Key modules

| Path | Role |
|------|------|
| `backend/pipeline/orchestrator.py` | Main pipeline entry |
| `backend/intelligence/pipeline.py` | Post-vision intelligence finalize |
| `backend/intelligence/shot/understanding.py` | intent, pressure, EPV, decision quality |
| `backend/intelligence/physics/ball_physics.py` | Kalman, bounce, glass reflection |
| `backend/highlights/coach_highlights.py` | Categorized, ranked coaching clips |
| `backend/tracking/ball_shot_detector.py` | Trajectory inflection hits (tennis_analysis style) |
| `backend/visualization/mini_court.py` | Top-down 10×20m overlay |
| `scripts/generate_dashboard.py` | AI Coach v2 HTML dashboard |
| `scripts/package_padel_output.py` | Package to `output_padel_sample/` |
| `configs/padel.yaml` | All tunables |

### Run

```bash
cd padel-analysis
pip install -r requirements.txt
python run.py run ../padel_sample.mp4
python scripts/package_padel_output.py --match-dir data/reports/<match_id> --video ../padel_sample.mp4 --out ../output_padel_sample
```

Open: `output_padel_sample/dashboard.html`

### Test video

- `padel_sample.mp4` — ~65s broadcast padel, audience close-ups, dead time
- Known limitation: without manual court calibration + padel ball YOLO, ball/rally accuracy is limited on broadcast footage

---

## Current Capabilities (Padel)

| Feature | Status |
|---------|--------|
| World Model per frame | ✅ |
| Shot understanding (not just stroke label) | ✅ |
| Rally interaction graphs | ✅ (improving) |
| Coach highlight categories (13 types) | ✅ |
| AI Coach dashboard v2 | ✅ static HTML |
| Confidence propagation + self-evaluation | ✅ |
| Pattern mining, EPV, opponent profiles | ✅ basic |
| Learning DB (multi-match SQLite) | ✅ schema |
| Fine-tuned padel ball YOLO | ❌ uses heuristic |
| React coach UI + chat (LLM) | ❌ planned |
| Annotated replay video export | ❌ planned |

---

## Known Issues / Active Work

1. **Ball detection** — heuristic fails on broadcast; need Roboflow/YOLO padel ball weights
2. **Rally chains** — were inflated by per-frame pose strokes; fixed with shot debouncing + ball trajectory hits + multi-player contact
3. **Highlight timing** — 30fps source vs 25fps analysis frames; fixed via FFmpeg normalization (`imageio-ffmpeg`)
4. **Court calibration** — auto homography weak on broadcast; manual `run.py calibrate` recommended
5. **Fake long rally chains** — caused by single-player stroke spam; use `ball_shot_frames` + deduped interaction graph

---

## Model & Dataset Strategy

**No single official padel model exists.** Practical path:

| Priority | Resource |
|----------|----------|
| Ball YOLO | [Roboflow padel-ball-ikshu](https://universe.roboflow.com/padel-fqrh4/padel-ball-ikshu) (~4.5k images) |
| Court keypoints | [padel-court-detection-vrupi](https://universe.roboflow.com/padel-vyxal/padel-court-detection-vrupi) |
| Research dataset | [PadelTracker100 on Zenodo](https://zenodo.org/records/17020011) (~100k frames, heavy) |
| Tennis transfer | [abdullahtarek tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) ball YOLO + trajectory logic |

Config hook: `configs/padel.yaml` → `models.ball_detector: yolo` (not yet wired with weights)

---

## Output Artifacts

Per match in `padel-analysis/data/reports/<match_id>/`:

```
match_stats.json
full_output.json      # world model + intelligence + coach_highlights
report.md
player_report.md
training_report.md
dashboard.html
highlights/
  manifest.json
  best_rallies/
  top_moments/
  ...
viz/                  # heatmap, radar, shot_chart, timeline
```

Packaged demo: `output_padel_sample/`

---

## Configuration Highlights (`padel-analysis/configs/padel.yaml`)

```yaml
court:
  court_length_m: 20.0
  court_width_m: 10.0
  use_manual_calibration: false   # set true after calibrate UI

active_play:
  enabled: true                 # skip audience close-ups

intelligence:
  enabled: true

coach_highlights:
  enabled: true

strokes:
  min_shot_gap_s: 0.55          # debounce duplicate hits
```

---

## Product Roadmap (UX > CV)

**Done:** World model, shot understanding, categorized highlights, dashboard v2 skeleton

**Next:**
1. Padel ball YOLO integration
2. Annotated replay video (mini court + ball path overlay)
3. Evidence-linked stats in React UI
4. LLM coach chat over knowledge graph
5. Season dashboard from `padel_learning.db`
6. Ghost player / optimal position overlay

---

## References

- [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) — mini court, ball shot frames, YOLO ball
- [PadelTracker100](https://zenodo.org/records/17020011) — academic padel dataset
- [Roboflow Padel Universe](https://universe.roboflow.com/search?q=padel) — community models

---

## For AI Agents

When editing this repo:

1. **Prefer `padel-analysis/`** for new padel features; keep `tennis-analysis/` stable unless syncing shared utils
2. **World Model is source of truth** — don't recompute court/ball state in downstream modules
3. **Minimize scope** — small focused diffs; match existing naming and patterns
4. **Don't commit** videos, `data/reports/*`, large weights, or `output_*` folders
5. **Rally unit = ball trajectory segments**, not raw pose frames per player
6. Read `padel-analysis/ARCHITECTURE.md` before intelligence-layer changes

**Primary test command:**
```bash
cd padel-analysis && python run.py run ../padel_sample.mp4
```
