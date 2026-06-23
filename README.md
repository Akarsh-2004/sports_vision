# Sports Vision

AI coaching platform for racket sports — computer vision pipelines, tactical intelligence, and a web UI for match analysis.

| Package | Sport | Docs |
|---------|-------|------|
| [`padel-analysis/`](padel-analysis/) | Padel (doubles, walls) | [**Full setup guide →**](padel-analysis/README.md) |
| [`tennis-analysis/`](tennis-analysis/) | Tennis | [Tennis README →](tennis-analysis/README.md) |

**Repository:** [github.com/Akarsh-2004/sports_vision](https://github.com/Akarsh-2004/sports_vision)

---

## What you need

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11+ | 3.12 recommended |
| **pip** | latest | Comes with Python |
| **FFmpeg** | any recent | Required for video ingest & highlight clips |
| **Node.js** | 18+ | Only for the web UI (`npm`) |
| **GPU** | optional | Speeds up YOLO; CPU works for short clips |

### Install FFmpeg

- **Windows:** `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Verify: `ffmpeg -version`

---

## Get started (5 minutes)

### 1. Clone the repo

```bash
git clone https://github.com/Akarsh-2004/sports_vision.git
cd sports_vision
```

### 2. Pick a sport

**Padel (recommended — includes web UI):**

```bash
cd padel-analysis
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Tennis:**

```bash
cd tennis-analysis
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Configure GPU / device

Edit the sport's config YAML and set the compute device for your machine:

| Platform | Set `models.device` to |
|----------|------------------------|
| NVIDIA GPU | `cuda` |
| Apple Silicon | `mps` |
| Windows / Linux (no GPU) | `cpu` |
| Auto-detect | `auto` |

- Padel: `padel-analysis/configs/padel.yaml`
- Tennis: `tennis-analysis/configs/default.yaml`

> The default padel config uses `mps` (Mac). **Windows users should change this to `cuda` or `cpu`.**

### 4. Run analysis

**Padel — command line:**

```bash
cd padel-analysis
python run.py run path/to/your_match.mp4
```

**Padel — web UI:**

```bash
# Terminal 1 — API
cd padel-analysis
python run.py serve

# Terminal 2 — frontend
cd padel-analysis/frontend
npm install
npm run dev
```

Open **http://localhost:5174**, upload a video, and wait for results.

**Tennis — command line:**

```bash
cd tennis-analysis
python run.py path/to/your_match.mp4
```

---

## Outputs

After a run, results are saved under `data/reports/<match_id>/`:

| File | Description |
|------|-------------|
| `report.md` | Coach-style written summary |
| `match_stats.json` | Structured performance metrics |
| `full_output.json` | Complete pipeline bundle (scores, rallies, highlights) |
| `dashboard.html` | Interactive charts (padel) |
| `highlights/` | Top rally video clips |

---

## Repository layout

```
sports_vision/
├── README.md                 ← you are here
├── context.md                ← architecture & agent context
├── padel-analysis/           ← padel pipeline + web UI
│   ├── run.py                ← CLI entry point
│   ├── configs/padel.yaml    ← all tunable settings
│   ├── backend/              ← CV + intelligence engine
│   ├── frontend/             ← React web UI
│   ├── scripts/              ← training & validation
│   └── data/                 ← raw videos, reports (gitignored)
└── tennis-analysis/          ← tennis pipeline + frontend
```

---

## Optional: model weights

Custom YOLO weights are **not** included in git (too large). The pipeline still runs without them:

- **Players:** Ultralytics downloads `yolo11n.pt` automatically on first run
- **Ball:** Falls back to heuristic color/motion detection
- **Court:** Falls back to classical line detection

For best accuracy, train or place weights in `padel-analysis/weights/`:

```
weights/padel_ball_yolo11n.pt      # ball detector
weights/padel_court_keypoints.pt   # court keypoints
```

See [`padel-analysis/README.md`](padel-analysis/README.md) for training commands.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ffmpeg not found` | Install FFmpeg and ensure it is on your `PATH` |
| Slow on CPU | Use `--max-frames 500` for a quick test run |
| CUDA / MPS errors | Set `models.device: cpu` in the config YAML |
| Port already in use | `python run.py serve --port 8002` |
| `npm` not found | Install Node.js 18+ from [nodejs.org](https://nodejs.org) |
| Out of memory | Lower `pipeline.target_width` / `target_height` in config |

---

## Tests

```bash
cd padel-analysis && pytest tests/ -v
cd tennis-analysis && pytest tests/ -v
```

---

## More documentation

- **Padel (full guide):** [`padel-analysis/README.md`](padel-analysis/README.md)
- **Tennis:** [`tennis-analysis/README.md`](tennis-analysis/README.md)
- **Architecture & roadmap:** [`context.md`](context.md)

## License

TBD — add a license before public distribution.
