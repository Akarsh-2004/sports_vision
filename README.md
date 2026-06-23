# Sports Vision

AI coaching platform for racket sports — computer vision pipelines plus sports intelligence (World Model, tactical reasoning, interactive coach UI).

**Read [`context.md`](context.md) first** for full project context, architecture, and agent instructions.

## Packages

| Directory | Description |
|-----------|-------------|
| [`tennis-analysis/`](tennis-analysis/) | Tennis match analysis — 17-stage CV pipeline, API, React frontend |
| [`padel-analysis/`](padel-analysis/) | Padel intelligence engine — World Model, coach highlights, AI dashboard |

## Quick start (Padel)

```bash
cd padel-analysis
pip install -r requirements.txt
python run.py run path/to/match.mp4

# Or use the web UI
python run.py serve
```

Outputs land in `padel-analysis/data/reports/<match_id>/` including `dashboard.html`.

## Quick start (Tennis)

```bash
cd tennis-analysis
pip install -r requirements.txt
python run.py run path/to/match.mp4
```

## License

TBD — add license before public distribution.
