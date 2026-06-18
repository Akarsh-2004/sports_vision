#!/usr/bin/env python3
"""CLI entry point for tennis match analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Tennis Match Analysis Pipeline")
    parser.add_argument("video", help="Path to input match video")
    parser.add_argument("--click-x", type=float, help="Target player click X")
    parser.add_argument("--click-y", type=float, help="Target player click Y")
    parser.add_argument("--max-frames", type=int, help="Limit frames for quick test")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run("backend.api.main:app", host=args.host, port=args.port, reload=True)
        return

    cfg = load_config(args.config) if args.config else None
    orch = PipelineOrchestrator(cfg)
    click = (args.click_x, args.click_y) if args.click_x is not None and args.click_y is not None else None

    def progress(cur, total, stage):
        pct = 100 * cur / max(total, 1)
        print(f"\r[{pct:5.1f}%] {stage} ({cur}/{total})", end="", flush=True)

    stats = orch.run(args.video, target_click=click, progress_callback=progress, max_frames=args.max_frames)
    print(f"\nDone. Match ID: {stats.match_id}")
    print(f"Overall score: {stats.scores.overall:.0f}/100")
    print(f"Report saved to data/reports/{stats.match_id}/")


if __name__ == "__main__":
    main()
