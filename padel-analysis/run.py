#!/usr/bin/env python3
"""CLI entry point for padel match analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Padel Match Analysis Pipeline")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run full analysis pipeline")
    run_p.add_argument("video", help="Path to input match video")
    run_p.add_argument("--click-x", type=float, help="Target player click X")
    run_p.add_argument("--click-y", type=float, help="Target player click Y")
    run_p.add_argument("--max-frames", type=int, help="Limit frames for quick test")
    run_p.add_argument("--config", help="Path to YAML config")
    run_p.add_argument(
        "--selection",
        choices=["single", "pair", "all"],
        help="Analyze one player, a pair, or all four",
    )

    cal_p = sub.add_parser("calibrate", help="Interactive court landmark calibration")
    cal_p.add_argument("video", help="Path to input video")
    cal_p.add_argument("--frame", type=int, default=0, help="Frame index for calibration")

    serve_p = sub.add_parser("serve", help="Start FastAPI server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8001)

    # Default: treat first positional arg as video for backward-style invocation
    parser.add_argument("video_legacy", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--click-x", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--click-y", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="0.0.0.0", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8001, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "serve" or getattr(args, "serve", False):
        import uvicorn

        port = args.port if args.command == "serve" else args.port
        uvicorn.run("backend.api.main:app", host=args.host, port=port, reload=True)
        return

    if args.command == "calibrate":
        from backend.court.calibrate_ui import run_calibration_ui

        run_calibration_ui(args.video, frame_index=args.frame)
        print("Calibration saved to data/calibration/")
        return

    video = None
    if args.command == "run":
        video = args.video
    elif args.video_legacy:
        video = args.video_legacy

    if not video:
        parser.print_help()
        sys.exit(1)

    cfg_path = args.config if args.command == "run" else getattr(args, "config", None)
    cfg = load_config(cfg_path) if cfg_path else load_config()
    if args.command == "run" and args.selection:
        cfg.setdefault("selection", {})["mode"] = args.selection

    orch = PipelineOrchestrator(cfg)
    click_x = args.click_x if args.command == "run" else getattr(args, "click_x", None)
    click_y = args.click_y if args.command == "run" else getattr(args, "click_y", None)
    max_frames = args.max_frames if args.command == "run" else getattr(args, "max_frames", None)
    click = (click_x, click_y) if click_x is not None and click_y is not None else None

    def progress(cur, total, stage):
        pct = 100 * cur / max(total, 1)
        print(f"\r[{pct:5.1f}%] {stage} ({cur}/{total})", end="", flush=True)

    stats = orch.run(video, target_click=click, progress_callback=progress, max_frames=max_frames)
    print(f"\nDone. Match ID: {stats.match_id}")
    print(f"Overall score: {stats.scores.overall:.0f}/100")
    print(f"Report saved to data/reports/{stats.match_id}/")


if __name__ == "__main__":
    main()
