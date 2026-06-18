#!/usr/bin/env python3
"""Copy pipeline report + generate highlights into a root output folder."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package(match_dir: Path, video: Path, out_root: Path, max_seconds: float = 60.0) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    stats = match_dir / "match_stats.json"
    if not stats.exists():
        sys.exit(f"Missing {stats}")

    for name in ("match_stats.json", "report.md", "full_output.json"):
        src = match_dir / name
        if src.exists():
            shutil.copy2(src, out_root / name)

    viz_src = ROOT / "data" / "reports" / "viz"
    viz_dst = out_root / "viz"
    if viz_src.exists():
        if viz_dst.exists():
            shutil.rmtree(viz_dst)
        shutil.copytree(viz_src, viz_dst)

    hl_dir = out_root / "highlights"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_highlights.py"),
            "--video", str(video),
            "--stats", str(stats),
            "--max-seconds", str(max_seconds),
            "--out-dir", str(hl_dir),
        ],
        check=True,
        cwd=str(ROOT),
    )

    summary = {
        "video": str(video),
        "analyzed_seconds": max_seconds,
        "match_id": match_dir.name,
        "outputs": {
            "stats": "match_stats.json",
            "report": "report.md",
            "visualizations": "viz/",
            "highlights": "highlights/",
            "manifest": "highlights/highlights_manifest.json",
        },
    }
    (out_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Packaged to {out_root}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--match-dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seconds", type=float, default=60.0)
    args = p.parse_args()
    package(Path(args.match_dir), Path(args.video), Path(args.out), args.max_seconds)
