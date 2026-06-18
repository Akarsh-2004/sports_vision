#!/usr/bin/env python3
"""Package padel analysis to root output folder with dashboard."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package(match_dir: Path, video: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ("match_stats.json", "report.md", "full_output.json", "player_report.md", "training_report.md", "dashboard.html"):
        src = match_dir / name
        if src.exists():
            shutil.copy2(src, out_root / name)

    viz_src = ROOT / "data" / "reports" / "viz"
    if viz_src.exists():
        viz_dst = out_root / "viz"
        if viz_dst.exists():
            shutil.rmtree(viz_dst)
        shutil.copytree(viz_src, viz_dst)

    hl_src = match_dir / "highlights"
    if not hl_src.exists():
        hl_src = ROOT / "data" / "reports" / "highlights"
    if hl_src.exists():
        hl_dst = out_root / "highlights"
        if hl_dst.exists():
            shutil.rmtree(hl_dst)
        shutil.copytree(hl_src, hl_dst)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_dashboard.py"), "--match-dir", str(match_dir), "--video", str(video)],
        check=True,
        cwd=str(ROOT),
    )
    dash = match_dir / "dashboard.html"
    if dash.exists():
        shutil.copy2(dash, out_root / "dashboard.html")

    summary = {"match_id": match_dir.name, "video": str(video), "dashboard": "dashboard.html", "viz": "viz/", "highlights": "highlights/"}
    (out_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Packaged to {out_root}")
    print(f"Open: {out_root / 'dashboard.html'}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--match-dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="../output_padel_sample")
    args = p.parse_args()
    package(Path(args.match_dir), Path(args.video), Path(args.out).resolve())
