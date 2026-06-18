#!/usr/bin/env python3
"""Regenerate coach highlights + dashboard from existing full_output.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.highlights.coach_highlights import CoachHighlightEngine
from backend.utils.config import load_config
from backend.utils.types import MatchEvent, RallySegment


def main():
    p = argparse.ArgumentParser(description="Build coach highlights from saved match output")
    p.add_argument("--match-dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--regen-dashboard", action="store_true")
    args = p.parse_args()

    match_dir = Path(args.match_dir)
    full_path = match_dir / "full_output.json"
    data = json.loads(full_path.read_text(encoding="utf-8"))
    cfg = load_config()
    intel = data.get("intelligence", {})
    stats = data.get("stats", data)

    rallies = [
        RallySegment(
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            rally_length_shots=r.get("rally_length_shots", 0),
            wall_hits=r.get("wall_hits", 0),
            excitement_score=r.get("excitement_score", 0),
        )
        for r in data.get("rallies_all", stats.get("rallies", []))
    ]
    scored = [
        RallySegment(
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            rally_length_shots=r.get("rally_length_shots", 0),
            wall_hits=r.get("wall_hits", 0),
            excitement_score=r.get("excitement_score", 0),
        )
        for r in stats.get("rallies", [])
    ]
    events = []
    for e in stats.get("events", []):
        from backend.utils.types import EventType

        events.append(
            MatchEvent(
                frame_idx=e["frame_idx"],
                event_type=EventType(e["event_type"]),
                player_track_id=e.get("player_track_id"),
            )
        )

    engine = CoachHighlightEngine(cfg, match_dir)
    playable = data.get("playable_video", args.video)
    result = engine.generate(
        playable,
        rallies,
        scored,
        events,
        intel,
        stats.get("movement", {}).get("max_speed_kmh", 80),
        stats.get("target_track_id"),
    )

    data["coach_highlights"] = result
    full_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Manifest: {result.get('manifest_path')}")
    print(f"Clips: {len(result.get('paths', []))}")

    if args.regen_dashboard:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_dashboard.py"), "--match-dir", str(match_dir), "--video", args.video],
            check=True,
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
