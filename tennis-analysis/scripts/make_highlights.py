#!/usr/bin/env python3
"""Extract labeled highlight clips from match stats + source video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.utils.config import load_config
from backend.utils.ffmpeg import get_ffmpeg, run_ffmpeg
from backend.utils.logging import get_logger
from backend.utils.types import RallySegment
from backend.highlights.highlight_generator import HighlightGenerator

logger = get_logger(__name__)


def load_stats(stats_path: Path) -> dict:
    return json.loads(stats_path.read_text(encoding="utf-8"))


def filter_rallies(rallies: list[RallySegment], fps: float, min_shots: int, min_duration_s: float) -> list[RallySegment]:
    valid: list[RallySegment] = []
    for r in rallies:
        duration_s = (r.end_frame - r.start_frame) / fps
        if r.rally_length_shots < min_shots:
            continue
        if duration_s < min_duration_s:
            continue
        valid.append(r)
    return sorted(valid, key=lambda r: r.excitement_score, reverse=True)


def load_rallies(data: dict, max_time_s: float | None, fps: float) -> list[RallySegment]:
    rallies: list[RallySegment] = []
    for r in data.get("rallies", []):
        if max_time_s is not None and r["start_frame"] / fps > max_time_s:
            continue
        rallies.append(
            RallySegment(
                start_frame=r["start_frame"],
                end_frame=min(r["end_frame"], int(max_time_s * fps)) if max_time_s else r["end_frame"],
                rally_length_shots=r.get("rally_length_shots", 0),
                outcome=r.get("outcome", "unknown"),
                excitement_score=r.get("excitement_score", 0.0),
            )
        )
    return rallies


def rally_label(r: RallySegment, idx: int, events: list[dict], fps: float) -> str:
    tags = []
    if r.rally_length_shots >= 8:
        tags.append("LONG RALLY")
    tags.append(f"{r.rally_length_shots} shots")
    end_s = r.end_frame / fps
    for e in events:
        if abs(e["frame_idx"] / fps - end_s) < 1.0:
            tags.append(e["event_type"].replace("_", " ").upper())
    title = f"Highlight {idx + 1}"
    if tags:
        title += " — " + " | ".join(tags)
    return title


def extract_segment(video: str, out: Path, start_s: float, end_s: float, label: str | None = None) -> None:
    duration = max(0.1, end_s - start_s)
    vf = None
    if label:
        safe = label.replace(":", "\\:").replace("'", "\\'")
        vf = (
            f"drawtext=text='{safe}':fontsize=28:fontcolor=white:borderw=2:bordercolor=black:"
            f"x=(w-text_w)/2:y=50:enable='between(t\\,0\\,4)'"
        )
    args = ["-y", "-ss", str(start_s), "-i", video, "-t", str(duration)]
    if vf:
        args += ["-vf", vf]
    args += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart", str(out),
    ]
    run_ffmpeg(args)


def concat_clips(clips: list[Path], out: Path) -> None:
    if not clips:
        return
    list_file = out.parent / "_concat_list.txt"
    lines = [f"file '{c.resolve().as_posix()}'" for c in clips]
    list_file.write_text("\n".join(lines), encoding="utf-8")
    run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)])
    list_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate labeled tennis highlight clips")
    parser.add_argument("--video", required=True)
    parser.add_argument("--stats", help="match_stats.json path")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--out-dir", help="Output directory")
    parser.add_argument("--min-shots", type=int, default=3, help="Min shots for a valid rally highlight")
    parser.add_argument("--include-raw-segment", action="store_true", help="Also export uncut first N seconds")
    args = parser.parse_args()

    cfg = load_config()
    fps = cfg["pipeline"]["target_fps"]
    preroll = cfg["highlights"]["preroll_frames"]
    postroll = cfg["highlights"]["postroll_frames"]

    if args.stats:
        stats_path = Path(args.stats)
    else:
        reports = sorted(
            Path(cfg["paths"]["data_reports"]).glob("*/match_stats.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not reports:
            sys.exit("No match_stats.json found. Run the pipeline first.")
        stats_path = reports[-1]

    data = load_stats(stats_path)
    out_dir = Path(args.out_dir) if args.out_dir else stats_path.parent / "highlights"
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()
    logger.info("Using ffmpeg: %s", ffmpeg)

    all_rallies = load_rallies(data, args.max_seconds, fps)
    valid_rallies = filter_rallies(all_rallies, fps, args.min_shots, min_duration_s=1.5)

    manifest: dict = {
        "source_video": str(Path(args.video).resolve()),
        "stats": str(stats_path.resolve()),
        "window_seconds": args.max_seconds,
        "fps_assumed": fps,
        "note": (
            "REAL HIGHLIGHTS = rally_*.mp4 and highlights_reel.mp4 (AI-detected rallies with labels). "
            "raw_segment.mp4 is NOT a highlight — it is an uncut time crop for reference."
        ),
        "clips": [],
    }

    if args.include_raw_segment:
        raw = out_dir / "raw_segment_first_60s.mp4"
        extract_segment(args.video, raw, 0.0, args.max_seconds)
        manifest["clips"].append({
            "file": raw.name,
            "type": "raw_segment",
            "start_s": 0.0,
            "end_s": args.max_seconds,
            "label": "UNCUT SEGMENT (not a highlight)",
            "description": "Plain crop of first minute — no rally detection applied.",
        })

    labeled_paths: list[Path] = []
    for i, rally in enumerate(valid_rallies):
        start_f = max(0, rally.start_frame - preroll)
        end_f = rally.end_frame + postroll
        start_s = start_f / fps
        end_s = end_f / fps
        label = rally_label(rally, i, data.get("events", []), fps)
        out_plain = out_dir / f"rally_{i:02d}_plain.mp4"
        out_labeled = out_dir / f"rally_{i:02d}_labeled.mp4"
        extract_segment(args.video, out_plain, start_s, end_s)
        extract_segment(args.video, out_labeled, start_s, end_s, label=label)
        labeled_paths.append(out_labeled)
        manifest["clips"].append({
            "file": out_labeled.name,
            "plain_file": out_plain.name,
            "type": "rally_highlight",
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "duration_s": round(end_s - start_s, 2),
            "rally_frames": [rally.start_frame, rally.end_frame],
            "shots": rally.rally_length_shots,
            "excitement_score": rally.excitement_score,
            "label": label,
            "description": f"Detected rally with {rally.rally_length_shots} shots; includes {preroll/fps:.1f}s pre-roll.",
        })

    # Rejected rallies (why they were skipped)
    rejected = []
    for r in all_rallies:
        if r in valid_rallies:
            continue
        rejected.append({
            "rally_frames": [r.start_frame, r.end_frame],
            "shots": r.rally_length_shots,
            "duration_s": round((r.end_frame - r.start_frame) / fps, 2),
            "reason": "Too short or too few shots — not a real highlight",
        })
    manifest["rejected_rallies"] = rejected

    if labeled_paths:
        reel = out_dir / "highlights_reel.mp4"
        concat_clips(labeled_paths, reel)
        manifest["highlights_reel"] = reel.name
    else:
        manifest["highlights_reel"] = None
        manifest["warning"] = "No valid rallies found in this window. Only raw segment exists."

    manifest_path = out_dir / "highlights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
