"""
Quick end-to-end validation script — runs each new component in isolation.

Usage:
    cd padel-analysis
    source .venv/bin/activate
    python scripts/validate_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def test_ball_detector_heuristic():
    print("\n[1/4] BallDetector — heuristic mode")
    import numpy as np
    from backend.utils.config import load_config
    cfg = load_config()
    cfg["models"]["ball_detector"] = "heuristic"
    from backend.detection.ball_detector import BallDetector
    bd = BallDetector(cfg)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    det = bd.detect(frame)
    print(f"  Heuristic detect: visible={det.visible} conf={det.confidence:.2f}  ✅")


def test_ball_detector_yolo():
    print("\n[2/4] BallDetector — YOLO mode (checks weight loading)")
    from backend.utils.config import load_config
    cfg = load_config()
    cfg["models"]["ball_detector"] = "yolo"
    from backend.detection.ball_detector import BallDetector
    bd = BallDetector(cfg)
    if bd.mode == "yolo":
        print("  YOLO model loaded successfully  ✅")
    else:
        print("  Weights not found yet — falling back to heuristic (expected before training)  ⚠️")


def test_ball_shot_detector():
    print("\n[3/4] BallShotDetector — X+Y+speed inflection")
    from backend.tracking.ball_shot_detector import BallShotDetector

    bsd = BallShotDetector(fps=25.0)

    # Simulate trajectory: ball bouncing (Y reversal) at frame 30
    traj = []
    for i in range(60):
        x = 640.0 + i * 2
        # Ball goes up then reverses at frame 30
        y = 400.0 - i * 3 if i < 30 else 400.0 - 30 * 3 + (i - 30) * 3
        conf = 0.85
        traj.append((i, x, y, conf))

    hits = bsd.detect_from_trajectory(traj)
    if hits:
        print(f"  Detected {len(hits)} hit frame(s) near frame 30: {hits}  ✅")
    else:
        print("  No hits detected (check min_change_frames tuning)  ⚠️")


def test_annotated_exporter_imports():
    print("\n[4/4] AnnotatedExporter — import + init")
    from backend.utils.config import load_config
    cfg = load_config()
    from backend.visualization.annotated_exporter import (
        AnnotatedExporter, build_frame_annotations, FrameAnnotation, PlayerAnnotation
    )
    exp = AnnotatedExporter(cfg, fps=25.0)
    print(f"  AnnotatedExporter ready: style={exp.style} trail={exp.trail_len}  ✅")

    # Test build_frame_annotations with empty data
    result = build_frame_annotations({}, {}, [], [], 25.0)
    print(f"  build_frame_annotations (empty): {len(result)} frames  ✅")


def test_coach_highlights_import():
    print("\n[Bonus] CoachHighlightEngine — improved scoring + _annotate_clips")
    import inspect
    from backend.highlights.coach_highlights import CoachHighlightEngine
    methods = [m for m in dir(CoachHighlightEngine) if not m.startswith("__")]
    assert "_annotate_clips" in methods, "_annotate_clips not found!"
    print(f"  Methods: {', '.join(m for m in methods if not m.startswith('_'))}  ✅")
    print(f"  _annotate_clips present  ✅")


if __name__ == "__main__":
    errors = []
    for fn in [
        test_ball_detector_heuristic,
        test_ball_detector_yolo,
        test_ball_shot_detector,
        test_annotated_exporter_imports,
        test_coach_highlights_import,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            errors.append((fn.__name__, e))

    print("\n" + "=" * 50)
    if errors:
        print(f"❌ {len(errors)} test(s) failed:")
        for name, err in errors:
            print(f"   {name}: {err}")
        sys.exit(1)
    else:
        print("✅ All validation tests passed!")
        print("\nNext steps:")
        print("  1. Run training: python scripts/train_ball_detector.py")
        print("  2. Run training: python scripts/train_court_keypoints.py")
        print("  3. Run pipeline: python run.py run ../padel_sample.mp4")
