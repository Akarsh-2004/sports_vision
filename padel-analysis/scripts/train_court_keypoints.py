"""
Fine-tune YOLOv8-pose on the Padel Court Keypoints dataset.

Dataset: /Users/akarshsaklani/Desktop/krateasy_vision/Padel.v1i.yolov8
Classes: Barrier_keypoint, Field_keypoint, Net_keypoint, Wall_keypoint (4 keypoint types)
Training: Mac M4 Silicon — uses MPS backend

Usage:
    cd padel-analysis
    python scripts/train_court_keypoints.py

Output:
    weights/padel_court_keypoints.pt   ← production weights (auto-copied)
    runs/train/padel_court_*/          ← training logs + checkpoints
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

DATASET_YAML = Path(
    "/Users/akarshsaklani/Desktop/krateasy_vision/Padel.v1i.yolov8/data.yaml"
)
WEIGHTS_OUT = PROJECT_DIR / "weights" / "padel_court_keypoints.pt"
RUNS_DIR = PROJECT_DIR / "runs" / "train"


def patch_yaml_paths(yaml_path: Path) -> Path:
    import yaml  # type: ignore

    data = yaml.safe_load(yaml_path.read_text())
    
    # Set the dataset root path in the yaml
    data["path"] = str(yaml_path.parent)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    if "test" in data:
        data["test"] = "test/images"

    patched = yaml_path.parent / "_padel_court_abs.yaml"
    patched.write_text(yaml.dump(data))
    print(f"[train_court] Patched data.yaml → {patched}")
    return patched


def train() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML not found: {DATASET_YAML}")
        sys.exit(1)

    print("=" * 60)
    print("  Padel Court Keypoints — Fine-Tune YOLOv8n-pose")
    print("=" * 60)
    print(f"  Dataset : {DATASET_YAML.parent.name}")
    print(f"  Classes : Barrier, Field, Net, Wall keypoints (4)")
    print(f"  Base    : yolov8n-pose.pt (COCO pose pretrained)")
    print(f"  Device  : mps (Mac M4)")
    print(f"  Output  : {WEIGHTS_OUT}")
    print("=" * 60)

    data_yaml = patch_yaml_paths(DATASET_YAML)

    checkpoint_path = RUNS_DIR / "padel_court" / "weights" / "last.pt"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        model = YOLO(str(checkpoint_path))
        resume_arg = True
    else:
        print("No checkpoint found. Starting training from scratch...")
        model = YOLO("yolov8n-pose.pt")
        resume_arg = False

    results = model.train(
        data=str(data_yaml),
        epochs=20,
        batch=8,             # Smaller batch — keypoint task is heavier
        device="mps",
        project=str(RUNS_DIR),
        name="padel_court",
        resume=resume_arg,
        patience=20,
        lr0=0.003,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Court doesn't change color much — moderate augmentation
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,         # Courts do appear at slight angles
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,          # Court is always right-side-up
        mosaic=0.8,
        mixup=0.0,
        # Keypoint-specific
        pose=12.0,           # Pose loss weight
        kobj=2.0,            # Keypoint obj loss weight
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, WEIGHTS_OUT)
        print(f"\n✅ Best court keypoint weights copied → {WEIGHTS_OUT}")
    else:
        print(f"\n⚠️  best.pt not found at {best}")

    print("\n── Validation on padel court test set ──")
    val_model = YOLO(str(WEIGHTS_OUT) if WEIGHTS_OUT.exists() else str(best))
    val_results = val_model.val(data=str(data_yaml), device="mps", verbose=True)
    print(f"  mAP50    : {val_results.box.map50:.4f}")
    print(f"  mAP50-95 : {val_results.box.map:.4f}")

    print("\n🏟️  Training complete. Update padel.yaml:")
    print("   court:")
    print("     use_yolo_court: true")
    print("     yolo_court_weights: weights/padel_court_keypoints.pt")


if __name__ == "__main__":
    train()
