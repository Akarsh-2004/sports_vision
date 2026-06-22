"""
Fine-tune YOLOv11n on the Padel Ball Detection v4 dataset.

Dataset: /Users/akarshsaklani/Desktop/krateasy_vision/Padel Ball Detection.v4i.yolov11
Training: Mac M4 Silicon — uses MPS backend automatically via Ultralytics

Usage:
    cd padel-analysis
    python scripts/train_ball_detector.py

Output:
    weights/padel_ball_yolo11n.pt   ← production weights (auto-copied)
    runs/train/padel_ball_*/        ← training logs + checkpoints
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ── resolve project root ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

DATASET_YAML = Path(
    "/Users/akarshsaklani/Desktop/krateasy_vision/Padel Ball Detection.v4i.yolov11/data.yaml"
)
WEIGHTS_OUT = PROJECT_DIR / "weights" / "padel_ball_yolo11n.pt"
RUNS_DIR = PROJECT_DIR / "runs" / "train"


def patch_yaml_paths(yaml_path: Path) -> Path:
    """
    Roboflow data.yaml uses relative paths like '../train/images'.
    Ultralytics needs absolute paths. Write a patched copy beside original.
    """
    import yaml  # type: ignore

    data = yaml.safe_load(yaml_path.read_text())
    
    # Set the dataset root path in the yaml
    data["path"] = str(yaml_path.parent)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    if "test" in data:
        data["test"] = "test/images"

    patched = yaml_path.parent / "_padel_ball_abs.yaml"
    patched.write_text(yaml.dump(data))
    print(f"[train_ball] Patched data.yaml → {patched}")
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
    print("  Padel Ball Detector — Fine-Tune YOLOv11n")
    print("=" * 60)
    print(f"  Dataset : {DATASET_YAML.parent.name}")
    print(f"  Base    : yolo11n.pt (COCO pretrained)")
    print(f"  Device  : mps (Mac M4)")
    print(f"  Output  : {WEIGHTS_OUT}")
    print("=" * 60)

    data_yaml = patch_yaml_paths(DATASET_YAML)

    checkpoint_path = RUNS_DIR / "padel_ball" / "weights" / "last.pt"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        model = YOLO(str(checkpoint_path))
        resume_arg = True
    else:
        print("No checkpoint found. Starting training from scratch...")
        model = YOLO("yolo11n.pt")  # downloads if not cached
        resume_arg = False

    results = model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=640,
        batch=16,            # M4 16GB unified memory handles this well
        device="mps",        # Metal Performance Shaders on M4
        project=str(RUNS_DIR),
        name="padel_ball",
        resume=resume_arg,
        patience=15,         # early stopping if no improvement
        lr0=0.005,           # lower LR for fine-tuning
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentation — important for small fast ball
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,         # ball is circular, don't rotate
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.05,
        auto_augment="randaugment",
        # Logging
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    # ── Copy best weights to padel-analysis/weights/ ──────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, WEIGHTS_OUT)
        print(f"\n✅ Best weights copied → {WEIGHTS_OUT}")
    else:
        print(f"\n⚠️  best.pt not found at {best}. Check training output.")

    # ── Quick validation ───────────────────────────────────────────────────────
    print("\n── Validation on padel ball test set ──")
    val_model = YOLO(str(WEIGHTS_OUT) if WEIGHTS_OUT.exists() else str(best))
    val_results = val_model.val(data=str(data_yaml), device="mps", verbose=True)
    print(f"  mAP50    : {val_results.box.map50:.4f}")
    print(f"  mAP50-95 : {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall   : {val_results.box.mr:.4f}")

    print("\n🎾 Training complete. Update padel.yaml:")
    print("   models:")
    print("     ball_detector: yolo")
    print("     ball_yolo_weights: weights/padel_ball_yolo11n.pt")


if __name__ == "__main__":
    train()
