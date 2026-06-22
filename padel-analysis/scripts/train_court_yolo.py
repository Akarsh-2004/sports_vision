#!/usr/bin/env python3
"""Fine-tune YOLOv8-pose on padel court keypoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "court_keypoints_yolo" / "data.yaml"
DEFAULT_COCO = ROOT.parents[1] / "Padel Court Detection.v1i.coco"


def _ensure_dataset(data_yaml: Path, coco_dir: Path, copy_images: bool) -> None:
    if data_yaml.exists():
        return
    convert = Path(__file__).with_name("convert_coco_to_yolo_pose.py")
    cmd = [
        sys.executable,
        str(convert),
        "--coco-dir",
        str(coco_dir),
        "--out-dir",
        str(data_yaml.parent),
    ]
    if copy_images:
        cmd.append("--copy-images")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 court keypoint model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset data.yaml")
    parser.add_argument("--coco-dir", type=Path, default=DEFAULT_COCO, help="Source COCO export")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Base YOLOv8 pose weights")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="cuda device id, cpu, or mps")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "court_pose")
    parser.add_argument("--name", default="yolov8n-padel-court")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()

    _ensure_dataset(args.data, args.coco_dir, args.copy_images)
    if args.convert_only:
        print(f"Dataset ready at {args.data}")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        mosaic=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    weights_dir = ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / "court_keypoints_yolov8.pt"
    if best.exists():
        import shutil

        shutil.copy2(best, dest)
        print(f"Copied best weights to {dest}")

    print(f"Training complete. Best weights: {best}")


if __name__ == "__main__":
    main()
