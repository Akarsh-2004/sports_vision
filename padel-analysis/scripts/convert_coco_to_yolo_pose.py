#!/usr/bin/env python3
"""Convert Roboflow COCO keypoint export to Ultralytics YOLO pose format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

KEYPOINT_NAMES = [
    "cage_bottom_left_close",
    "cage_bottom_left_far",
    "cage_bottom_right_close",
    "cage_bottom_right_far",
    "cage_top_left_close",
    "cage_top_left_far",
    "cage_top_right_close",
    "cage_top_right_far",
    "court_bottom_left_close",
    "court_bottom_left_far",
    "court_bottom_right_close",
    "court_bottom_right_far",
    "court_top_left_close",
    "court_top_left_far",
    "court_top_right_close",
    "court_top_right_far",
    "net_bottom_left",
    "net_bottom_right",
    "net_top_left",
    "net_top_right",
    "service_centre_close",
    "service_centre_far",
    "service_left_close",
    "service_left_far",
    "service_right_close",
    "service_right_far",
]

SKELETON = [
    [9, 10],
    [11, 12],
    [13, 14],
    [15, 16],
    [9, 11],
    [10, 12],
    [13, 15],
    [14, 16],
    [17, 18],
    [19, 20],
    [17, 19],
    [18, 20],
    [21, 23],
    [22, 24],
    [23, 25],
    [24, 26],
]


def _load_coco(split_dir: Path) -> dict:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations: {ann_path}")
    return json.loads(ann_path.read_text(encoding="utf-8"))


def _keypoints_for_image(annotations: list[dict], image_id: int) -> list[tuple[float, float, int]]:
    """Return 26 keypoints (x, y, v) in fixed category order 1..26."""
    by_cat: dict[int, tuple[float, float, int]] = {}
    for ann in annotations:
        if ann["image_id"] != image_id or ann["category_id"] == 0:
            continue
        kpts = ann.get("keypoints") or []
        if len(kpts) < 3:
            continue
        by_cat[ann["category_id"]] = (float(kpts[0]), float(kpts[1]), int(kpts[2]))

    ordered: list[tuple[float, float, int]] = []
    for cat_id in range(1, 27):
        ordered.append(by_cat.get(cat_id, (0.0, 0.0, 0)))
    return ordered


def _bbox_from_keypoints(
    keypoints: list[tuple[float, float, int]], width: int, height: int, pad: float = 0.02
) -> tuple[float, float, float, float]:
    visible = [(x, y) for x, y, v in keypoints if v > 0]
    if not visible:
        return 0.5, 0.5, 1.0, 1.0

    xs = [x for x, _ in visible]
    ys = [y for _, y in visible]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad_x = pad * width
    pad_y = pad * height
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(width), x2 + pad_x)
    y2 = min(float(height), y2 + pad_y)

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def convert_split(coco_dir: Path, split: str, out_dir: Path, link_images: bool) -> int:
    src = coco_dir / split
    data = _load_coco(src)
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id, meta in images.items():
        file_name = meta["file_name"]
        src_img = src / file_name
        if not src_img.exists():
            continue

        dst_img = img_out / file_name
        if link_images:
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())
        else:
            shutil.copy2(src_img, dst_img)

        w, h = int(meta["width"]), int(meta["height"])
        keypoints = _keypoints_for_image(anns_by_image.get(image_id, []), image_id)
        if not any(v > 0 for _, _, v in keypoints):
            continue

        xc, yc, bw, bh = _bbox_from_keypoints(keypoints, w, h)
        parts = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]
        for x, y, v in keypoints:
            parts.append(f"{x / w:.6f} {y / h:.6f} {v}")
        (lbl_out / f"{Path(file_name).stem}.txt").write_text(" ".join(parts) + "\n", encoding="utf-8")
        written += 1

    return written


def write_data_yaml(out_dir: Path, yaml_path: Path) -> None:
  content = f"""# Padel court keypoint dataset (YOLO pose)
path: {out_dir.resolve()}
train: images/train
val: images/valid
test: images/test

names:
  0: court

kpt_shape: [26, 3]
flip_idx: []

keypoint_names:
{chr(10).join(f"  - {n}" for n in KEYPOINT_NAMES)}

skeleton:
{chr(10).join(f"  - {pair}" for pair in SKELETON)}
"""
  yaml_path.parent.mkdir(parents=True, exist_ok=True)
  yaml_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Padel Court Detection COCO to YOLO pose")
    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "Padel Court Detection.v1i.coco",
        help="Roboflow COCO export directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "court_keypoints_yolo",
        help="Output YOLO dataset directory",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (default: symlink)",
    )
    args = parser.parse_args()

    if not args.coco_dir.exists():
        raise SystemExit(f"COCO dataset not found: {args.coco_dir}")

    totals = {}
    for split in ("train", "valid", "test"):
        n = convert_split(args.coco_dir, split, args.out_dir, link_images=not args.copy_images)
        totals[split] = n
        print(f"{split}: {n} images")

    yaml_path = args.out_dir / "data.yaml"
    write_data_yaml(args.out_dir, yaml_path)
    print(f"Wrote {yaml_path}")
    print(f"Total: {sum(totals.values())} labeled images")


if __name__ == "__main__":
    main()
