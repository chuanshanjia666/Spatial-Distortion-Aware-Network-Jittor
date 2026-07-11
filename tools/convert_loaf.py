#!/usr/bin/env python3
"""
Convert LOAF dataset annotations to unified COCO-style JSON.

LOAF format (per resolution):
    - instances_train.json, instances_val.json, instances_test.json
    - Each annotation has "bbox" (axis-aligned) and "rotated_box" (rotated) fields

Output COCO format:
    {
      "images": [{"file_name": str, "id": int, "width": int, "height": int}],
      "annotations": [
        {"id": int, "image_id": int, "category_id": int,
         "bbox": [cx, cy, w, h, R], "area": float, "iscrowd": 0}
      ],
      "categories": [{"id": 1, "name": "person"}]
    }

Usage:
    python tools/convert_loaf.py [--root datasets/LOAF] [--resolution resolution_512]
"""

import os
import sys
import json
import argparse

CATEGORIES = [{"id": 1, "name": "person", "supercategory": "person"}]

SPLITS = ["train", "val", "test"]


def convert_loaf(root: str, resolution: str, out_dir: str):
    """Convert LOAF annotations for a specific resolution to COCO format."""
    ann_root = os.path.join(root, "annotations", resolution)
    img_root = os.path.join(root, resolution)

    if not os.path.isdir(ann_root):
        print(f"  SKIP: {ann_root} not found")
        return

    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    img_id_counter = 1
    ann_id_counter = 1

    for split in SPLITS:
        in_file = os.path.join(ann_root, f"instances_{split}.json")
        if not os.path.exists(in_file):
            print(f"  SKIP: {in_file} not found")
            continue

        with open(in_file) as f:
            data = json.load(f)

        images = []
        annotations = []

        # Build image id mapping
        image_id_map = {img["id"]: img for img in data["images"]}

        for ann in data["annotations"]:
            # Use rotated_box if available, otherwise fall back to bbox
            if "rotated_box" in ann and len(ann["rotated_box"]) >= 5:
                cx, cy, w, h, angle = ann["rotated_box"][:5]
            elif "bbox" in ann and len(ann["bbox"]) >= 4:
                x, y, bw, bh = ann["bbox"][:4]
                cx = x + bw / 2
                cy = y + bh / 2
                w, h, angle = bw, bh, 0.0
            else:
                continue

            # Get image info
            image_id = ann["image_id"]
            if image_id not in image_id_map:
                continue
            img_info = image_id_map[image_id]

            # Add image if not already added
            img_file_name = img_info["file_name"]
            # e.g., "0000_00105.jpg" -> "resolution_512/train/0000_00105.jpg"
            img_file_name_with_path = f"{resolution}/{split}/{img_file_name}"
            existing_img = next((img for img in images if img["id"] == image_id), None)
            if existing_img is None:
                images.append({
                    "file_name": img_file_name_with_path,
                    "id": image_id,
                    "width": img_info["width"],
                    "height": img_info["height"],
                })

            # Convert angle from degrees to radians if needed (LOAF uses degrees)
            # Check: LOAF rotated_box angle is in degrees
            angle_rad = angle if abs(angle) > 3.14 else angle

            annotations.append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": ann.get("category_id", 1),
                "bbox": [round(cx, 4), round(cy, 4), round(w, 4), round(h, 4), round(angle, 4)],
                "area": round(w * h, 4),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id_counter += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES,
        }
        out_path = os.path.join(out_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        summary[split] = (len(images), len(annotations))
        print(f"  {split}: {len(images)} images, {len(annotations)} annotations → {out_path}")

    # Merged all.json
    merged = {"images": [], "annotations": [], "categories": CATEGORIES}
    img_id_counter = 1
    ann_id_counter = 1
    for split in SPLITS:
        split_path = os.path.join(out_dir, f"{split}.json")
        if os.path.exists(split_path):
            with open(split_path) as f:
                d = json.load(f)
            # Remap ids for merged dataset
            old_img_id_to_new = {}
            for img in d["images"]:
                old_img_id_to_new[img["id"]] = img_id_counter
                img["id"] = img_id_counter
                img_id_counter += 1

            for ann in d["annotations"]:
                ann["id"] = ann_id_counter
                ann["image_id"] = old_img_id_to_new[ann["image_id"]]
                ann_id_counter += 1

            merged["images"].extend(d["images"])
            merged["annotations"].extend(d["annotations"])

    merged_path = os.path.join(out_dir, "all.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  all: {len(merged['images'])} images, {len(merged['annotations'])} annotations → {merged_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert LOAF to COCO format")
    parser.add_argument("--root", default="datasets/LOAF")
    parser.add_argument("--resolution", default="resolution_512",
                        choices=["resolution_1k", "resolution_2k", "resolution_3k", "resolution_512"],
                        help="LOAF resolution to convert")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = args.root
    resolution = args.resolution
    # Output to datasets/LOAF/annotations/ (same level as other datasets)
    out_dir = args.out or os.path.join(root, "annotations")

    print(f"Converting LOAF ({resolution}): {root} → {out_dir}/")
    convert_loaf(root, resolution, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()