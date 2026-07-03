#!/usr/bin/env python3
"""
Convert HABBOF dataset to COCO-style JSON annotations.

HABBOF format (per-image .txt):
    person cx cy w h R

Output COCO format (per-subdir .json):
    {
      "images": [{"file_name": str, "id": int, "width": int, "height": int}],
      "annotations": [
        {"id": int, "image_id": int, "category_id": int,
         "bbox": [cx, cy, w, h, R], "area": float, "iscrowd": 0}
      ],
      "categories": [{"id": 1, "name": "person"}]
    }

Usage:
    python tools/convert_habbof.py [--root datasets/HABBOF] [--out datasets/HABBOF/annotations]
"""

import os
import sys
import json
import argparse
import glob
from PIL import Image


SUBDIRS = ["Lab1", "Lab2", "Meeting1", "Meeting2"]
CATEGORIES = [{"id": 1, "name": "person"}]


def convert_one_subdir(root: str, subdir: str, out_dir: str, start_img_id: int, start_ann_id: int):
    sdir = os.path.join(root, subdir)
    if not os.path.isdir(sdir):
        print(f"  SKIP: {sdir} not found")
        return start_img_id, start_ann_id

    images = []
    annotations = []
    img_id = start_img_id
    ann_id = start_ann_id

    jpg_files = sorted(glob.glob(os.path.join(sdir, "*.jpg")))
    for jpg_path in jpg_files:
        base = os.path.splitext(os.path.basename(jpg_path))[0]
        txt_path = os.path.join(sdir, f"{base}.txt")
        if not os.path.exists(txt_path):
            continue

        # Read image dimensions
        with Image.open(jpg_path) as im:
            width, height = im.size

        images.append({
            "file_name": os.path.join(subdir, os.path.basename(jpg_path)),
            "id": img_id,
            "width": width,
            "height": height,
        })

        # Read annotations
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    _, cx, cy, w, h, R = parts[0], *map(float, parts[1:6])
                except (ValueError, IndexError):
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,  # person
                    "bbox": [cx, cy, w, h, R],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    # Write COCO JSON for this subdir
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    out_path = os.path.join(out_dir, f"{subdir}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  {subdir}: {len(images)} images, {len(annotations)} annotations → {out_path}")
    return img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Convert HABBOF to COCO format")
    parser.add_argument("--root", default="datasets/HABBOF")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out or os.path.join(root, "annotations")

    print(f"Converting HABBOF: {root} → {out_dir}/")
    img_id, ann_id = 1, 1
    for subdir in SUBDIRS:
        img_id, ann_id = convert_one_subdir(root, subdir, out_dir, img_id, ann_id)

    # Write a merged all-in-one file
    merged = {"images": [], "annotations": [], "categories": CATEGORIES}
    for subdir in SUBDIRS:
        sub_path = os.path.join(out_dir, f"{subdir}.json")
        if os.path.exists(sub_path):
            with open(sub_path) as f:
                data = json.load(f)
            merged["images"].extend(data["images"])
            merged["annotations"].extend(data["annotations"])

    merged_path = os.path.join(out_dir, "all.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged: {len(merged['images'])} images, {len(merged['annotations'])} annotations → {merged_path}")
    print("Done.")


if __name__ == "__main__":
    main()
