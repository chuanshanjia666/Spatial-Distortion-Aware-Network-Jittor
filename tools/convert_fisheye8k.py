#!/usr/bin/env python3
"""
Convert FishEye8k dataset to COCO-style JSON annotations.

FishEye8k format (single samples.json):
    {
      "samples": [
        {
          "filepath": "data/camera3_A_0.png",
          "tags": ["train" / "test"],
          "metadata": {"width": ..., "height": ...},
          "detections": {
            "detections": [
              {"label": "Car", "bounding_box": [x, y, w, h]}
            ]
          }
        }
      ]
    }

Output COCO format:
    {
      "images": [{"file_name": str, "id": int, "width": int, "height": int}],
      "annotations": [
        {"id": int, "image_id": int, "category_id": int,
         "bbox": [cx, cy, w, h, 0.0], "area": float, "iscrowd": 0}
      ],
      "categories": [{"id": 1, "name": "Bike"}, ...]
    }

Notes:
  - FishEye8k uses axis-aligned bboxes [x, y, w, h] (top-left corner, normalised).
    We convert them to rotated format [cx, cy, w, h, 0.0] consistent with CEPDOF/HABBOF.
  - The original split tags ("train" / "test") are preserved via separate JSON files.

Usage:
    python tools/convert_fisheye8k.py [--root datasets/FishEye8k] [--out datasets/FishEye8k/annotations]
"""

import os
import sys
import json
import argparse


CATEGORIES = [
    {"id": 1, "name": "Bike"},
    {"id": 2, "name": "Bus"},
    {"id": 3, "name": "Car"},
    {"id": 4, "name": "Pedestrian"},
    {"id": 5, "name": "Truck"},
]
NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def main():
    parser = argparse.ArgumentParser(description="Convert FishEye8k to COCO format")
    parser.add_argument("--root", default="datasets/FishEye8k")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out or os.path.join(root, "annotations")

    samples_path = os.path.join(root, "samples.json")
    with open(samples_path) as f:
        data = json.load(f)

    # Split by tag
    splits = {}  # tag → list of samples
    for s in data["samples"]:
        for tag in s.get("tags", []):
            splits.setdefault(tag, []).append(s)

    summary = {}
    img_id_counter = 1
    ann_id_counter = 1

    os.makedirs(out_dir, exist_ok=True)

    for split_tag, samples in splits.items():
        images = []
        annotations = []

        for sample in samples:
            filepath = sample["filepath"]

            # Dimensions
            meta = sample.get("metadata", {})
            width = meta.get("width")
            height = meta.get("height")
            if width is None or height is None:
                # Fall back to reading the image
                from PIL import Image
                img_path = os.path.join(root, filepath)
                with Image.open(img_path) as im:
                    width, height = im.size

            img_id = img_id_counter
            img_id_counter += 1

            images.append({
                "file_name": filepath,
                "id": img_id,
                "width": width,
                "height": height,
            })

            for det in sample["detections"]["detections"]:
                label = det["label"]
                cat_id = NAME_TO_ID.get(label)
                if cat_id is None:
                    print(f"  WARN: unknown label '{label}' in {filepath}, skipping")
                    continue

                # Normalised [x, y, w, h] (top-left) → pixel [cx, cy, w, h, 0.0]
                nx, ny, nbw, nbh = det["bounding_box"]
                w = nbw * width
                h = nbh * height
                cx = (nx + nbw / 2.0) * width
                cy = (ny + nbh / 2.0) * height

                annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(cx, 4), round(cy, 4), round(w, 4), round(h, 4), 0.0],
                    "area": round(w * h, 4),
                    "iscrowd": 0,
                })
                ann_id_counter += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES,
        }
        out_path = os.path.join(out_dir, f"{split_tag}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        summary[split_tag] = (len(images), len(annotations))
        print(f"  {split_tag}: {len(images)} images, {len(annotations)} annotations → {out_path}")

    # Merged all.json
    merged = {"images": [], "annotations": [], "categories": CATEGORIES}
    for tag in sorted(splits.keys()):
        tag_path = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(tag_path):
            with open(tag_path) as f:
                d = json.load(f)
            merged["images"].extend(d["images"])
            merged["annotations"].extend(d["annotations"])
    merged_path = os.path.join(out_dir, "all.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n  Merged: {len(merged['images'])} images, {len(merged['annotations'])} annotations → {merged_path}")

    print("Done.")


if __name__ == "__main__":
    main()
