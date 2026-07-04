#!/usr/bin/env python3
"""
Merge all WEPDTOF scene COCO JSONs into a single ``all.json``.

Usage:
    python tools/merge_wepdtof.py --ann-dir datasets/WEPDTOF/annotations
"""

import os
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Merge WEPDTOF scene JSONs into all.json")
    parser.add_argument("--ann-dir", default="datasets/WEPDTOF/annotations")
    args = parser.parse_args()

    ann_dir = args.ann_dir

    merged = {"images": [], "annotations": [], "categories": []}
    cat_set = set()

    json_files = sorted(
        f for f in os.listdir(ann_dir)
        if f.endswith(".json") and f not in ("all.json", "train.json", "val.json", "test.json")
    )

    if not json_files:
        print("No scene JSON files to merge.")
        sys.exit(1)

    img_id_counter = 1
    ann_id_counter = 1

    for fname in json_files:
        scene_name = os.path.splitext(fname)[0]
        path = os.path.join(ann_dir, fname)
        with open(path) as fh:
            data = json.load(fh)

        # Map original string image_id → new integer id
        old_to_new = {}
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = img_id_counter
            old_to_new[str(old_id)] = new_id
            img_id_counter += 1

            merged["images"].append({
                "id": new_id,
                "file_name": os.path.join("frames", scene_name,
                                          img.get("file_name", f"{old_id}.jpg")),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            })

        for ann in data.get("annotations", []):
            merged["annotations"].append({
                "id": ann_id_counter,
                "image_id": old_to_new[str(ann["image_id"])],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],          # [cx, cy, w, h, R]
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id_counter += 1

        for cat in data.get("categories", []):
            if cat["id"] not in cat_set:
                cat_set.add(cat["id"])
                merged["categories"].append(cat)

    merged["categories"].sort(key=lambda c: c["id"])

    all_path = os.path.join(ann_dir, "all.json")
    with open(all_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged: {len(merged['images'])} images, "
          f"{len(merged['annotations'])} annotations → {all_path}")
    print(f"Categories: {merged['categories']}")
    print("Done.  Run split_train_val.py next to create train/val/test splits.")


if __name__ == "__main__":
    main()
