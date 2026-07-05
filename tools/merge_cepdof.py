#!/usr/bin/env python3
"""
Merge all CEPDOF scene COCO JSONs into a single ``all.json``.

Also fixes casing mismatches in ``file_name`` (e.g. ``Edge_Cases`` vs
``Edge_cases``) so that paths resolve correctly on case-sensitive
filesystems (Linux).

Usage:
    python tools/merge_cepdof.py --ann-dir datasets/CEPDOF/annotations
"""

import os
import sys
import json
import argparse


def _find_real_path(root_dir: str, file_name: str):
    """Return the real disk path for *file_name*, fixing casing mismatches.

    Walks up the path components and does a case-insensitive match at each
    level, returning the correctly-cased path relative to *root_dir*.
    """
    parts = file_name.replace("\\", "/").split("/")
    real_parts = []
    cur = root_dir
    for part in parts:
        try:
            entries = os.listdir(cur)
        except FileNotFoundError:
            return file_name   # bail, keep original
        matched = None
        for e in entries:
            if e == part:
                matched = e
                break
        if matched is not None:
            real_parts.append(matched)
            cur = os.path.join(cur, matched)
            continue
        # Try case-insensitive fallback
        for e in entries:
            if e.lower() == part.lower():
                matched = e
                break
        if matched is not None:
            real_parts.append(matched)
            cur = os.path.join(cur, matched)
        else:
            real_parts.append(part)   # keep original, will fail later
            cur = os.path.join(cur, part)
    return "/".join(real_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Merge CEPDOF scene JSONs into all.json")
    parser.add_argument("--ann-dir", default="datasets/CEPDOF/annotations")
    parser.add_argument("--root", default=None,
                        help="Dataset root directory (auto-detected if None)")
    args = parser.parse_args()

    ann_dir = os.path.abspath(args.ann_dir)
    if args.root:
        root_dir = os.path.abspath(args.root)
    else:
        root_dir = os.path.dirname(ann_dir)

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
    fixed_count = 0

    for fname in json_files:
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

            raw_fn = img.get("file_name", f"{old_id}.jpg")
            fixed_fn = _find_real_path(root_dir, raw_fn)
            if fixed_fn != raw_fn:
                fixed_count += 1

            merged["images"].append({
                "id": new_id,
                "file_name": fixed_fn,
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
    if fixed_count:
        print(f"Casing fixes applied: {fixed_count} file_name(s)")
    print(f"Categories: {merged['categories']}")
    print("Done.  Run split_train_val.py next to create train/val/test splits.")


if __name__ == "__main__":
    main()
