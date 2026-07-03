#!/usr/bin/env python3
"""
Merge all CEPDOF scene COCO JSONs into a single ``all.json``,
then split into train/val subsets.

Usage:
    python tools/merge_cepdof.py \\
        --ann-dir datasets/CEPDOF/annotations \\
        --train-ratio 0.8 \\
        --seed 42
"""

import os, sys, json, argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Merge CEPDOF scene JSONs into all.json and split train/val")
    parser.add_argument("--ann-dir", default="datasets/CEPDOF/annotations")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ann_dir = args.ann_dir

    # ------------------------------------------------------------------
    # Step 1: Merge all scene JSONs into all.json
    # ------------------------------------------------------------------
    merged = {"images": [], "annotations": [], "categories": []}
    cat_set = set()
    json_files = sorted(f for f in os.listdir(ann_dir)
                        if f.endswith(".json") and f not in ("all.json", "train.json", "val.json"))

    if not json_files:
        print("No scene JSON files to merge.  Refusing to overwrite train/val from all.json.")
        print(f"Use --ann-dir to point at the CEPDOF annotations folder.")
        sys.exit(1)

    for f in json_files:
        path = os.path.join(ann_dir, f)
        with open(path) as fh:
            data = json.load(fh)

        id_offset = max((img["id"] for img in merged["images"]), default=0)
        ann_id_offset = max((a["id"] for a in merged["annotations"]), default=0)

        old_to_new = {}
        for img in data["images"]:
            old_id = img["id"]
            new_id = old_id + id_offset
            old_to_new[old_id] = new_id
            img["id"] = new_id
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann["id"] = ann["id"] + ann_id_offset
            ann["image_id"] = old_to_new[ann["image_id"]]
            merged["annotations"].append(ann)

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

    # ------------------------------------------------------------------
    # Step 2: Split into train / val
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_total = len(merged["images"])
    indices = rng.permutation(n_total).tolist()
    n_train = int(n_total * args.train_ratio)

    train_img_ids = {merged["images"][i]["id"] for i in indices[:n_train]}
    val_img_ids   = {merged["images"][i]["id"] for i in indices[n_train:]}

    train_data = {"images": [], "annotations": [], "categories": merged["categories"]}
    val_data   = {"images": [], "annotations": [], "categories": merged["categories"]}

    for img in merged["images"]:
        if img["id"] in train_img_ids:
            train_data["images"].append(img)
        else:
            val_data["images"].append(img)

    for ann in merged["annotations"]:
        if ann["image_id"] in train_img_ids:
            train_data["annotations"].append(ann)
        else:
            val_data["annotations"].append(ann)

    train_path = os.path.join(ann_dir, "train.json")
    val_path   = os.path.join(ann_dir, "val.json")
    json.dump(train_data, open(train_path, "w"), indent=2)
    json.dump(val_data,   open(val_path, "w"),   indent=2)

    print(f"Train: {len(train_data['images'])} images, {len(train_data['annotations'])} anns → {train_path}")
    print(f"Val:   {len(val_data['images'])} images, {len(val_data['annotations'])} anns → {val_path}")
    print("Done.")


if __name__ == "__main__":
    main()
