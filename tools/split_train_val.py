#!/usr/bin/env python3
"""
Split a merged COCO all.json into train / val / test subsets.

Usage:
    # Default: 70% train, 20% val, 10% test
    python tools/split_train_val.py --ann datasets/HABBOF/annotations/all.json

    # Custom ratios
    python tools/split_train_val.py --ann datasets/CEPDOF/annotations/all.json \\
        --train 0.7 --val 0.2 --test 0.1

    # Without test set (only train/val)
    python tools/split_train_val.py --ann datasets/FishEye8k/annotations/all.json \\
        --train 0.8 --test 0.0
"""

import os
import sys
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Split a COCO all.json into train/val/test")
    parser.add_argument("--ann", required=True,
                        help="Path to the merged COCO JSON (e.g. all.json)")
    parser.add_argument("--train", type=float, default=0.7,
                        help="Train ratio (default 0.7)")
    parser.add_argument("--val", type=float, default=0.2,
                        help="Validation ratio (default 0.2)")
    parser.add_argument("--test", type=float, default=0.1,
                        help="Test ratio (default 0.1). Set to 0 to disable.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train + args.val + args.test
    assert abs(total_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got train={args.train} + val={args.val} "
        f"+ test={args.test} = {total_ratio}")

    # Load
    with open(args.ann) as f:
        coco = json.load(f)

    ann_dir = os.path.dirname(os.path.abspath(args.ann))
    images = coco["images"]
    categories = coco.get("categories", [])

    # Index annotations by image_id
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Only include images that have annotations
    valid_images = [img for img in images if img["id"] in anns_by_img]
    print(f"  {len(images)} images total, {len(valid_images)} have annotations")

    # Shuffle
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(valid_images)).tolist()

    n_train = int(len(valid_images) * args.train)
    n_val   = int(len(valid_images) * args.val)

    train_ids = {valid_images[i]["id"] for i in indices[:n_train]}
    val_ids   = {valid_images[i]["id"] for i in indices[n_train:n_train + n_val]}
    test_ids  = {valid_images[i]["id"] for i in indices[n_train + n_val:]}

    splits = {
        "train": (train_ids, f"{args.train:.0%}"),
        "val":   (val_ids,   f"{args.val:.0%}"),
        "test":  (test_ids,  f"{args.test:.0%}"),
    }

    for split_name, (id_set, pct) in splits.items():
        if not id_set:
            continue
        split_data = {
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        for img in valid_images:
            if img["id"] in id_set:
                split_data["images"].append(img)

        for img in split_data["images"]:
            for ann in anns_by_img.get(img["id"], []):
                split_data["annotations"].append(ann)

        out_path = os.path.join(ann_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)

        n_imgs = len(split_data["images"])
        n_anns = len(split_data["annotations"])
        print(f"  {split_name} ({pct}): {n_imgs} images, {n_anns} annotations → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
