#!/usr/bin/env python3
"""
Offline PFDAug dataset generator.

Applies Prominent Fisheye Distortion Augmentation to a COCO dataset
and saves the augmented dataset to disk.  Unlike the online version in
datasets/pfdaug.py, this produces a static augmented dataset that can
be used without modifying the training loop.

Usage:
    python tools/pfdaug.py \\
        --ann datasets/HABBOF/annotations/train.json \\
        --out datasets/HABBOF_PFDAug \\
        --k 0.5 --p 0.5 --copy-all
"""

import os, sys, json, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.pfdaug import PFDAug


def augment_dataset(ann_file: str, out_dir: str, k: float, split: str = "train",
                    seed: int = 42, copy_all: bool = False, p: float = 0.5):
    with open(ann_file) as f:
        coco = json.load(f)

    ann_dir = os.path.dirname(os.path.abspath(ann_file))
    if os.path.basename(ann_dir) == "annotations":
        src_root = os.path.dirname(ann_dir)
    else:
        src_root = ann_dir

    img_dir = os.path.join(out_dir, "images")
    ann_out_dir = os.path.join(out_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_out_dir, exist_ok=True)

    aug = PFDAug(k=k, p=p, seed=seed)

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    out_images, out_annotations = [], []
    ann_id, n_augmented = 1, 0
    total = len(coco["images"])

    for i, img_info in enumerate(coco["images"]):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        src_path = os.path.join(src_root, file_name)

        image = cv2.imread(src_path)
        if image is None:
            print(f"  WARN: cannot read {src_path}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = anns_by_img.get(img_id, [])
        boxes = np.array([a["bbox"][:5] for a in anns], dtype=np.float32)

        aug_image, aug_boxes = aug(image, boxes)
        was_augmented = not np.array_equal(image, aug_image)
        if was_augmented:
            n_augmented += 1
        elif not copy_all:
            continue

        out_name = file_name
        out_path = os.path.join(img_dir, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        out_images.append({
            "file_name": out_name, "id": img_id,
            "width": img_info["width"], "height": img_info["height"],
        })
        for j, a in enumerate(anns):
            out_annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": a["category_id"],
                "bbox": aug_boxes[j].tolist(),
                "area": float(aug_boxes[j, 2] * aug_boxes[j, 3]),
                "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1

        if (i + 1) % max(1, total // 10) == 0:
            print(f"  progress: {i + 1}/{total}, {n_augmented} augmented")

    out_coco = {"images": out_images, "annotations": out_annotations,
                "categories": coco["categories"]}
    out_ann_path = os.path.join(ann_out_dir, f"{split}.json")
    with open(out_ann_path, "w") as f:
        json.dump(out_coco, f, indent=2)
    print(f"\nDone: {len(out_images)} images ({n_augmented} augmented), "
          f"{len(out_annotations)} annotations → {out_ann_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PFDAug — offline fisheye distortion augmentation for COCO datasets")
    parser.add_argument("--ann", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-all", action="store_true",
                        help="Copy all images (including unaugmented) to output")
    args = parser.parse_args()

    print(f"PFDAug: k={args.k}, p={args.p}, split={args.split}")
    augment_dataset(args.ann, args.out, args.k, args.split,
                    args.seed, args.copy_all, args.p)


if __name__ == "__main__":
    main()
