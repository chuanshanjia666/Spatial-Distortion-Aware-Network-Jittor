import os
import sys
import json
import argparse
import shutil

import numpy as np
import cv2
from PIL import Image


# ===================================================================
#  Core PFDAug class
# ===================================================================

class PFDAug:
    def __init__(self, k: float = 0.5, p: float = 0.5, seed: int | None = None):
        self.k = k
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        """Apply PFDAug with the configured probability.

        Args:
            image: (H, W, 3) uint8 numpy array.
            boxes: (N, 5) float32 array ``[cx, cy, w, h, R]`` in pixel coords,
                   R in **degrees**.

        Returns:
            (image, boxes) — the augmented pair (may be unchanged
            when the random draw exceeds ``self.p``).
        """
        if self.p < 1.0 and self.rng.random() > self.p:
            return image, boxes
        return self.forward(image, boxes, self.k)

    def forward(self, image: np.ndarray, boxes: np.ndarray, k: float):
        """Deterministic forward pass (always applies distortion)."""
        H, W = image.shape[:2]
        image_distorted = self._warp_image(image, W, H, k)
        boxes_distorted = self._warp_boxes(boxes, W, H, k)
        return image_distorted, boxes_distorted

    # ------------------------------------------------------------------
    # Image warp (barrel distortion via backward mapping + Newton)
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_image(image: np.ndarray, W: int, H: int, k: float) -> np.ndarray:
        """Apply barrel distortion to the whole image.

        Backward mapping: for every output pixel solve for the source pixel
        via Newton's method on  k·r_d³ + r_d - r_u = 0.
        """
        cx, cy = W / 2.0, H / 2.0
        scale = max(cx, cy)

        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        xn = (xs - cx) / scale
        yn = (ys - cy) / scale
        r_u = np.sqrt(xn ** 2 + yn ** 2)

        eps = 1e-8
        valid = r_u > eps
        r_d = r_u.copy()

        for _ in range(5):
            r2 = r_d ** 2
            f = k * r_d * r2 + r_d - r_u
            fprime = 3.0 * k * r2 + 1.0
            r_d = np.where(valid, r_d - f / fprime, r_d)

        ratio = np.ones_like(r_u, dtype=np.float32)
        ratio[valid] = r_d[valid] / r_u[valid]
        xs_src = ratio * xn * scale + cx
        ys_src = ratio * yn * scale + cy

        return cv2.remap(image, xs_src.astype(np.float32), ys_src.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # ------------------------------------------------------------------
    # Box warp — Algorithm 1 from the paper
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_boxes(boxes: np.ndarray, W: int, H: int, k: float) -> np.ndarray:
        """Adjust oriented bounding boxes after barrel distortion.

        For each box: compute four rotated corners, distort each corner
        (Alg. 2), then compute the minimum area bounding rectangle (MBR).
        """
        if len(boxes) == 0:
            return boxes.copy()

        boxes = np.asarray(boxes, dtype=np.float32)
        result = np.empty_like(boxes)

        for i, box in enumerate(boxes):
            cx, cy, w, h, R = box
            R_rad = np.deg2rad(R)
            half_w, half_h = w / 2.0, h / 2.0

            local_corners = np.array([
                [-half_w, -half_h],
                [ half_w, -half_h],
                [-half_w,  half_h],
                [ half_w,  half_h],
            ], dtype=np.float32)

            cos_a, sin_a = np.cos(R_rad), np.sin(R_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            rotated_corners = cx + local_corners @ rot.T

            distorted_corners = np.empty_like(rotated_corners)
            for j in range(4):
                distorted_corners[j] = PFDAug._point_distortion(
                    rotated_corners[j, 0], rotated_corners[j, 1], W, H, k)

            contour = distorted_corners.reshape(-1, 1, 2)
            rect = cv2.minAreaRect(contour)
            (new_cx, new_cy), (new_w, new_h), new_angle = rect

            if new_w < new_h:
                new_w, new_h = new_h, new_w
                new_angle += 90.0
            new_angle = (new_angle + 90.0) % 180.0 - 90.0

            result[i] = [new_cx, new_cy, new_w, new_h, new_angle]

        return result

    # ------------------------------------------------------------------
    # Point distortion — Algorithm 2 from the paper
    # ------------------------------------------------------------------

    @staticmethod
    def _point_distortion(x: float, y: float, W: int, H: int, k: float):
        """Distort a single pixel coordinate (Alg. 2)."""
        halfW, halfH = W / 2.0, H / 2.0

        xn = (x - halfW) / halfW
        yn = (y - halfH) / halfH
        r = np.sqrt(xn * xn + yn * yn)
        theta = np.arctan2(yn, xn)
        r_dis = r * (1.0 + k * r * r)

        x_dis = r_dis * np.cos(theta) * halfW + halfW
        y_dis = r_dis * np.sin(theta) * halfH + halfH

        return np.float32(x_dis), np.float32(y_dis)


def pfdaug_transform(k: float = 0.5, p: float = 0.5) -> PFDAug:
    """Factory that returns a callable :class:`PFDAug` instance."""
    return PFDAug(k=k, p=p)


# ===================================================================
#  CLI: offline dataset augmentation
# ===================================================================

def augment_dataset(ann_file: str, out_dir: str, k: float, split: str = "train",
                    seed: int = 42, copy_unselected: bool = False, p: float = 0.5):
    """Apply PFDAug to a COCO dataset and save augmented images + annotations.

    Args:
        ann_file:  Path to the COCO JSON annotation file.
        out_dir:   Output root directory.  Structure::

                       <out_dir>/
                         annotations/<split>.json
                         images/<file_name>

        k:         Distortion coefficient.
        split:     Split name written into the output JSON filename.
        seed:      RNG seed.
        copy_unselected: If True, images NOT picked by probability p are
                         copied verbatim (with original boxes) into the output.
                         If False (default), only augmented images are saved.
        p:         Probability of applying distortion to each image.

    The output preserves the original ``file_name`` with an optional suffix
    if ``copy_unselected=True``.
    """
    with open(ann_file) as f:
        coco = json.load(f)

    # Resolve root directory (if ann_file is in annotations/, go up one level)
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

    # Index annotations by image_id
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    out_images = []
    out_annotations = []
    ann_id = 1

    total = len(coco["images"])
    n_augmented = 0

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

        # Apply PFDAug
        aug_image, aug_boxes = aug(image, boxes)
        was_augmented = not np.array_equal(image, aug_image)

        if was_augmented:
            n_augmented += 1
        elif not copy_unselected:
            # Skip images where PFDAug probability was not hit
            if (i + 1) % max(1, total // 10) == 0:
                print(f"  progress: {i+1}/{total} images, {n_augmented} augmented so far")
            continue

        # Write image (preserve subdirectory structure from file_name)
        out_name = file_name
        out_path = os.path.join(img_dir, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        # Collect annotations
        out_images.append({
            "file_name": out_name,
            "id": img_id,
            "width": img_info["width"],
            "height": img_info["height"],
        })

        for j, a in enumerate(anns):
            out_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": a["category_id"],
                "bbox": aug_boxes[j].tolist(),
                "area": float(aug_boxes[j, 2] * aug_boxes[j, 3]),
                "iscrowd": 0,
            })
            ann_id += 1

        if (i + 1) % max(1, total // 10) == 0:
            print(f"  progress: {i+1}/{total} images, {n_augmented} augmented so far")

    # Write output COCO JSON
    out_coco = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco["categories"],
    }
    out_ann_path = os.path.join(ann_out_dir, f"{split}.json")
    with open(out_ann_path, "w") as f:
        json.dump(out_coco, f, indent=2)

    print(f"\nDone: {len(out_images)} images ({n_augmented} augmented), "
          f"{len(out_annotations)} annotations → {out_ann_path}")


# ===================================================================
#  CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PFDAug — offline fisheye distortion augmentation for COCO datasets")
    parser.add_argument("--ann", required=True,
                        help="Path to the COCO JSON annotation file")
    parser.add_argument("--out", required=True,
                        help="Output directory root")
    parser.add_argument("--k", type=float, default=0.5,
                        help="Distortion coefficient (default 0.5)")
    parser.add_argument("--p", type=float, default=0.5,
                        help="Probability of distorting each image (default 0.5)")
    parser.add_argument("--split", default="train",
                        help="Split name for the output JSON filename (default 'train')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--copy-all", action="store_true",
                        help="Also copy images not selected by probability p verbatim")
    args = parser.parse_args()

    print(f"PFDAug: k={args.k}, p={args.p}, split={args.split}")
    print(f"  ann = {args.ann}")
    print(f"  out = {args.out}")
    augment_dataset(
        ann_file=args.ann,
        out_dir=args.out,
        k=args.k,
        p=args.p,
        split=args.split,
        seed=args.seed,
        copy_unselected=args.copy_all,
    )


if __name__ == "__main__":
    main()
