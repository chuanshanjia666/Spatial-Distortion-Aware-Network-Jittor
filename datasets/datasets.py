"""
Unified Fisheye Dataset for Spatial-Distortion-Aware Network.

Supports both PyTorch and Jittor backends (controlled by config.BACKEND).
All datasets are expected in COCO-style rotated-bbox JSON format::

    {
      "images": [{"file_name": str, "id": int, "width": int, "height": int}],
      "annotations": [
        {"id": int, "image_id": int, "category_id": int,
         "bbox": [cx, cy, w, h, R], "area": float, "iscrowd": 0}
      ],
      "categories": [{"id": int, "name": str}]
    }

To convert raw datasets to this format, use:
    python tools/convert_habbof.py
    python tools/convert_fisheye8k.py

PFDAug is applied online (in ``__getitem__``) when the config enables it
and the dataset is in training mode.  See ``config.PFDAUG_ENABLED``.

Returns (image, boxes, labels) where:
  - image:  (C, H, W) float32 tensor, normalized to [0, 1]
  - boxes:  (N, 5) float32 tensor of [cx, cy, w, h, R] in pixels
  - labels: (N,) int64 tensor of class IDs
"""

import os
import json
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BACKEND, INPUT_SIZE, PFDAUG_ENABLED, PFDAUG_K, PFDAUG_P
from datasets.pfdaug import PFDAug

# ---------------------------------------------------------------------------
# Backend-agnostic helpers
# ---------------------------------------------------------------------------

if BACKEND == "pytorch":
    import torch
    from torch.utils.data import Dataset

    FLOAT_DTYPE = torch.float32
    LONG_DTYPE = torch.long

    def to_tensor(data, dtype=None):
        return torch.tensor(data, dtype=dtype)

    def to_container(image_tensor, boxes_tensor, labels_tensor):
        return image_tensor, boxes_tensor, labels_tensor

else:  # jittor
    import jittor as jt
    from jittor.dataset import Dataset

    FLOAT_DTYPE = jt.float32
    LONG_DTYPE = jt.int64

    def to_tensor(data, dtype=None):
        return jt.array(data, dtype=dtype)

    def to_container(image_tensor, boxes_tensor, labels_tensor):
        return image_tensor, boxes_tensor, labels_tensor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pil_to_float_tensor(pil_img):
    """Convert a PIL image to a (C, H, W) float32 array normalized to [0, 1]."""
    arr = np.array(pil_img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.transpose(2, 0, 1) / 255.0


def _letterbox_resize(image_np, boxes_np, target_size):
    """Resize image with uniform scaling + padding to square.

    Both image and boxes are uniformly scaled so that the original
    aspect ratio is preserved.  Rotated-box angles are NOT altered
    because uniform scaling doesn't change rotation.

    Args:
        image_np:   (H, W, 3) uint8 numpy array.
        boxes_np:   (N, 5) float32 [cx, cy, w, h, R] in original pixels.
        target_size: int, output square size.

    Returns:
        (image, boxes) — image is (target_size, target_size, 3) uint8,
        boxes in target_size pixel space.
    """
    H, W = image_np.shape[:2]
    scale = target_size / max(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    # Uniform resize
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target_size (top-left aligned)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w, :] = resized

    # Scale boxes uniformly
    if boxes_np.shape[0] > 0:
        boxes_np = boxes_np.copy()
        boxes_np[:, 0] *= scale
        boxes_np[:, 1] *= scale
        boxes_np[:, 2] *= scale
        boxes_np[:, 3] *= scale
        # R unchanged — uniform scaling preserves angle

    return canvas, boxes_np


# ===================================================================
#  Unified Fisheye Dataset
# ===================================================================

class FisheyeDataset(Dataset):
    """COCO-style rotated-bbox dataset for fisheye images.

    PFDAug is applied online in ``__getitem__`` when ``split="train"``.
    ``split="val"`` and ``split="test"`` always use raw images.

    The three splits are read from separate annotation files underneath
    the same annotations directory::

        annotations/
          train.json
          val.json
          test.json

    Use ``tools/split_train_val.py`` to generate them from all.json.

    Args:
        name:     Dataset preset name (see ``DATASET_PRESETS``) or direct
                  path to a COCO JSON.  When using splits, pass the preset
                  name and the split is appended automatically.
        split:    ``"train"`` | ``"val"`` | ``"test"``.
        root_dir: Override root directory. Auto-detected if ``None``.
        transform: Optional callable applied to PIL image before tensor.
        seed:     Random seed for reproducibility.
    """

    def __init__(self, name="habbof", split="train", root_dir=None,
                 transform=None, seed=42):
        super().__init__()

        # Resolve annotation file
        ann_file = _resolve_ann(name, split)

        with open(ann_file) as f:
            coco = json.load(f)

        ann_dir = os.path.dirname(os.path.abspath(ann_file))
        if os.path.basename(ann_dir) == "annotations":
            self.root_dir = root_dir or os.path.dirname(ann_dir)
        else:
            self.root_dir = root_dir or ann_dir
        self.transform = transform
        self.split = split

        # Index
        self._images = {img["id"]: img for img in coco["images"]}
        self._cats = {c["id"]: c["name"] for c in coco["categories"]}

        self._anns_by_image = {}
        for ann in coco["annotations"]:
            self._anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self._entries = [
            (img["id"], img["file_name"])
            for img in coco["images"]
            if img["id"] in self._anns_by_image
        ]

        assert len(self._entries) > 0, f"No valid entries in {ann_file}"

        # PFDAug — online, training only, controlled by config
        if PFDAUG_ENABLED and split == "train":
            self.pfdaug = PFDAug(k=PFDAUG_K, p=PFDAUG_P, seed=seed)
        else:
            self.pfdaug = None

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self):
        return len(self._cats)

    @property
    def class_names(self):
        return [self._cats[i] for i in sorted(self._cats)]

    # ------------------------------------------------------------------
    #  Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        img_id, file_name = self._entries[idx]
        anns = self._anns_by_image.get(img_id, [])

        # Image
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        # Annotations
        boxes = []
        labels = []
        for a in anns:
            bbox = a["bbox"]
            if len(bbox) >= 5:
                boxes.append(bbox[:5])
            else:
                x, y, w, h = bbox[:4]
                boxes.append([x + w / 2, y + h / 2, w, h, 0.0])
            labels.append(a.get("category_id", 1) - 1)

        if self.transform is not None:
            image = self.transform(image)

        # ---- Online PFDAug (train only, numpy → numpy) ----
        if self.pfdaug is not None and len(boxes) > 0:
            image_np = np.array(image)
            boxes_np = np.array(boxes, dtype=np.float32)
            image_np, boxes_np = self.pfdaug(image_np, boxes_np)
            image = Image.fromarray(image_np)
            boxes = boxes_np.tolist()

        # ---- Uniform resize + letterbox to INPUT_SIZE ----
        image_np = np.array(image)
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 5), dtype=np.float32)
        image_np, boxes_np = _letterbox_resize(image_np, boxes_np, INPUT_SIZE)

        boxes_arr = boxes_np.astype(np.float32)
        labels_arr = (
            np.array(labels, dtype=np.int64) if labels
            else np.empty((0,), dtype=np.int64)
        )

        img_tensor = to_tensor(_pil_to_float_tensor(Image.fromarray(image_np)), dtype=FLOAT_DTYPE)
        boxes_tensor = to_tensor(boxes_arr, dtype=FLOAT_DTYPE)
        labels_tensor = to_tensor(labels_arr, dtype=LONG_DTYPE)

        return to_container(img_tensor, boxes_tensor, labels_tensor)

    def __repr__(self):
        s = f"{self.__class__.__name__}(n={len(self)}, split={self.split}, "
        s += f"classes={len(self._cats)})"
        if self.pfdaug is not None:
            s += f" +PFDAug(k={self.pfdaug.k}, p={self.pfdaug.p})"
        return s


# ===================================================================
#  Presets & resolution
# ===================================================================

# Each preset points to an *annotations directory*.  The actual annotation
# file is ``{preset_path}/{split}.json``.
DATASET_PRESETS = {
    "habbof":     "datasets/HABBOF/annotations",
    "fisheye8k":  "datasets/FishEye8k/annotations",
    "cepdof":     "datasets/CEPDOF/annotations",
    "wepdtof":    "datasets/WEPDTOF/annotations",
}

# Presets for single-file datasets (no split → one file per scene).
# These are passed as ann_file directly.
SINGLE_FILE_PRESETS = {
    "habbof-all":     "datasets/HABBOF/annotations/all.json",
    "habbof-train":   "datasets/HABBOF/annotations/train.json",
    "habbof-val":     "datasets/HABBOF/annotations/val.json",
    "habbof-test":    "datasets/HABBOF/annotations/test.json",
    "fisheye8k-all":  "datasets/FishEye8k/annotations/all.json",
    "fisheye8k-train":"datasets/FishEye8k/annotations/train.json",
    # "fisheye8k-val":  "datasets/FishEye8k/annotations/val.json",
    "fisheye8k-test": "datasets/FishEye8k/annotations/test.json",
    "cepdof-all":     "datasets/CEPDOF/annotations/all.json",
    "cepdof-train":   "datasets/CEPDOF/annotations/train.json",
    "cepdof-val":     "datasets/CEPDOF/annotations/val.json",
    "cepdof-test":    "datasets/CEPDOF/annotations/test.json",
    "wepdtof-all":    "datasets/WEPDTOF/annotations/all.json",
    "wepdtof-train":  "datasets/WEPDTOF/annotations/train.json",
    "wepdtof-val":    "datasets/WEPDTOF/annotations/val.json",
    "wepdtof-test":   "datasets/WEPDTOF/annotations/test.json",
}


def _resolve_ann(name: str, split: str) -> str:
    """Resolve a preset name + split to an annotation file path.

    1. If ``name`` is a path to an existing file, return it directly.
    2. Try ``SINGLE_FILE_PRESETS[name-split]`` (e.g. "habbof-all").
    3. Try ``name/split.json`` (e.g. if name is a directory path).
    4. Try ``<preset_dir>/split.json`` (e.g. if name is a DATASET_PRESETS key).
    """
    # Direct file path
    if name.endswith(".json") and os.path.isfile(name):
        return name

    # Single-file preset: "habbof-all", "wepdtof-test", etc.
    combo = f"{name}-{split}"
    if combo in SINGLE_FILE_PRESETS:
        return SINGLE_FILE_PRESETS[combo]

    # Directory preset → <dir>/<split>.json
    if name in DATASET_PRESETS:
        ann_dir = DATASET_PRESETS[name]
    else:
        # Last resort: treat name as a directory path
        ann_dir = name

    ann_file = os.path.join(ann_dir, f"{split}.json")
    if os.path.isfile(ann_file):
        return ann_file

    raise FileNotFoundError(
        f"Cannot resolve dataset '{name}' split '{split}': "
        f"{ann_file} not found.  Run tools/split_train_val.py first."
    )


def _parse_ds_spec(spec: str):
    """Parse a ``preset[split]`` spec, e.g. ``"habbof[all]"`` → ("habbof", "all").

    If no brackets, defaults split to ``"all"``.  Plain file paths are returned as-is
    with split ``"all"``.
    """
    spec = spec.strip()
    if "[" in spec and spec.endswith("]"):
        pos = spec.index("[")
        name = spec[:pos]
        split = spec[pos + 1:-1]
        return name, split
    return spec, "all"


# ===================================================================
#  Factory
# ===================================================================

def get_dataset(spec="habbof[all]", root_dir=None, transform=None, seed=42):
    """Create a :class:`FisheyeDataset` from a ``preset[split]`` spec.

    Args:
        spec:      Dataset spec like ``"habbof[all]"``, ``"cepdof[train]"``,
                   or a plain path to a COCO JSON.
        root_dir:  Override root directory. Auto-detected if ``None``.
        transform: Optional PIL→PIL transform.
        seed:      RNG seed.

    Returns:
        FisheyeDataset instance.

    Examples::

        ds = get_dataset("habbof[all]")
        ds = get_dataset("cepdof[train]")
        ds = get_dataset("wepdtof[test]")
        ds = get_dataset("datasets/HABBOF/annotations/all.json")
    """
    name, split = _parse_ds_spec(spec)
    return FisheyeDataset(
        name=name,
        split=split,
        root_dir=root_dir,
        transform=transform,
        seed=seed,
    )


# ===================================================================
#  Demo / smoke-test
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset smoke-test / visualization")
    parser.add_argument("--vis", type=str, nargs="?", const="datasets/vis",
                        metavar="DIR",
                        help="Visualize first 8 samples and save to DIR")
    parser.add_argument("--spec", type=str, default="habbof[train]",
                        help="Dataset spec, e.g. habbof[train]")
    args = parser.parse_args()

    print(f"Backend: {BACKEND}")
    print(f"PFDAug: enabled={PFDAUG_ENABLED}, k={PFDAUG_K}, p={PFDAUG_P}")
    print(f"Input size: {INPUT_SIZE}\n")

    ds = get_dataset(args.spec)
    n = len(ds)
    print(f"  {args.spec}: {n} samples, {ds.num_classes} classes")
    total_boxes = sum(len(v) for v in ds._anns_by_image.values())
    print(f"  total annotations: {total_boxes} ({total_boxes / n:.1f}/sample)")

    largest_img = max(ds._images.items(), key=lambda x: x[1].get("width", 0) * x[1].get("height", 0))
    info = largest_img[1]
    print(f"  largest image: {info['width']}×{info['height']}")

    # ---- Visualization ----
    if args.vis:
        os.makedirs(args.vis, exist_ok=True)
        print(f"\n  Visualizing first 8 samples → {args.vis}/")
        for i in range(min(8, n)):
            img_tensor, boxes_tensor, labels_tensor = ds[i]
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
            boxes = boxes_tensor.cpu().numpy()

            for (cx, cy, w, h, R) in boxes:
                pts = cv2.boxPoints(((float(cx), float(cy)),
                                     (float(w), float(h)), float(R)))
                cv2.drawContours(img, [np.intp(pts)], 0, (0, 255, 0), 2)

            cv2.putText(img, f"#{i}  {len(boxes)} boxes", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out_path = os.path.join(args.vis, f"dataset_sample_{i:02d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("  Done.")
    else:
        # Quick smoke: check first sample
        img, boxes, labels = ds[0]
        if BACKEND == "pytorch":
            print(f"  sample 0: img={tuple(img.shape)}, boxes={boxes.shape}, labels={labels.shape}")
        else:
            print(f"  sample 0: img={img.shape}, boxes={boxes.shape}, labels={labels.shape}")
