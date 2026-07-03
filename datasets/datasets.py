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

To apply PFDAug distortion augmentation, use:
    python tools/pfdaug.py

Returns (image, boxes, labels) where:
  - image:  (C, H, W) float32 tensor, normalized to [0, 1]
  - boxes:  (N, 5) float32 tensor of [cx, cy, w, h, R] in pixels
  - labels: (N,) int64 tensor of class IDs
"""

import os
import json
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BACKEND

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


# ===================================================================
#  Unified Fisheye Dataset
# ===================================================================

class FisheyeDataset(Dataset):
    """COCO-style rotated-bbox dataset for fisheye images.

    Pure data loader — no augmentation.  Use ``tools/pfdaug.py`` to
    generate augmented datasets offline::

        python tools/pfdaug.py --ann datasets/HABBOF/annotations/all.json \\
                               --out datasets/HABBOF_PFDAug --k 0.5

    Args:
        ann_file:  Path to the COCO-style JSON annotation file.
        root_dir:  Root directory where ``file_name`` paths are relative to.
                   Auto-detected if ``None``.
        transform: Optional callable applied to PIL image before tensor conversion.
    """

    def __init__(self, ann_file, root_dir=None, transform=None):
        super().__init__()
        with open(ann_file) as f:
            coco = json.load(f)

        # If annotations live in e.g. HABBOF/annotations/, root is HABBOF/
        ann_dir = os.path.dirname(os.path.abspath(ann_file))
        if os.path.basename(ann_dir) == "annotations":
            self.root_dir = root_dir or os.path.dirname(ann_dir)
        else:
            self.root_dir = root_dir or ann_dir
        self.transform = transform

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

        boxes_arr = (
            np.array(boxes, dtype=np.float32) if boxes
            else np.empty((0, 5), dtype=np.float32)
        )
        labels_arr = (
            np.array(labels, dtype=np.int64) if labels
            else np.empty((0,), dtype=np.int64)
        )

        img_tensor = to_tensor(_pil_to_float_tensor(image), dtype=FLOAT_DTYPE)
        boxes_tensor = to_tensor(boxes_arr, dtype=FLOAT_DTYPE)
        labels_tensor = to_tensor(labels_arr, dtype=LONG_DTYPE)

        return to_container(img_tensor, boxes_tensor, labels_tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={len(self)}, classes={len(self._cats)})"


# ===================================================================
#  Factory
# ===================================================================

DATASET_PRESETS = {
    "habbof":     "datasets/HABBOF/annotations/all.json",
    "habbof-lab1": "datasets/HABBOF/annotations/Lab1.json",
    "habbof-lab2": "datasets/HABBOF/annotations/Lab2.json",
    "habbof-meeting1": "datasets/HABBOF/annotations/Meeting1.json",
    "habbof-meeting2": "datasets/HABBOF/annotations/Meeting2.json",
    "fisheye8k":  "datasets/FishEye8k/annotations/all.json",
    "fisheye8k-train": "datasets/FishEye8k/annotations/train.json",
    "fisheye8k-test":  "datasets/FishEye8k/annotations/test.json",
    "cepdof":     "",
}


def get_dataset(name, ann_file=None, root_dir=None, transform=None, **kwargs):
    """Create a :class:`FisheyeDataset` by name or annotation file.

    Args:
        name:    Dataset key (see ``DATASET_PRESETS``) or path to a COCO JSON.
        ann_file: Override annotation file path.
        root_dir: Override root directory. Auto-detected if ``None``.
        transform: Optional PIL→PIL transform.
        **kwargs:  Forwarded for backward compatibility (ignored).

    Returns:
        FisheyeDataset instance.
    """
    if ann_file is None:
        if name in DATASET_PRESETS:
            ann_file = DATASET_PRESETS[name]
        elif os.path.isfile(name):
            ann_file = name

    assert ann_file and os.path.isfile(ann_file), (
        f"Unknown dataset '{name}'. Options: {list(DATASET_PRESETS.keys())}"
    )

    return FisheyeDataset(
        ann_file=ann_file,
        root_dir=root_dir,
        transform=transform,
    )


# ===================================================================
#  Demo / smoke-test
# ===================================================================

if __name__ == "__main__":
    print(f"Backend: {BACKEND}")
    for name in ["habbof", "fisheye8k", "cepdof"]:
        try:
            if name == "cepdof":
                ann = "datasets/CEPDOF/annotations/All_off.json"
            else:
                ann = DATASET_PRESETS[name]
            ds = FisheyeDataset(ann)
            print(f"\n{ds}")
            img, boxes, labels = ds[0]
            print(f"  image:  {type(img).__name__} shape={img.shape}")
            print(f"  boxes:  {type(boxes).__name__} shape={boxes.shape}")
            print(f"  labels: {type(labels).__name__} shape={labels.shape}")
        except Exception as e:
            print(f"\n{name}: SKIPPED — {e}")
