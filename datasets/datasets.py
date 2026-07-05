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
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BACKEND, PFDAUG_ENABLED, PFDAUG_K, PFDAUG_P
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
    import sys

    def _sizeof_obj(obj, seen=None):
        """Recursively estimate memory size of a Python object (approximate)."""
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                size += _sizeof_obj(k, seen) + _sizeof_obj(v, seen)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += _sizeof_obj(item, seen)
        elif hasattr(obj, '__dict__'):
            size += _sizeof_obj(vars(obj), seen)
        return size

    def sizeof_mb(obj):
        return _sizeof_obj(obj) / (1024 * 1024)

    print(f"Backend: {BACKEND}")
    print(f"PFDAug: enabled={PFDAUG_ENABLED}, k={PFDAUG_K}, p={PFDAUG_P}\n")

    specs = [
        "habbof[all]",
        "cepdof[all]",
        "wepdtof[all]",
        "fisheye8k[all]",
    ]
    total_annotation_mb = 0.0

    for spec in specs:
        try:
            ds = get_dataset(spec)
            n = len(ds)

            # ---- Annotation metadata memory ----
            ann_mb = sizeof_mb(ds._images) + sizeof_mb(ds._anns_by_image) + sizeof_mb(ds._cats)
            total_annotation_mb += ann_mb

            # ---- Estimate image memory from metadata ----
            total_pixels = 0
            max_w, max_h = 0, 0
            for img_id, info in ds._images.items():
                w, h = info.get("width", 0), info.get("height", 0)
                total_pixels += w * h
                max_w = max(max_w, w)
                max_h = max(max_h, h)

            avg_pixels = total_pixels / n if n > 0 else 0
            # Images stored as float32 (C,H,W): channels=3, 4 bytes per pixel
            avg_img_mb = avg_pixels * 3 * 4 / (1024 * 1024)
            total_img_mb = total_pixels * 3 * 4 / (1024 * 1024)
            # Image tensors after resize+stack per batch (e.g. batch=8) — peak GPU for images only
            img_batch_gpu_mb = 8 * 3 * 640 * 640 * 4 / (1024 * 1024)  # 640 resize

            # ---- Boxes / annotations estimate ----
            total_boxes = sum(len(v) for v in ds._anns_by_image.values())
            avg_boxes = total_boxes / n if n > 0 else 0
            boxes_meta_mb = sizeof_mb(ds._anns_by_image)

            print(f"  {spec}:")
            print(f"    samples:      {n:6d}")
            print(f"    annotations:  {total_boxes:6d}  ({avg_boxes:.1f}/sample)")
            print(f"    max img:      {max_w}x{max_h}")
            print(f"    avg pixels:   {avg_pixels:,.0f}  ({avg_pixels:.0f} px)")
            print(f"    annotations metadata:  {ann_mb:.1f} MB")
            if total_boxes > 0:
                print(f"    boxes metadata:        {boxes_meta_mb:.1f} MB")
            print(f"    est. all img (FP32):   {total_img_mb:.1f} MB  (NOT loaded at once)")
            print(f"    est. 1 img mean:       {avg_img_mb:.2f} MB")
            print(f"    est. 1 batch img GPU:  {img_batch_gpu_mb:.1f} MB  (8x640x640, FP32)")
        except FileNotFoundError:
            print(f"  {spec}: SKIP (no file)")
        except Exception as e:
            print(f"  {spec}: ERROR — {e}")
            import traceback
            traceback.print_exc()
        print()

    print(f"  Total annotations metadata (all datasets): {total_annotation_mb:.1f} MB")
