"""
Prediction decoder for SDANet — converts YOLO-style grid outputs to
absolute oriented bounding boxes.

Also includes dataset building and collation helpers.

Supports both PyTorch and Jittor backends.
"""

import numpy as np
from config import (BACKEND, ANCHORS, STRIDES, BOX_FIELDS, INPUT_SIZE,
                    DEVICE, LR, MIN_LR, WARMUP_ITERS)

if BACKEND == "pytorch":
    import torch
    import torch.nn.functional as F
    import torch.utils.data as data
else:
    import jittor as jt
    import jittor.nn as nn
    import jittor.nn.functional as F

# ===================================================================
#  Prediction decoder
# ===================================================================

def decode_predictions(predictions, conf_thresh=0.3):
    """Decode YOLO-style outputs → per-image (boxes, scores, labels).

    Args:
        predictions: list of 3 tensors (B, A*6, Hs, Ws).
        conf_thresh: confidence threshold for filtering.

    Returns:
        List of (boxes, scores, labels) per batch item.
        boxes:  (N, 5) float32 [cx, cy, w, h, R] in pixels.
        scores: (N,)   float32.
        labels: (N,)   int64.
    """
    B = predictions[0].shape[0]
    all_dets = []

    for b in range(B):
        img_boxes, img_scores, img_labels = [], [], []
        for s_idx, (pred, stride) in enumerate(zip(predictions, STRIDES)):
            A = len(ANCHORS[s_idx])
            _, _, Hs, Ws = pred.shape

            if BACKEND == "pytorch":
                pred_np = pred[b].view(A, BOX_FIELDS, Hs, Ws).detach().cpu().numpy()
            else:
                pred_np = pred[b].view(A, BOX_FIELDS, Hs, Ws).numpy()

            for a in range(A):
                aw, ah = ANCHORS[s_idx][a]
                for gy in range(Hs):
                    for gx in range(Ws):
                        obj = 1.0 / (1.0 + np.exp(-pred_np[a, 5, gy, gx]))
                        if obj < conf_thresh:
                            continue
                        tx = pred_np[a, 0, gy, gx]
                        ty = pred_np[a, 1, gy, gx]
                        tw = pred_np[a, 2, gy, gx]
                        th = pred_np[a, 3, gy, gx]
                        tR = pred_np[a, 4, gy, gx]

                        cx = (gx + 1.0 / (1.0 + np.exp(-tx))) * stride
                        cy = (gy + 1.0 / (1.0 + np.exp(-ty))) * stride
                        w = np.exp(tw) * aw
                        h = np.exp(th) * ah
                        R = tR * 90.0
                        R = (R + 90.0) % 180.0 - 90.0

                        img_boxes.append([cx, cy, w, h, R])
                        img_scores.append(obj)
                        img_labels.append(0)  # TODO: class prediction

        if img_boxes:
            all_dets.append((np.array(img_boxes, dtype=np.float32),
                             np.array(img_scores, dtype=np.float32),
                             np.array(img_labels, dtype=np.int64)))
        else:
            all_dets.append(
                (np.empty((0, 5), dtype=np.float32),
                 np.empty((0,), dtype=np.float32),
                 np.empty((0,), dtype=np.int64)))
    return all_dets


# ===================================================================
#  Image / box resizing
# ===================================================================

def resize_image_and_boxes(image, boxes, target_size):
    """Resize image and scale boxes to ``target_size × target_size``.

    Args:
        image:  (C, H, W) float32 tensor.
        boxes:  (N, 5) tensor [cx, cy, w, h, R] in pixel coords.
        target_size: output square size.

    Returns:
        (image, boxes) — resized.
    """
    _, H, W = image.shape
    image = F.interpolate(image.unsqueeze(0), size=(target_size, target_size),
                          mode='bilinear', align_corners=False).squeeze(0)
    if boxes.shape[0] > 0:
        sx, sy = target_size / W, target_size / H
        boxes = boxes.clone()
        boxes[:, 0] *= sx
        boxes[:, 1] *= sy
        boxes[:, 2] *= sx
        boxes[:, 3] *= sy
    return image, boxes


# ===================================================================
#  DataLoader helpers
# ===================================================================

def collate_fn(batch):
    """Resize all images to INPUT_SIZE and stack into a batch.

    Args:
        batch: list of (image, boxes, labels) from dataset.

    Returns:
        (images, boxes_list)
        images: (B, C, INPUT_SIZE, INPUT_SIZE).
        boxes_list: list of (N, 5) tensors per image (variable N).
    """
    images, boxes_list, labels_list = zip(*batch)
    resized_images = []
    resized_boxes = []
    for img, bxs in zip(images, boxes_list):
        img, bxs = resize_image_and_boxes(img, bxs, INPUT_SIZE)
        resized_images.append(img)
        resized_boxes.append(bxs)

    if BACKEND == "pytorch":
        images = torch.stack(resized_images, dim=0)
    else:
        images = jt.stack(resized_images, dim=0)
    return images, list(zip(resized_boxes, labels_list))


# ===================================================================
#  Dataset building
# ===================================================================

def build_datasets(train_specs, val_specs):
    """Build training and validation ConcatDatasets from spec lists.

    Args:
        train_specs: list of ``"preset[split]"`` specs for training.
        val_specs:   list of ``"preset[split]"`` specs for validation.

    Returns:
        (train_dataset, val_dataset) — either may be None.
    """
    from datasets import get_dataset

    train_ds = []
    val_ds = []
    for spec in train_specs:
        try:
            ds = get_dataset(spec)
            train_ds.append(ds)
            print(f"  train: {spec} → {len(ds)} samples, {ds.num_classes} classes")
        except FileNotFoundError:
            print(f"  train: {spec} → SKIP (no file)")
    for spec in val_specs:
        try:
            ds = get_dataset(spec)
            val_ds.append(ds)
            print(f"  val:   {spec} → {len(ds)} samples, {ds.num_classes} classes")
        except FileNotFoundError:
            print(f"  val:   {spec} → SKIP (no file)")

    if BACKEND == "pytorch":
        from torch.utils.data import ConcatDataset
        train = ConcatDataset(train_ds) if train_ds else None
        val = ConcatDataset(val_ds) if val_ds else None
    else:
        # Jittor equivalent: simple concat via ConcatDataset
        train = _JittorConcatDataset(train_ds) if train_ds else None
        val = _JittorConcatDataset(val_ds) if val_ds else None
    return train, val


class _JittorConcatDataset:
    """Minimal ConcatDataset for Jittor (iterates all sub-datasets)."""

    def __init__(self, datasets):
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._cumsum = [0]
        for l in self._lengths:
            self._cumsum.append(self._cumsum[-1] + l)
        self.total = self._cumsum[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        ds_idx = 0
        for cs, cl in zip(self._cumsum[:-1], self._cumsum[1:]):
            if cs <= idx < cl:
                break
            ds_idx += 1
        local_idx = idx - self._cumsum[ds_idx]
        return self.datasets[ds_idx][local_idx]


# ===================================================================
#  Model building
# ===================================================================

def build_model(num_classes):
    """Build SDANet model and move to device.

    Args:
        num_classes: number of object categories.

    Returns:
        SDANet model instance.
    """
    from model import SDANet

    model = SDANet(
        num_classes=num_classes,
    )
    if DEVICE == "cuda":
        model = model.cuda()
    return model


def build_optimizer(model):
    """Build optimizer for the model.

    Returns:
        Optimizer instance (PyTorch or Jittor).
    """
    if BACKEND == "pytorch":
        return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    else:
        return jt.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)


# ===================================================================
#  LR scheduling
# ===================================================================

def warmup_lr(optimizer, warmup_iters: int, current_iter: int):
    """Linear LR warmup.

    Args:
        optimizer:     PyTorch or Jittor optimizer.
        warmup_iters:  Number of warmup iterations.
        current_iter:  Current global step (1-indexed).
    """
    if current_iter >= warmup_iters:
        return
    lr = LR * (current_iter / warmup_iters)
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def build_lr_scheduler(optimizer, max_iters: int):
    """Build LR scheduler (cosine annealing after warmup).

    For PyTorch: returns a manual step function; call step_lr() each iter.
    For Jittor:  returns a jt.nn.lr_scheduler attached to the optimizer.

    Args:
        optimizer: PyTorch or Jittor optimizer.
        max_iters:  Total training iterations.

    Returns:
        A scheduler object with a step(current_iter) method.
    """
    if BACKEND == "pytorch":
        return _PyTorchScheduler(optimizer, max_iters)
    else:
        # Jittor: use CosineAnnealingLR with warmup-like step
        return _JittorScheduler(optimizer, max_iters)


class _PyTorchScheduler:
    def __init__(self, optimizer, max_iters):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self._step = 0

    def step(self, current_iter: int):
        warmup_lr(self.optimizer, WARMUP_ITERS, current_iter)
        cosine_annealing_lr(self.optimizer, self.max_iters, current_iter)


def cosine_annealing_lr(optimizer, max_iters: int, current_iter: int):
    """Cosine annealing LR schedule (after warmup).

    Args:
        optimizer:     PyTorch optimizer.
        max_iters:     Total training iterations.
        current_iter:  Current global step (1-indexed).
    """
    if current_iter < WARMUP_ITERS:
        return  # warmup handles this phase

    progress = (current_iter - WARMUP_ITERS) / max(1, max_iters - WARMUP_ITERS)
    lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr


class _JittorScheduler:
    """Jittor-compatible LR scheduler wrapping cosine annealing."""

    def __init__(self, optimizer, max_iters):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.current_iter = 0

    def step(self, current_iter: int = None):
        if current_iter is not None:
            self.current_iter = current_iter
        else:
            self.current_iter += 1

        warmup_lr(self.optimizer, WARMUP_ITERS, self.current_iter)
        if self.current_iter >= WARMUP_ITERS:
            progress = (self.current_iter - WARMUP_ITERS) / max(
                1, self.max_iters - WARMUP_ITERS)
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (
                1.0 + np.cos(np.pi * progress))
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr


# ===================================================================
#  Compatibility shims (use model output tensors, not these directly)
# ===================================================================

# Expose tensor library alias so other modules can import from here
# instead of branching on BACKEND everywhere.
if BACKEND == "pytorch":
    __all__ = ['torch', 'nn', 'F']
else:
    __all__ = ['jt', 'nn', 'F']