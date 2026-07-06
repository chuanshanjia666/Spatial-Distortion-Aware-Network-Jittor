#!/usr/bin/env python3
"""
SDANet training script.

All hyperparameters are read from ``config.py`` — no command-line arguments required.

Supports both PyTorch and Jittor backends (controlled by config.BACKEND).

Usage:
    python tools/train.py

To override settings, edit ``config.py`` before running.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BACKEND, DEVICE, NUM_WORKERS, BATCH_SIZE, INPUT_SIZE,
    STEP_BATCH_SIZE, TRAIN_DATASETS, VAL_DATASETS, TEST_DATASETS,
    USE_ACCUMULATION_STEP, EPOCHS, STRIDES, CONF_THRESH, NMS_IOU_THRESH,
    RESUME, OUTPUT_DIR,
)

if BACKEND == "pytorch":
    import torch
    import torch.cuda.amp as amp
    from torch.utils.data import DataLoader
else:
    import jittor as jt

from util import (
    compute_map, compute_map_coco, SDALoss,
    decode_predictions, collate_fn, build_datasets, build_model,
    build_optimizer, build_lr_scheduler, oriented_nms,
)


# ---------------------------------------------------------------------------
# Jittor DataLoader shim
# ---------------------------------------------------------------------------
if BACKEND == "jittor":
    import math
    import jittor as jt

    class DataLoader:
        """Minimal DataLoader for Jittor (sequential + optional shuffle)."""

        def __init__(self, dataset, batch_size=1, shuffle=False,
                     num_workers=0, collate_fn=None, pin_memory=False,
                     drop_last=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn
            self.drop_last = drop_last
            self._rng = None

            ds_len = len(dataset)
            if drop_last:
                self._num_batches = ds_len // batch_size
            else:
                self._num_batches = math.ceil(ds_len / batch_size)

        def __len__(self):
            return self._num_batches

        def __iter__(self):
            ds_len = len(self.dataset)
            indices = list(range(ds_len))
            if self.shuffle:
                if self._rng is None:
                    self._rng = jt.rand(ds_len).argsort()
                else:
                    self._rng = jt.rand(ds_len).argsort()
                indices = self._rng.numpy().tolist()

            batch = []
            for idx in indices:
                batch.append(self.dataset[idx])
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self._collate(batch)

        def _collate(self, batch):
            if self.collate_fn is not None:
                return self.collate_fn(batch)
            return batch


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print(f"Backend: {BACKEND}, Device: {DEVICE}")
    print(f"Train datasets: {TRAIN_DATASETS}")
    print(f"Val datasets:   {VAL_DATASETS}")
    print(f"Input size: {INPUT_SIZE}, Step batch: {STEP_BATCH_SIZE}, "
          f"Epochs: {EPOCHS}")

    # ---- Datasets ----
    train_ds, val_ds, resolved_anchors = build_datasets(TRAIN_DATASETS, VAL_DATASETS)
    if train_ds is None:
        raise RuntimeError("No training dataset found.  Check TRAIN_DATASETS in config.py.")

    total_classes = max(
        (ds.num_classes if hasattr(ds, 'num_classes') else 1)
        for ds in ([train_ds, val_ds] if val_ds else [train_ds])
        if ds is not None
    )
    print(f"Num classes: {total_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=STEP_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"), drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=STEP_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=(DEVICE == "cuda"),
        )

    # ---- Model ----
    model = build_model(total_classes)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    # ---- Loss & optimizer & scheduler ----
    criterion = SDALoss(total_classes, resolved_anchors, STRIDES)
    optimizer = build_optimizer(model)

    # AMP mixed precision (PyTorch only)
    if BACKEND == "pytorch":
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    max_iters = len(train_loader) * EPOCHS
    lr_scheduler = build_lr_scheduler(optimizer, max_iters)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Accumulation ----
    accumulation_steps = 1
    if USE_ACCUMULATION_STEP:
        accumulation_steps = max(1, BATCH_SIZE // STEP_BATCH_SIZE)
    print(f"Accumulation steps: {accumulation_steps}")

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    if RESUME and os.path.isfile(RESUME):
        if BACKEND == "pytorch":
            ckpt = torch.load(RESUME, map_location='cpu', weights_only=True)
        else:
            ckpt = jt.load(RESUME)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"Resumed from {RESUME} (epoch {start_epoch}, step {global_step})")

    # ---- Training loop ----
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, boxes_list) in enumerate(train_loader):
            global_step += 1

            # LR schedule
            lr_scheduler.step(global_step)

            if DEVICE == "cuda":
                if BACKEND == "pytorch":
                    images = images.cuda()
                else:
                    images = images.cuda()

            # AMP autocast for PyTorch forward pass
            if BACKEND == "pytorch":
                with amp.autocast():
                    preds = model(images)
                    loss = criterion(preds, boxes_list, images)
            else:
                preds = model(images)
                loss = criterion(preds, boxes_list, images)

            loss = loss / accumulation_steps

            if BACKEND == "pytorch":
                scaler.scale(loss).backward()
            else:
                optimizer.backward(loss)

            if (batch_idx + 1) % accumulation_steps == 0:
                if BACKEND == "pytorch":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item() if BACKEND == "pytorch" else loss.numpy()[0]
            epoch_loss += batch_loss * accumulation_steps

            if batch_idx % 20 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  epoch {epoch:3d} | iter {batch_idx:4d}/{len(train_loader)} "
                      f"| loss {batch_loss * accumulation_steps:.4f} "
                      f"| lr {cur_lr:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- epoch {epoch} done | avg_loss={avg_loss:.4f} ---")

        # ---- Save checkpoint ----
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'global_step': global_step,
        }
        ckpt_path = os.path.join(OUTPUT_DIR, f"sdanet_epoch{epoch:03d}.pth")
        if BACKEND == "pytorch":
            torch.save(ckpt, ckpt_path)
        else:
            jt.save(ckpt, ckpt_path)
        print(f"  saved: {ckpt_path}")

        # ---- Validation ----
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            for images, boxes_list in val_loader:
                if DEVICE == "cuda":
                    if BACKEND == "pytorch":
                        images = images.cuda()
                    else:
                        images = images.cuda()

                if BACKEND == "pytorch":
                    with torch.no_grad():
                        preds = model(images)
                        val_loss += criterion(preds, boxes_list, images)
                else:
                    preds = model(images)
                    val_loss += criterion(preds, boxes_list, images)

                # Decode predictions for mAP (with NMS)
                dets = decode_predictions(preds, conf_thresh=CONF_THRESH)
                batch_preds = []
                for dets_per_img in dets:
                    pred_boxes, pred_scores, pred_labels = dets_per_img
                    if len(pred_boxes) > 0:
                        keep = oriented_nms(pred_boxes, pred_scores, iou_thresh=NMS_IOU_THRESH)
                        pred_boxes = pred_boxes[keep]
                        pred_scores = pred_scores[keep]
                        pred_labels = pred_labels[keep]
                    batch_preds.append((pred_boxes, pred_scores, pred_labels))
                all_preds.extend(batch_preds)
                gts = []
                for item in boxes_list:
                    # item is (boxes, labels) from collate_fn
                    if isinstance(item, (list, tuple)):
                        b = item[0]
                        l = item[1]
                    else:
                        b = item
                        l = np.zeros(len(b) if hasattr(b, '__len__') else 0, dtype=np.int64)
                    if BACKEND == "pytorch":
                        bn = b.cpu().numpy() if torch.is_tensor(b) else b
                        ln = l.cpu().numpy() if torch.is_tensor(l) else l
                    else:
                        bn = b.numpy() if hasattr(b, 'numpy') else b
                        ln = l.numpy() if hasattr(l, 'numpy') else l
                    if not isinstance(ln, np.ndarray):
                        ln = np.array(ln, dtype=np.int64)
                    gts.append((bn, ln))
                all_targets.extend(gts)

            val_loss = val_loss.item() if BACKEND == "pytorch" else val_loss.numpy()[0]
            val_loss = val_loss / len(val_loader)
            if total_classes > 0 and len(all_preds) > 0:
                map_results = compute_map(all_preds, all_targets,
                                           [f"class_{i}" for i in range(total_classes)])
                print(f"  val_loss={val_loss:.4f}  mAP@50={map_results['mAP']:.4f}")
            else:
                print(f"  val_loss={val_loss:.4f}")
            model.train()

    print("Training complete.")


if __name__ == "__main__":
    main()