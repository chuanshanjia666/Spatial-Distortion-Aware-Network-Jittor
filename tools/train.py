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
    USE_ACCUMULATION_STEP, MAX_ITER, STRIDES, CONF_THRESH, NMS_IOU_THRESH,
    RESUME, AUTO_RESUME, OUTPUT_DIR, USE_FP16,
)

if BACKEND == "pytorch":
    import torch
    import torch.cuda.amp as amp
    from torch.utils.data import DataLoader
else:
    import jittor as jt

    jt.flags.enable_tuner = 1
    jt.flags.use_tensorcore = 1
    jt.flags.use_cuda_managed_allocator = 1
    # jt.flags.auto_mixed_precision_level = 4  # 智能 fp16
    jt.flags.lazy_execution = 1
    jt.flags.use_threading = 1

from util import (
    compute_map, compute_map_coco, SDALoss,
    decode_predictions, collate_fn, build_datasets, build_model,
    build_optimizer, build_lr_scheduler, oriented_nms,
)


# ---------------------------------------------------------------------------
# DataLoader implementations per backend
# ---------------------------------------------------------------------------

if BACKEND == "pytorch":
    import threading

    class DataLoader:
        """PyTorch DataLoader with optimized prefetching and CUDA async transfer."""

        def __init__(self, dataset, batch_size=1, shuffle=False,
                     num_workers=4, collate_fn=None, pin_memory=True,
                     drop_last=False, prefetch_factor=2):
            self._loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                persistent_workers=True if num_workers > 0 else False,
            )
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.collate_fn = collate_fn
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            self.prefetch_factor = prefetch_factor

        def __len__(self):
            return len(self._loader)

        def __iter__(self):
            return iter(self._loader)


# ---------------------------------------------------------------------------
# Jittor DataLoader with multi-threaded prefetch
# ---------------------------------------------------------------------------

if BACKEND == "jittor":
    import math

    class DataLoader:
        """Simple DataLoader for Jittor without threading complications."""

        def __init__(self, dataset, batch_size=1, shuffle=False,
                     num_workers=0, collate_fn=None, pin_memory=False,
                     drop_last=False, prefetch_factor=2):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn
            self.drop_last = drop_last
            self.num_workers = num_workers

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
                import numpy as np
                np.random.shuffle(indices)

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
#  Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir):
    """Find the most recent checkpoint in ``output_dir``.

    Priority:
        1. ``latest.pth`` (or ``latest.pkl`` for Jittor) — the last saved.
        2. The file with the highest ``iter`` number in its name.

    Returns:
        Path to the checkpoint file, or ``None`` if nothing found.
    """
    ext = ".pkl" if BACKEND == "jittor" else ".pth"
    latest_path = os.path.join(output_dir, f"latest{ext}")

    if os.path.isfile(latest_path):
        return latest_path

    # Fallback: scan for the highest iter number
    if not os.path.isdir(output_dir):
        return None
    best_iter = -1
    best_path = None
    for fname in os.listdir(output_dir):
        if not fname.startswith("sdanet_iter") or not fname.endswith(ext):
            continue
        try:
            it = int(fname.replace("sdanet_iter", "").replace(ext, ""))
        except ValueError:
            continue
        if it > best_iter:
            best_iter = it
            best_path = os.path.join(output_dir, fname)
    return best_path


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print(f"Backend: {BACKEND}, Device: {DEVICE}")
    print(f"Train datasets: {TRAIN_DATASETS}")
    print(f"Val datasets:   {VAL_DATASETS}")
    print(f"Input size: {INPUT_SIZE}, Step batch: {STEP_BATCH_SIZE}, "
            f"Max iters: {MAX_ITER}")

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

    # AMP mixed precision
    if USE_FP16:
        if BACKEND == "pytorch":
            scaler = torch.amp.GradScaler('cuda')
        else:
            # Jittor 1.x has no GradScaler nor autocast equivalent.
            # model.float16() doesn't do FP32 master weights — use PyTorch for FP16.
            scaler = None
    else:
        scaler = None

    lr_scheduler = build_lr_scheduler(optimizer, MAX_ITER)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Accumulation ----
    accumulation_steps = 1
    if USE_ACCUMULATION_STEP:
        accumulation_steps = max(1, BATCH_SIZE // STEP_BATCH_SIZE)
    print(f"Accumulation steps: {accumulation_steps}")

    # ---- Resume ----
    start_epoch = 0
    global_step = 0

    resume_path = RESUME
    if resume_path is None and AUTO_RESUME:
        resume_path = _find_latest_checkpoint(OUTPUT_DIR)
        if resume_path:
            print(f"Auto-resume: found {resume_path}")

    if resume_path and os.path.isfile(resume_path):
        print(f"[DEBUG] Loading checkpoint from {resume_path}")
        if BACKEND == "pytorch":
            ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        else:
            ckpt = jt.load(resume_path)
        print(f"[DEBUG] Checkpoint loaded, loading model state...")
        model.load_state_dict(ckpt['model_state'])
        print(f"[DEBUG] Model state loaded, loading optimizer state...")
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        if scaler is not None and ckpt.get('scaler_state'):
            scaler.load_state_dict(ckpt['scaler_state'])
            print(f"  AMP scaler state restored")
        print(f"Resumed from {resume_path} (epoch {start_epoch}, step {global_step})")
    elif resume_path and not os.path.isfile(resume_path):
        print(f"WARNING: RESUME={resume_path} not found, starting from scratch.")
    else:
        print(f"[DEBUG] No checkpoint to resume, starting from scratch.")

    # ---- Training loop ----
    epoch = start_epoch
    while global_step < MAX_ITER:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        epoch_updates = 0
        optimizer.zero_grad()
        train_len = len(train_loader)
        tail_chunk = train_len % accumulation_steps
        if tail_chunk == 0:
            tail_chunk = accumulation_steps
        chunk_loss = 0.0
        chunk_size = accumulation_steps
        chunk_count = 0

        for batch_idx, (images, boxes_list) in enumerate(train_loader):
            if global_step >= MAX_ITER:
                break

            if DEVICE == "cuda":
                if BACKEND == "pytorch":
                    images = images.cuda()
                else:
                    images = images.cuda()

            # Forward
            if USE_FP16 and BACKEND == "pytorch":
                with amp.autocast():
                    preds = model(images)
                    loss = criterion(preds, boxes_list, images)
            else:
                preds = model(images)
                loss = criterion(preds, boxes_list, images)

            raw_loss = loss.item() if BACKEND == "pytorch" else loss.numpy()[0]

            if tail_chunk != accumulation_steps and batch_idx >= train_len - tail_chunk:
                chunk_size = tail_chunk
            else:
                chunk_size = accumulation_steps

            loss = loss / chunk_size

            if USE_FP16 and BACKEND == "pytorch":
                scaler.scale(loss).backward()
            else:
                optimizer.backward(loss)

            chunk_count += 1
            chunk_loss += raw_loss

            if chunk_count == chunk_size:
                global_step += 1
                lr_scheduler.step(global_step)
                if USE_FP16 and BACKEND == "pytorch":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                update_loss = chunk_loss / chunk_count
                epoch_loss += update_loss
                epoch_updates += 1

                if global_step == 1 or global_step % 20 == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    print(f"  epoch {epoch:3d} | iter {global_step:5d}/{MAX_ITER} "
                          f"| loss {update_loss:.4f} "
                          f"| lr {cur_lr:.6f}")

                chunk_loss = 0.0
                chunk_count = 0

        if chunk_count > 0:
            global_step += 1
            lr_scheduler.step(global_step)
            if USE_FP16 and BACKEND == "pytorch":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            update_loss = chunk_loss / chunk_count
            epoch_loss += update_loss
            epoch_updates += 1

            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch:3d} | iter {global_step:5d}/{MAX_ITER} "
                  f"| loss {update_loss:.4f} "
                  f"| lr {cur_lr:.6f}")

        if epoch_updates > 0:
            avg_loss = epoch_loss / epoch_updates
        else:
            avg_loss = 0.0
        print(f"--- epoch {epoch} done | avg_loss={avg_loss:.4f} | step={global_step}/{MAX_ITER} ---")

        # ---- Save checkpoint ----
        scaler_state = scaler.state_dict() if (USE_FP16 and BACKEND == "pytorch") else None
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler_state,
            'global_step': global_step,
        }
        ext = ".pkl" if BACKEND == "jittor" else ".pth"
        ckpt_path = os.path.join(OUTPUT_DIR, f"sdanet_iter{global_step:06d}{ext}")
        if BACKEND == "pytorch":
            torch.save(ckpt, ckpt_path)
        else:
            jt.save(ckpt, ckpt_path)
        # Also save as latest for auto-resume
        latest_path = os.path.join(OUTPUT_DIR, f"latest{ext}")
        if BACKEND == "pytorch":
            torch.save(ckpt, latest_path)
        else:
            jt.save(ckpt, latest_path)
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
                    with jt.no_grad():
                        preds = model(images)
                        val_loss += criterion(preds, boxes_list, images)

                # Decode predictions for mAP (with NMS)
                dets = decode_predictions(preds, conf_thresh=CONF_THRESH, anchors=resolved_anchors)
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
