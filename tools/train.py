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
    RESUME, AUTO_RESUME, OUTPUT_DIR, USE_FP16, RANDOM_SEED,
    VALIDATE_INTERVAL, SAVE_INTERVAL, LOG_INTERVAL,
)

if BACKEND == "pytorch":
    import torch
    import torch.cuda.amp as amp
    from torch.utils.data import DataLoader
else:
    import jittor as jt

    # jt.flags.enable_tuner = 1
    # jt.flags.use_tensorcore = 1
    # jt.flags.use_cuda_managed_allocator = 1
    # # jt.flags.auto_mixed_precision_level = 4  # 智能 fp16
    # jt.flags.lazy_execution = 1
    # jt.flags.use_threading = 1

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
# Jittor DataLoader: use native Dataset with multiprocessing prefetch
# ---------------------------------------------------------------------------

if BACKEND == "jittor":
    class DataLoader:
        """Jittor DataLoader using native Dataset with multiprocess prefetching.

        Jittor's Dataset has built-in support for:
        - Multi-process data loading via num_workers
        - RingBuffer-based prefetch mechanism
        - Worker status monitoring via display_worker_status()
        """

        def __init__(self, dataset, batch_size=1, shuffle=False,
                     num_workers=4, collate_fn=None, pin_memory=False,
                     drop_last=False, prefetch_factor=2):
            # Jittor Dataset.set_attrs() enables multiprocess prefetching
            self._dataset = dataset.set_attrs(
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                keep_numpy_array=False,
            )
            self._collate_fn = collate_fn
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.collate_fn = collate_fn
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            self._len = len(self._dataset)

        def __len__(self):
            return self._len

        def __iter__(self):
            for batch in self._dataset:
                if self._collate_fn is not None:
                    # Apply collate_fn to unpacked batch
                    yield self._collate_fn(batch)
                else:
                    yield batch


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
    # ---- Set random seed for reproducibility ----
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if BACKEND == "pytorch":
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        # Note: cudnn.deterministic + benchmark=False hurts speed noticeably.
        # CPU seeds alone are sufficient for practical reproducibility.
        # Uncomment below for bitwise-reproducible results at the cost of speed:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    else:
        jt.set_global_seed(RANDOM_SEED)

    print(f"Backend: {BACKEND}, Device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")
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
        if ckpt.get('optimizer_state') is not None:
            print(f"[DEBUG] Model state loaded, loading optimizer state...")
            optimizer.load_state_dict(ckpt['optimizer_state'])
        else:
            print(f"[DEBUG] No optimizer state in checkpoint (weights-only save)")
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

    # ---- Helpers for save and validation ----

    def _save_checkpoint(step, ep):
        scaler_state = scaler.state_dict() if (USE_FP16 and BACKEND == "pytorch") else None

        if BACKEND == "jittor":
            # Jittor: 只保存 model weights。
            # optimizer state_dict (Adam momentum/variance) 的 .numpy()
            # 底层必须走 cudaMallocHost，Docker ulimit -l=64MB 不够，
            # 暂无绕过的办法。
            model_state = model.state_dict()
            opt_state = None
        else:
            model_state = model.state_dict()
            opt_state = optimizer.state_dict()

        ckpt = {
            'epoch': ep,
            'model_state': model_state,
            'optimizer_state': opt_state,
            'scaler_state': scaler_state,
            'global_step': step,
        }
        ext = ".pkl" if BACKEND == "jittor" else ".pth"
        ckpt_path = os.path.join(OUTPUT_DIR, f"sdanet_iter{step:06d}{ext}")
        if BACKEND == "pytorch":
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f"latest{ext}"))
        else:
            jt.save(ckpt, ckpt_path)
            jt.save(ckpt, os.path.join(OUTPUT_DIR, f"latest{ext}"))
        print(f"  saved: {ckpt_path}")

    def _validate():
        if val_loader is None:
            return
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        for images, boxes_list in val_loader:
            if DEVICE == "cuda":
                images = images.cuda()

            if BACKEND == "pytorch":
                with torch.no_grad():
                    preds = model(images)
                    val_loss += criterion(preds, boxes_list, images)
            else:
                with jt.no_grad():
                    preds = model(images)
                    val_loss += criterion(preds, boxes_list, images)

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
                if isinstance(item, (list, tuple)):
                    b, l = item[0], item[1]
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

        vloss = val_loss.item() if BACKEND == "pytorch" else val_loss.numpy()[0]
        vloss = vloss / len(val_loader)
        if total_classes > 0 and len(all_preds) > 0:
            map_results = compute_map(all_preds, all_targets,
                                       [f"class_{i}" for i in range(total_classes)])
            print(f"  val_loss={vloss:.4f}  mAP@50={map_results['mAP']:.4f}")
        else:
            print(f"  val_loss={vloss:.4f}")
        model.train()

    # ---- Training loop ----
    epoch = start_epoch
    epoch_loss = 0.0
    epoch_updates = 0
    samples_seen = 0
    train_len = len(train_loader.dataset)
    model.train()
    optimizer.zero_grad()
    chunk_loss = 0.0
    chunk_count = 0

    train_iter = iter(train_loader)
    while global_step < MAX_ITER:
        try:
            images, boxes_list = next(train_iter)
        except StopIteration:
            # Epoch finished — print stats and start new epoch
            avg_loss = epoch_loss / max(1, epoch_updates)
            print(f"--- epoch {epoch} done | avg_loss={avg_loss:.4f} | step={global_step}/{MAX_ITER} ---")
            epoch += 1
            epoch_loss = 0.0
            epoch_updates = 0
            samples_seen = 0
            train_iter = iter(train_loader)
            images, boxes_list = next(train_iter)

        if DEVICE == "cuda":
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
        loss = loss / accumulation_steps

        if USE_FP16 and BACKEND == "pytorch":
            scaler.scale(loss).backward()
        else:
            if BACKEND == "jittor":
                optimizer.backward(loss)
            else:
                loss.backward()

        chunk_count += 1
        chunk_loss += raw_loss
        samples_seen += images.shape[0]

        if chunk_count == accumulation_steps:
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

            if global_step == 1 or global_step % LOG_INTERVAL == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  epoch {epoch:3d} | iter {global_step:5d}/{MAX_ITER} "
                      f"| loss {update_loss:.4f} "
                      f"| lr {cur_lr:.6f}")

            chunk_loss = 0.0
            chunk_count = 0

            # Validation
            if val_loader is not None and global_step % VALIDATE_INTERVAL == 0:
                _validate()

            # Save
            if global_step % SAVE_INTERVAL == 0:
                _save_checkpoint(global_step, epoch)

    print("Training complete.")


if __name__ == "__main__":
    main()
