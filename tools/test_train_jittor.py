#!/usr/bin/env python3
"""
Jittor end-to-end test for SDANet training and validation.

Run with:
    python tools/test_train_jittor.py
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BACKEND, DEVICE, STRIDES, INPUT_SIZE

if BACKEND != "jittor":
    print("This test only supports Jittor backend.")
    sys.exit(1)

import jittor as jt
from jittor.dataset import Dataset

from util import compute_map, SDALoss, decode_predictions, collate_fn
from util.anchor_cluster import load_or_cluster_anchors
from util.metrics import rotated_iou_single, oriented_nms
from datasets import get_dataset
from model import SDANet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_targets(boxes_list):
    """Convert collate_fn output → list of (boxes_np, labels_np)."""
    targets = []
    for item in boxes_list:
        b, l = item if isinstance(item, (list, tuple)) else (item, np.zeros(0, dtype=np.int64))
        bn = b.numpy() if hasattr(b, 'numpy') else b
        ln = l.numpy() if hasattr(l, 'numpy') else l
        if not isinstance(ln, np.ndarray):
            ln = np.array(ln, dtype=np.int64)
        targets.append((bn, ln))
    return targets


def _validate(model, loader, anchors, conf_thresh=0.3, max_per_image=500):
    """Run validation pass → return (all_preds, all_targets)."""
    model.eval()
    preds, targets = [], []

    t_data = t_forward = t_post = t_target = 0.0
    n_batches = 0

    for batch in loader:
        t0 = time.time()
        images, boxes_list = batch if isinstance(batch, (tuple, list)) else (batch, None)
        if DEVICE == "cuda":
            images = images.cuda()
        t_data += time.time() - t0
        n_batches += 1

        t1 = time.time()
        with jt.no_grad():
            dets = decode_predictions(model(images), conf_thresh=conf_thresh, anchors=anchors)
        t_forward += time.time() - t1

        # Cap per-image detections (top-k by score)
        t2 = time.time()
        capped = []
        for boxes, scores, labels in dets:
            if len(scores) > max_per_image:
                topk = np.argsort(scores)[-max_per_image:]
                boxes, scores, labels = boxes[topk], scores[topk], labels[topk]
            capped.append((boxes, scores, labels))
        preds.extend(capped)
        t_post += time.time() - t2

        t3 = time.time()
        targets.extend(_extract_targets(boxes_list))
        t_target += time.time() - t3

    total = t_data + t_forward + t_post + t_target
    print(f"  [validate] batches={n_batches}  total={total:.2f}s  "
          f"data={t_data:.2f}s({t_data/total*100:.0f}%)  "
          f"forward={t_forward:.2f}s({t_forward/total*100:.0f}%)  "
          f"post={t_post:.2f}s({t_post/total*100:.0f}%)  "
          f"target={t_target:.2f}s({t_target/total*100:.0f}%)")
    return preds, targets


def _val_metrics(preds, targets, class_names, label=""):
    """Print metrics for collected preds/targets."""
    tp = sum(len(p[0]) for p in preds)
    tg = sum(len(t[0]) for t in targets)
    if tp == 0 and tg == 0:
        return True

    coco = compute_map(preds, targets, class_names)
    print(f"  {label}  preds={tp}  GT={tg}  "
          f"mAP={coco['mAP']:.4f}")
    if np.isnan(coco['mAP']):
        print("\n  ❌ FAIL: mAP is NaN")
        sys.exit(1)
    return False


def _visualize(model, loader, anchors, vis_dir, max_vis=8):
    """Draw pred (red) and GT (green) boxes on first N images."""
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    count = 0

    for batch in loader:
        images, boxes_list = batch if isinstance(batch, (tuple, list)) else (batch, None)
        if DEVICE == "cuda":
            images = images.cuda()

        with jt.no_grad():
            dets = decode_predictions(model(images), conf_thresh=0.1, anchors=anchors)

        for i in range(images.shape[0]):
            if count >= max_vis:
                return
            img = images[i].numpy().transpose(1, 2, 0).copy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            for box in boxes_list[i][0]:
                pts = cv2.boxPoints(((float(box[0]), float(box[1])),
                                     (float(box[2]), float(box[3])), float(box[4])))
                cv2.drawContours(img, [np.intp(pts)], 0, (0, 255, 0), 2)
                cv2.putText(img, "GT", (int(box[0]) - 10, int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            pb, ps, _ = dets[i]
            if len(pb) > 0:
                keep = oriented_nms(pb, ps, iou_thresh=0.45)
                pb, ps = pb[keep], ps[keep]
            for j, box in enumerate(pb):
                pts = cv2.boxPoints(((float(box[0]), float(box[1])),
                                     (float(box[2]), float(box[3])), float(box[4])))
                cv2.drawContours(img, [np.intp(pts)], 0, (0, 0, 255), 1)
                cv2.putText(img, f"{ps[j]:.2f}", (int(box[0]), int(box[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            cv2.putText(img, "GREEN=GT  RED=Pred", (5, img.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(vis_dir, f"sample_{count:02d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            count += 1


def _train_epoch(model, loader, criterion, optimizer):
    """Run one training epoch, return average loss."""
    model.train()
    losses = []

    t_data = t_forward = t_backward = 0.0
    n_batches = 0

    for batch in loader:
        t0 = time.time()
        images, boxes_list = batch if isinstance(batch, (tuple, list)) else (batch, None)
        if DEVICE == "cuda":
            images = images.cuda()
        t_data += time.time() - t0
        n_batches += 1

        t1 = time.time()
        loss = criterion(model(images), boxes_list, images)
        t_forward += time.time() - t1

        t2 = time.time()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        t_backward += time.time() - t2

        losses.append(loss.numpy()[0])

    total = t_data + t_forward + t_backward
    print(f"  [train] batches={n_batches}  total={total:.2f}s  "
          f"data={t_data:.2f}s({t_data/total*100:.0f}%)  "
          f"forward={t_forward:.2f}s({t_forward/total*100:.0f}%)  "
          f"backward={t_backward:.2f}s({t_backward/total*100:.0f}%)")
    return np.mean(losses)


# ---------------------------------------------------------------------------
# Simple Subset dataset wrapper for Jittor
# ---------------------------------------------------------------------------

class Subset(Dataset):
    """Simple subset wrapper for Jittor."""

    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.set_attrs(total_len=len(indices))

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def collate_batch(self, batch):
        """Return raw batch list."""
        return batch


# ---------------------------------------------------------------------------
# DataLoader for Jittor (with collate_fn support)
# ---------------------------------------------------------------------------

class DataLoader:
    """Jittor DataLoader with collate_fn support."""

    def __init__(self, dataset, batch_size=4, shuffle=False,
                 num_workers=0, collate_fn=None, drop_last=False):
        self._dataset = dataset.set_attrs(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            keep_numpy_array=True,
        )
        self._collate_fn = collate_fn
        self._len = len(self._dataset)

    def __len__(self):
        return self._len

    def __iter__(self):
        for batch in self._dataset:
            if self._collate_fn is not None:
                yield self._collate_fn(batch)
            else:
                yield batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_test_config():
    """Get test configuration. Default to full config for servers."""
    # Default: 50 samples, batch_size=4 (for servers with enough GPU memory)
    # Reduce NUM_SAMPLES if OOM, keep BATCH_SIZE=4 for proper training
    return 50, 4


def main():
    # ---- Resolve dataset ----
    TEST_SPEC = None
    for spec in ["habbof[train]", "wepdtof[train]", "cepdof[train]"]:
        try:
            ds = get_dataset(spec)
            TEST_SPEC = spec
            break
        except FileNotFoundError:
            continue
    if TEST_SPEC is None:
        print("No datasets found. Run tools/split_train_val.py first.")
        sys.exit(1)

    NUM_SAMPLES, BATCH_SIZE = _get_test_config()
    subset_ds = Subset(ds, list(range(min(NUM_SAMPLES, len(ds)))))
    num_classes = ds.num_classes
    class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"Test dataset: {TEST_SPEC} → {len(ds)} total, {len(subset_ds)} samples")
    print(f"Num classes: {num_classes}  |  Input size: {INPUT_SIZE}  |  Batch size: {BATCH_SIZE}")
    print(f"Backend: {BACKEND}  |  Device: {DEVICE}\n")

    # ---- DataLoader ----
    loader = DataLoader(subset_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, collate_fn=collate_fn, drop_last=False)

    # ---- Model, anchors, loss ----
    model = SDANet(num_classes=num_classes)
    if DEVICE == "cuda":
        model = model.cuda()
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    resolved_anchors = load_or_cluster_anchors(
        [TEST_SPEC], num_clusters=9, input_size=INPUT_SIZE)
    print(f"Anchors: {resolved_anchors}\n")

    criterion = SDALoss(num_classes, resolved_anchors, STRIDES)
    optimizer = jt.optim.Adam(model.parameters(), lr=5e-4)

    # ===================================================================
    # Training (50 samples × 15 epochs)
    # ===================================================================
    NUM_EPOCHS = 15
    print("=" * 60)
    print(f"Training {NUM_EPOCHS} epochs on {len(subset_ds)} samples")
    print("=" * 60 + "\n")

    total_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        avg_loss = _train_epoch(model, loader, criterion, optimizer)
        t = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            t_val = time.time()
            preds, targets = _validate(model, loader, resolved_anchors)
            val_t = time.time() - t_val
            print(f"  epoch {epoch:3d}: loss={avg_loss:.1f}  "
                  f"train={t:.1f}s  val={val_t:.1f}s")
            _val_metrics(preds, targets, class_names, label=f"       ")
            model.train()

    print(f"\n  Total: {time.time() - total_start:.1f}s  "
          f"({(time.time() - total_start) / NUM_EPOCHS:.1f}s/epoch)")

    # ===================================================================
    # Final validation, visualization
    # ===================================================================
    print("\n--- Final Validation & Visualization ---")
    vis_dir = "output/vis"

    all_preds, all_targets = _validate(model, loader, resolved_anchors)
    _visualize(model, loader, resolved_anchors, vis_dir, max_vis=8)
    print(f"  Saved images to {vis_dir}/")

    failed = _val_metrics(all_preds, all_targets, class_names, label="final")
    if failed:
        sys.exit(1)

    # Diagnostic dump
    print("\n--- Diagnostic: top predictions vs GT ---")
    np.set_printoptions(precision=1, suppress=True)
    for i, ((pb, ps, _), (tb, _)) in enumerate(zip(all_preds[:3], all_targets[:3])):
        print(f"\n  Image {i}: GT={len(tb)}  Pred={len(pb)}")
        for j in range(min(3, len(tb))):
            print(f"    GT[{j}]:   [{tb[j][0]:6.1f} {tb[j][1]:6.1f} {tb[j][2]:6.1f} {tb[j][3]:6.1f} {tb[j][4]:6.1f}]")
        for j in range(min(5, len(pb))):
            print(f"    Pred[{j}]: [{pb[j][0]:6.1f} {pb[j][1]:6.1f} {pb[j][2]:6.1f} {pb[j][3]:6.1f} {pb[j][4]:6.1f}]  score={ps[j]:.3f}")
        if len(pb) > 0 and len(tb) > 0:
            best = max(rotated_iou_single(pb[0], gtb) for gtb in tb[:10])
            print(f"    Best IoU (top pred vs GT): {best:.4f}")

    # ===================================================================
    # Overfit test (3 samples × 10 epochs)
    # ===================================================================
    print("\n--- Overfit test (3 samples, 10 epochs) ---")

    small_loader = DataLoader(Subset(ds, list(range(3))), batch_size=3,
                               shuffle=False, num_workers=0, collate_fn=collate_fn)

    of_model = SDANet(num_classes=num_classes)
    if DEVICE == "cuda":
        of_model = of_model.cuda()
    of_opt = jt.optim.Adam(of_model.parameters(), lr=1e-3)
    of_crit = SDALoss(num_classes, resolved_anchors, STRIDES)

    initial_loss = final_loss = None
    for epoch in range(1, 11):
        avg = _train_epoch(of_model, small_loader, of_crit, of_opt)
        if epoch == 1:
            initial_loss = avg
        final_loss = avg

    ratio = final_loss / initial_loss
    print(f"  initial loss: {initial_loss:.2f}")
    print(f"  final loss:   {final_loss:.2f}")
    print(f"  ratio:        {ratio:.4f}")

    if ratio < 0.5:
        print("  ✅ PASS: Loss decreased significantly (model is learning)")
    elif ratio < 0.9:
        print("  ⚠️  WARN: Loss decreased but slowly")
    else:
        print("  ❌ FAIL: Loss did not decrease")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()