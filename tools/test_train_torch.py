import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BACKEND, DEVICE, STRIDES, INPUT_SIZE

if BACKEND != "pytorch":
    print("This test only supports PyTorch backend.")
    sys.exit(1)

import torch
from torch.utils.data import DataLoader, Subset

from util import compute_map_coco, SDALoss, decode_predictions, collate_fn
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
        bn = b.cpu().numpy() if torch.is_tensor(b) else b
        ln = l.cpu().numpy() if torch.is_tensor(l) else l
        if not isinstance(ln, np.ndarray):
            ln = np.array(ln, dtype=np.int64)
        targets.append((bn, ln))
    return targets


def _validate(model, loader, anchors, conf_thresh=0.3, max_per_image=500):
    """Run validation pass → return (all_preds, all_targets)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, boxes_list in loader:
            if DEVICE == "cuda":
                images = images.cuda()
            dets = decode_predictions(model(images), conf_thresh=conf_thresh, anchors=anchors)
            # Cap per-image detections (top-k by score) to avoid O(n²) metric explosion
            capped = []
            for boxes, scores, labels in dets:
                if len(scores) > max_per_image:
                    topk = np.argsort(scores)[-max_per_image:]
                    boxes, scores, labels = boxes[topk], scores[topk], labels[topk]
                capped.append((boxes, scores, labels))
            preds.extend(capped)
            targets.extend(_extract_targets(boxes_list))
    return preds, targets


def _val_metrics(preds, targets, class_names, label=""):
    """Print COCO metrics for collected preds/targets."""
    tp = sum(len(p[0]) for p in preds)
    tg = sum(len(t[0]) for t in targets)
    if tp == 0 and tg == 0:
        return True

    coco = compute_map_coco(preds, targets, class_names)
    print(f"  {label}  preds={tp}  GT={tg}  "
          f"AP@50={coco['AP50']:.4f}  AP@75={coco['AP75']:.4f}  "
          f"mAP={coco['mAP']:.4f}  "
          f"AP_S={coco['AP_small']:.4f}  "
          f"AP_M={coco['AP_medium']:.4f}  "
          f"AP_L={coco['AP_large']:.4f}")
    if np.isnan(coco['mAP']):
        print("\n  ❌ FAIL: mAP is NaN")
        sys.exit(1)
    return False


def _visualize(model, loader, anchors, vis_dir, max_vis=8):
    """Draw pred (red) and GT (green) boxes on first N images."""
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for images, boxes_list in loader:
            if DEVICE == "cuda":
                images = images.cuda()
            dets = decode_predictions(model(images), conf_thresh=0.1, anchors=anchors)
            for i in range(images.shape[0]):
                if count >= max_vis:
                    return
                img = images[i].cpu().numpy().transpose(1, 2, 0).copy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

                for box in boxes_list[i][0].cpu().numpy():
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
    for images, boxes_list in loader:
        if DEVICE == "cuda":
            images = images.cuda()
        optimizer.zero_grad()
        loss = criterion(model(images), boxes_list, images)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    NUM_SAMPLES = 50
    subset_ds = Subset(ds, list(range(min(NUM_SAMPLES, len(ds)))))
    num_classes = ds.num_classes
    class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"Test dataset: {TEST_SPEC} → {len(ds)} total, {len(subset_ds)} samples")
    print(f"Num classes: {num_classes}  |  Input size: {INPUT_SIZE}\n")

    # ---- DataLoader ----
    loader = DataLoader(subset_ds, batch_size=4, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, drop_last=False)

    # ---- Model, anchors, loss ----
    model = SDANet(num_classes=num_classes)
    if DEVICE == "cuda":
        model = model.cuda()
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    resolved_anchors = load_or_cluster_anchors(
        [TEST_SPEC], num_clusters=9, input_size=INPUT_SIZE)
    print(f"Anchors: {resolved_anchors}\n")

    criterion = SDALoss(num_classes, resolved_anchors, STRIDES)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

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
                  f"train={t:.1f}s  val={val_t:.1f}s  "
                  f"step={t / len(loader) * 1000:.0f}ms")
            _val_metrics(preds, targets, class_names, label=f"       ")
            model.train()

    print(f"\n  Total: {time.time() - total_start:.1f}s  "
          f"({(time.time() - total_start) / NUM_EPOCHS:.1f}s/epoch)")

    # ===================================================================
    # Final validation, visualization, diagnostic
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
    # Overfit test (5 samples × 10 epochs)
    # ===================================================================
    print("\n--- Overfit test (5 samples, 10 epochs) ---")

    small_loader = DataLoader(Subset(ds, list(range(5))), batch_size=5,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    of_model = SDANet(num_classes=num_classes)
    if DEVICE == "cuda":
        of_model = of_model.cuda()
    of_opt = torch.optim.Adam(of_model.parameters(), lr=1e-3)
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
