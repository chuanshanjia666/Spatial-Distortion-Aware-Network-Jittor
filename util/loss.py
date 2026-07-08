"""
SDANet loss function — YOLOv3-style loss extended to oriented bounding boxes.

From "Towards Better Distortion Feature Learning for Object Detection
in Top-View Fisheye Cameras" (Guo et al., IEEE TMM 2026).

Prediction per anchor: [cx, cy, w, h, theta, obj_conf].
Loss = box_loss (MSE on tx,ty,tw,th,tR) + obj_loss (BCE).

Supports both PyTorch and Jittor backends.
"""

from config import (
    BACKEND, BOX_FIELDS, BOX_LOSS_WEIGHT, CLS_LOSS_WEIGHT,
    OBJ_LOSS_WEIGHT, STRIDES
)

import numpy as np

if BACKEND == "pytorch":
    import torch
    import torch.nn as nn
    import torch.nn as nn_module
else:
    import jittor as jt
    import jittor.nn as nn


def _mse_loss(pred, target, reduction='none'):
    """MSE loss with optional reduction."""
    if BACKEND == "pytorch":
        return nn_module.MSELoss(reduction=reduction)(pred, target)
    # Jittor: compute manually since mse_loss doesn't support reduction='none'
    diff = pred - target
    sqr = diff * diff
    if reduction == 'none':
        return sqr
    elif reduction == 'mean':
        return sqr.mean()
    else:  # 'sum'
        return sqr.sum()


def _bce_loss(pred, target):
    """Binary cross entropy with logits, no reduction."""
    if BACKEND == "pytorch":
        return nn_module.BCEWithLogitsLoss(reduction='none')(pred, target)
    # Jittor: compute manually since bce_with_logits doesn't support reduction='none'
    max_val = jt.clamp(-pred, min_v=0)
    loss = (1 - target) * pred + max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()
    return loss


def _wh_iou(bw, bh, aws, ahs):
    """Compute IoU between a box (bw, bh) and anchors (aws, ahs) — w/h only."""
    if BACKEND == "pytorch":
        inter_w = torch.min(bw, aws)
        inter_h = torch.min(bh, ahs)
    else:
        inter_w = jt.minimum(bw, aws)
        inter_h = jt.minimum(bh, ahs)
    inter = inter_w * inter_h
    union = bw * bh + aws * ahs - inter + 1e-8
    return inter / union


class SDALoss(nn.Module):
    """YOLOv3-style loss for oriented bounding boxes.

    Each prediction per anchor = [cx, cy, w, h, theta, obj_conf].

    Args:
        num_classes: Number of object categories.
        anchors:     List of 3 anchor groups (one per scale).
        strides:     Feature map strides for each detection scale.
        box_weight:  Weight for box regression loss.
        obj_weight:  Weight for objectness loss.
        cls_weight:  Weight for classification loss.
    """

    def __init__(self, num_classes: int, anchors: list, strides: list,
                 box_weight=BOX_LOSS_WEIGHT, obj_weight=OBJ_LOSS_WEIGHT,
                 cls_weight=CLS_LOSS_WEIGHT):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors[0])
        self.strides = strides
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        # Loss functions are handled by _bce_loss and _mse_loss helper functions

        # Register anchors as a persistent buffer (normalized grid-cell scale)
        if BACKEND == "pytorch":
            anchors_t = torch.tensor(anchors, dtype=torch.float32)  # (3, 3, 2)
            self.register_buffer('anchors', anchors_t)
        else:
            self.anchors = jt.array(anchors, dtype=jt.float32)  # (3, 3, 2)

    def forward(self, predictions, targets, images):
        """
        Args:
            predictions: list of 3 tensors (B, A*6, Hs, Ws)
            targets:     list of (boxes, labels) per image, boxes=(N,5) in pixels
            images:      (B, C, H, W) — needed for device reference only

        Returns:
            Scalar loss tensor.
        """
        B = images.shape[0]

        # Build scalar zero on the same device/dtype as images
        if BACKEND == "pytorch":
            device = images.device
            total_loss = torch.tensor(0.0, device=device)
            zeros = lambda *shape: torch.zeros(*shape, device=device)
            ones = lambda *shape: torch.ones(*shape, device=device)
            log_fn = torch.log
            arange_fn = torch.arange
        else:
            total_loss = jt.array(0.0, dtype=jt.float32)
            zeros = lambda *shape: jt.zeros(shape, dtype=jt.float32)
            ones = lambda *shape: jt.ones(shape, dtype=jt.float32)
            log_fn = jt.log
            arange_fn = lambda n: jt.arange(n, dtype=jt.int32)

        for s_idx, (pred, stride) in enumerate(zip(predictions, self.strides)):
            _, _, Hs, Ws = pred.shape
            A = self.num_anchors
            F = BOX_FIELDS  # 6

            # Reshape: (B, A*6, Hs, Ws) → (B, A, 6, Hs, Ws)
            pred = pred.view(B, A, F, Hs, Ws)

            # Build target tensor and masks
            tgt = zeros(B, A, F, Hs, Ws)
            obj_mask = zeros(B, A, Hs, Ws)     # 1 = positive
            noobj_mask = ones(B, A, Hs, Ws)

            # Pre-compute anchors on CPU to avoid GPU-CPU sync in loop
            if BACKEND == "pytorch":
                anchor_ws = self.anchors[s_idx, :, 0].cpu().numpy() / stride  # (A,) grid-scale
                anchor_hs = self.anchors[s_idx, :, 1].cpu().numpy() / stride
            else:
                anchor_ws = self.anchors[s_idx, :, 0].numpy() / stride  # (A,) grid-scale
                anchor_hs = self.anchors[s_idx, :, 1].numpy() / stride

            for b in range(B):
                boxes = targets[b][0]  # (N, 5) [cx, cy, w, h, R] in pixels
                if boxes.shape[0] == 0:
                    continue

                # Scale boxes to this feature map's grid
                boxes_scaled = boxes.clone()
                boxes_scaled[:, :4] /= stride

                # Jittor optimization: convert to numpy BEFORE loop
                # PyTorch: keep as tensor (CUDA ops are async, .item() is cheap)
                if BACKEND == "jittor":
                    boxes_np = boxes_scaled.numpy()

                    # Canonicalise w/h per-box: w=short, h=long (matches anchor clustering)
                    for ib in range(len(boxes_np)):
                        w, h = boxes_np[ib, 2], boxes_np[ib, 3]
                        if w > h:
                            boxes_np[ib, 2], boxes_np[ib, 3] = h, w

                    # Process boxes - all operations now on CPU (numpy)
                    for box in boxes_np:
                        cx_g, cy_g = box[0], box[1]  # grid-cell coords (float)
                        gx = int(cx_g)
                        gy = int(cy_g)
                        if gx < 0 or gx >= Ws or gy < 0 or gy >= Hs:
                            continue

                        # Find best anchor by IoU of w, h only
                        bw, bh = box[2], box[3]

                        # Compute IoU on CPU (no GPU sync)
                        inter_w = np.minimum(bw, anchor_ws)
                        inter_h = np.minimum(bh, anchor_hs)
                        inter = inter_w * inter_h
                        union = bw * bh + anchor_ws * anchor_hs - inter + 1e-8
                        ious = inter / union
                        best_a = int(np.argmax(ious))

                        # Fill target
                        tgt[b, best_a, 0, gy, gx] = cx_g - gx                  # tx
                        tgt[b, best_a, 1, gy, gx] = cy_g - gy                  # ty
                        tgt[b, best_a, 2, gy, gx] = log_fn(
                            bw / (anchor_ws[best_a] + 1e-8) + 1e-8)             # tw
                        tgt[b, best_a, 3, gy, gx] = log_fn(
                            bh / (anchor_hs[best_a] + 1e-8) + 1e-8)            # th
                        tgt[b, best_a, 4, gy, gx] = box[4] / 90.0               # R in [-90,90) → [-1,1)
                        tgt[b, best_a, 5, gy, gx] = 1.0                         # obj = 1
                        obj_mask[b, best_a, gy, gx] = 1.0
                        noobj_mask[b, best_a, gy, gx] = 0.0
                else:
                    # PyTorch: use tensor operations (async CUDA, .item() is cheap)
                    for ib in range(boxes_scaled.shape[0]):
                        w, h = boxes_scaled[ib, 2].item(), boxes_scaled[ib, 3].item()
                        if w > h:
                            boxes_scaled[ib, 2], boxes_scaled[ib, 3] = h, w

                    for box in boxes_scaled:
                        cx_g, cy_g = box[0], box[1]  # grid-cell coords (float)
                        gx = int(cx_g)
                        gy = int(cy_g)
                        if gx < 0 or gx >= Ws or gy < 0 or gy >= Hs:
                            continue

                        # Find best anchor by IoU of w, h only
                        bw, bh = box[2], box[3]
                        ious = _wh_iou(bw, bh, anchor_ws, anchor_hs)
                        best_a = int(ious.argmax())

                        # Fill target
                        tgt[b, best_a, 0, gy, gx] = cx_g - gx                  # tx
                        tgt[b, best_a, 1, gy, gx] = cy_g - gy                  # ty
                        tgt[b, best_a, 2, gy, gx] = log_fn(
                            bw / (anchor_ws[best_a] + 1e-8) + 1e-8)             # tw
                        tgt[b, best_a, 3, gy, gx] = log_fn(
                            bh / (anchor_hs[best_a] + 1e-8) + 1e-8)            # th
                        tgt[b, best_a, 4, gy, gx] = box[4] / 90.0               # R in [-90,90) → [-1,1)
                        tgt[b, best_a, 5, gy, gx] = 1.0                         # obj = 1
                        obj_mask[b, best_a, gy, gx] = 1.0
                        noobj_mask[b, best_a, gy, gx] = 0.0

            # ---- Box loss (MSE on tx, ty, tw, th, tR for positives) ----
            box_pred = pred[:, :, :5, :, :]   # (B, A, 5, Hs, Ws)
            box_tgt = tgt[:, :, :5, :, :]
            box_loss = _mse_loss(box_pred, box_tgt).sum(dim=2)  # (B, A, Hs, Ws)
            box_loss = (box_loss * obj_mask).sum() / max(obj_mask.sum(), 1)
            total_loss = total_loss + self.box_weight * box_loss

            # ---- Objectness loss (BCE, pos:neg = 2:1) ----
            obj_pred = pred[:, :, 5, :, :]  # (B, A, Hs, Ws)
            obj_tgt = obj_mask
            obj_loss_pos = _bce_loss(obj_pred, obj_tgt) * obj_mask
            obj_loss_neg = _bce_loss(obj_pred, obj_tgt) * noobj_mask * 0.5
            obj_loss = (obj_loss_pos.sum() + obj_loss_neg.sum()) / max(obj_mask.sum() * 2, 1)
            total_loss = total_loss + self.obj_weight * obj_loss

        return total_loss

    # Jittor 使用 execute 而不是 forward，PyTorch 使用 forward
    def execute(self, predictions, targets, images):
        return self.forward(predictions, targets, images)


if __name__ == "__main__":
    print(f"Testing with BACKEND={BACKEND}")

    # Create dummy data
    B, A, F, H, W = 2, 3, 6, 13, 13
    if BACKEND == "pytorch":
        import torch
        predictions = [torch.randn(B, A*F, H, W).cuda() for H, W in [(52, 52), (26, 26), (13, 13)]]
        images = torch.randn(B, 3, 416, 416).cuda()
        targets = [(
            torch.tensor([[100, 100, 30, 50, 15]], dtype=torch.float32),
            torch.tensor([0])
        ) for _ in range(B)]
    else:
        predictions = [jt.randn(B, A*F, H, W) for H, W in [(52, 52), (26, 26), (13, 13)]]
        images = jt.randn(B, 3, 416, 416)
        targets = [(
            jt.array([[100, 100, 30, 50, 15]], dtype=jt.float32),
            jt.array([0], dtype=jt.int64)
        ) for _ in range(B)]

    anchors = [[[32, 32], [48, 48], [64, 64]] for _ in range(3)]
    strides = [8, 16, 32]

    criterion = SDALoss(num_classes=1, anchors=anchors, strides=strides)
    loss = criterion(predictions, targets, images)
    print(f"Loss computed successfully: {loss}")
    print(f"Loss value: {loss if BACKEND == 'pytorch' else loss.numpy()}")
