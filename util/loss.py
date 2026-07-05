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

if BACKEND == "pytorch":
    import torch
    import torch.nn as nn
else:
    import jittor as jt
    import jittor.nn as nn


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
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

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

            for b in range(B):
                boxes = targets[b][0]  # (N, 5) [cx, cy, w, h, R] in pixels
                if boxes.shape[0] == 0:
                    continue

                # Scale boxes to this feature map's grid
                boxes_scaled = boxes.clone()
                boxes_scaled[:, :4] /= stride

                for box in boxes_scaled:
                    cx_g, cy_g = box[0], box[1]  # grid-cell coords (float)
                    gx = int(cx_g)
                    gy = int(cy_g)
                    if gx < 0 or gx >= Ws or gy < 0 or gy >= Hs:
                        continue

                    # Find best anchor by IoU of w, h only
                    bw, bh = box[2], box[3]
                    if BACKEND == "pytorch":
                        anchor_ws = self.anchors[s_idx, :, 0]  # (A,)
                        anchor_hs = self.anchors[s_idx, :, 1]
                    else:
                        anchor_ws = self.anchors[s_idx, :, 0]
                        anchor_hs = self.anchors[s_idx, :, 1]
                    ious = _wh_iou(bw, bh, anchor_ws, anchor_hs)
                    best_a = int(ious.argmax().item())

                    # Fill target
                    tgt[b, best_a, 0, gy, gx] = cx_g - gx                  # tx
                    tgt[b, best_a, 1, gy, gx] = cy_g - gy                  # ty
                    tgt[b, best_a, 2, gy, gx] = log_fn(
                        bw / (anchor_ws[best_a] + 1e-8) + 1e-8)             # tw
                    tgt[b, best_a, 3, gy, gx] = log_fn(
                        bh / (anchor_hs[best_a] + 1e-8) + 1e-8)             # th
                    tgt[b, best_a, 4, gy, gx] = box[4] / 90.0               # R in [-90,90) → [-1,1)
                    tgt[b, best_a, 5, gy, gx] = 1.0                         # obj = 1
                    obj_mask[b, best_a, gy, gx] = 1.0
                    noobj_mask[b, best_a, gy, gx] = 0.0

            # ---- Box loss (MSE on tx, ty, tw, th, tR for positives) ----
            box_pred = pred[:, :, :5, :, :]   # (B, A, 5, Hs, Ws)
            box_tgt = tgt[:, :, :5, :, :]
            box_loss = self.mse(box_pred, box_tgt).sum(dim=2)  # (B, A, Hs, Ws)
            box_loss = (box_loss * obj_mask).sum() / max(obj_mask.sum(), 1)
            total_loss = total_loss + self.box_weight * box_loss

            # ---- Objectness loss (BCE, pos:neg = 2:1) ----
            obj_pred = pred[:, :, 5, :, :]  # (B, A, Hs, Ws)
            obj_tgt = obj_mask
            obj_loss_pos = self.bce(obj_pred, obj_tgt) * obj_mask
            obj_loss_neg = self.bce(obj_pred, obj_tgt) * noobj_mask * 0.5
            obj_loss = (obj_loss_pos.sum() + obj_loss_neg.sum()) / max(obj_mask.sum() * 2, 1)
            total_loss = total_loss + self.obj_weight * obj_loss

        return total_loss
