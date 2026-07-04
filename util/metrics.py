"""
Utility functions for SDANet: rotated IoU, AP/mAP, NMS.

All functions operate on oriented bounding boxes:
    boxes: (N, 5)  [cx, cy, w, h, R]  with R in degrees, range [-90, 90).
"""

import math
import numpy as np
import cv2


# ===================================================================
#  Rotated IoU (via OpenCV box intersection)
# ===================================================================

def rotated_iou_single(box1, box2):
    """IoU of two rotated boxes using OpenCV's rotatedRect intersection.

    Args:
        box1, box2: (5,)  [cx, cy, w, h, R] in degrees.

    Returns:
        float IoU in [0, 1].
    """
    rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
    # Only succeeds if boxes actually intersect
    try:
        inter = _rotated_rect_intersection_area(rect1, rect2)
    except Exception:
        return 0.0
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def rotated_iou(boxes1, boxes2):
    """Pairwise rotated IoU between two sets of boxes.

    Args:
        boxes1: (M, 5)
        boxes2: (N, 5)

    Returns:
        (M, N) float32 IoU matrix.
    """
    M, N = boxes1.shape[0], boxes2.shape[0]
    iou = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            iou[i, j] = rotated_iou_single(boxes1[i], boxes2[j])
    return iou


def _rotated_rect_intersection_area(rect1, rect2):
    """Compute intersection area of two cv2 rotated rects."""
    pts1 = cv2.boxPoints(rect1)
    pts2 = cv2.boxPoints(rect2)
    try:
        ret, pts = cv2.intersectConvexConvex(pts1, pts2)
    except Exception:
        return 0.0
    if ret is None or ret == 0:
        return 0.0
    return cv2.contourArea(pts)


# ===================================================================
#  NMS (oriented)
# ===================================================================

def oriented_nms(boxes, scores, iou_thresh=0.45):
    """Oriented bounding box NMS.

    Args:
        boxes:  (N, 5) float32  [cx, cy, w, h, R].
        scores: (N,)   float32  confidence.
        iou_thresh: IoU threshold for suppression.

    Returns:
        keep: list of indices to keep.
    """
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.array([rotated_iou_single(boxes[i], boxes[j]) for j in order[1:]])
        mask = ious < iou_thresh
        order = order[1:][mask]
    return keep


# ===================================================================
#  mAP / AP calculation
# ===================================================================

def compute_ap(recall: np.ndarray, precision: np.ndarray):
    """Compute Average Precision (VOC 2010+ — all-point interpolation).

    Args:
        recall:    (N,) sorted ascending.
        precision: (N,) corresponding precision values.

    Returns:
        float AP.
    """
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    ap = 0.0
    for i in range(1, len(r)):
        if r[i] > r[i - 1]:
            ap += (r[i] - r[i - 1]) * p[i]
    return float(ap)


def compute_map(predictions, targets, class_names, iou_thresh=0.5):
    """Compute mAP@IoU_thresh for oriented boxes.

    Args:
        predictions:  list of (boxes, scores, labels) per image.
            boxes:  (N_i, 5)  [cx, cy, w, h, R].
            scores: (N_i,)    confidence.
            labels: (N_i,)    int class id.
        targets:      list of (boxes, labels) per image (ground truth).
        class_names:  list of class name strings.
        iou_thresh:   IoU threshold for positive match (default 0.5 = AP50).

    Returns:
        dict with per-class AP and mAP.
    """
    n_classes = len(class_names)
    results = {}

    for c in range(n_classes):
        # Collect all detections and all GT for this class
        all_dets = []   # (img_idx, score, is_match)
        n_gt = 0

        for img_idx, (pred, gt) in enumerate(zip(predictions, targets)):
            p_boxes, p_scores, p_labels = pred
            t_boxes, t_labels = gt

            # GT for this class
            gt_mask = t_labels == c
            gt_boxes = t_boxes[gt_mask] if len(t_boxes) else t_boxes
            n_gt += len(gt_boxes)
            matched = np.zeros(len(gt_boxes), dtype=bool)

            # Dets for this class (sorted by score)
            det_mask = p_labels == c
            det_boxes = p_boxes[det_mask] if len(p_boxes) else p_boxes
            det_scores = p_scores[det_mask] if len(p_scores) else p_scores

            sort_idx = np.argsort(-det_scores)
            for idx in sort_idx:
                is_match = False
                if len(gt_boxes) > 0:
                    ious = np.array([rotated_iou_single(det_boxes[idx], gb)
                                     for gb in gt_boxes])
                    best = ious.argmax()
                    if ious[best] >= iou_thresh and not matched[best]:
                        matched[best] = True
                        is_match = True
                all_dets.append((det_scores[idx], is_match))

        if n_gt == 0 and len(all_dets) == 0:
            ap = 1.0  # no GT and no predictions → perfect by convention
        elif n_gt == 0:
            ap = 0.0  # noise detections only
        else:
            all_dets.sort(key=lambda x: -x[0])
            tp = np.array([d[1] for d in all_dets], dtype=np.float32)
            fp = 1.0 - tp
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / n_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)
            ap = compute_ap(recall, precision)

        results[class_names[c]] = ap

    results["mAP"] = np.mean(list(results.values())[:n_classes])
    return results


# ===================================================================
#  Box format conversion helpers
# ===================================================================

def cxcywhR_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h, R] (degrees) to four corner points.

    Args:
        boxes: (N, 5)

    Returns:
        (N, 4, 2) corner points.
    """
    N = boxes.shape[0]
    corners = np.zeros((N, 4, 2), dtype=np.float32)
    for i, (cx, cy, w, h, R) in enumerate(boxes):
        rect = ((cx, cy), (w, h), R)
        corners[i] = cv2.boxPoints(rect)
    return corners


def corners_to_cxcywhR(corners: np.ndarray) -> np.ndarray:
    """Convert four corner points back to [cx, cy, w, h, R]."""
    N = corners.shape[0]
    boxes = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        rect = cv2.minAreaRect(corners[i].reshape(-1, 1, 2).astype(np.float32))
        (cx, cy), (w, h), R = rect
        if w < h:
            w, h = h, w
            R += 90
        R = (R + 90) % 180 - 90
        boxes[i] = [cx, cy, w, h, R]
    return boxes
