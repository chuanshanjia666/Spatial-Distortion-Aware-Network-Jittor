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
    """Compute AP@IoU_thresh and mAP for oriented boxes.

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
    total_gt = 0
    total_dets = 0

    # Precision-Recall data across all classes (for mAP)
    all_recalls = []
    all_precisions = []

    for c in range(n_classes):
        # Collect all detections and all GT for this class
        all_dets = []   # (score, is_match)
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

        total_gt += n_gt
        total_dets += len(all_dets)

        if n_gt == 0 and len(all_dets) == 0:
            ap = float("nan")  # no GT no pred → undefined, excluded from mAP
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
            all_recalls.append(recall)
            all_precisions.append(precision)

        results[class_names[c]] = ap

    results["mAP"] = np.nanmean(list(results.values())[:n_classes])
    results["GT_count"] = total_gt
    results["Det_count"] = total_dets
    return results


def compute_map_coco(predictions, targets, class_names):
    """Compute COCO-style mAP across multiple IoU thresholds.

    Computes AP@[.50, .55, ..., .95] (10 levels) and reports:
        - AP@50       standard PASCAL VOC metric
        - AP@75       stricter IoU threshold
        - mAP         COCO AP averaged over all thresholds
        - AP_small    AP for objects with area < 32² pixels
        - AP_medium   AP for objects with 32² ≤ area < 96² pixels
        - AP_large    AP for objects with area ≥ 96² pixels

    Returns:
        dict with keys: AP50, AP75, mAP, AP_small, AP_medium, AP_large,
        AP50_per_class, GT_count, Det_count.
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    area_ranges = {
        "small":  (0,     32 * 32),
        "medium": (32 * 32, 96 * 96),
        "large":  (96 * 96, float("inf")),
    }

    results = {}
    all_ap_across_ious = {area_key: [] for area_key in area_ranges}

    for iou in iou_thresholds:
        ap_results = compute_map(predictions, targets, class_names, iou_thresh=float(iou))
        all_ap_across_ious.setdefault("all", []).append(ap_results["mAP"])
        results.setdefault("AP_per_iou", []).append(ap_results["mAP"])

    # Compute area-stratified AP
    for area_key, (lo, hi) in area_ranges.items():
        # Filter predictions and targets by box area
        area_results = _compute_ap_by_area(predictions, targets, class_names,
                                           lo, hi, iou_thresholds)
        all_ap_across_ious[area_key] = area_results

    results["AP50"] = float(all_ap_across_ious["all"][0])
    results["AP75"] = float(all_ap_across_ious["all"][5])  # index 5 = 0.75
    results["mAP"] = float(np.nanmean(all_ap_across_ious["all"]))
    for k in area_ranges:
        results[f"AP_{k}"] = float(np.nanmean(all_ap_across_ious[k]))

    results["GT_count"] = sum(
        len(t[0]) for t in targets if hasattr(t, '__len__'))
    results["Det_count"] = sum(
        len(p[0]) for p in predictions if hasattr(p, '__len__'))

    return results


def _compute_ap_by_area(predictions, targets, class_names,
                        lo: float, hi: float, iou_thresholds: np.ndarray):
    """Compute mAP across thresholds, only for boxes in [lo, hi) pixel area."""
    aps = []
    for iou in iou_thresholds:
        filtered_preds = []
        filtered_targets = []
        for pred, gt in zip(predictions, targets):
            p_boxes, p_scores, p_labels = pred
            t_boxes, t_labels = gt

            # Filter by area
            p_area = p_boxes[:, 2] * p_boxes[:, 3]
            t_area = t_boxes[:, 2] * t_boxes[:, 3]
            p_keep = (p_area >= lo) & (p_area < hi)
            t_keep = (t_area >= lo) & (t_area < hi)

            filtered_preds.append((
                p_boxes[p_keep], p_scores[p_keep], p_labels[p_keep]
            ))
            filtered_targets.append((t_boxes[t_keep], t_labels[t_keep]))

        map_result = compute_map(filtered_preds, filtered_targets,
                                 class_names, iou_thresh=float(iou))
        aps.append(map_result["mAP"])
    return aps


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
