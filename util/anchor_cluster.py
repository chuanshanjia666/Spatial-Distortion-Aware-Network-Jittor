#!/usr/bin/env python3
"""
K-means anchor box clustering for fisheye datasets.

Computes anchor priors from dataset bounding boxes using the standard
YOLOv2-style distance metric (1 - IoU).  Results are cached to disk so
clustering runs only once per dataset combination.

Usage:
    >>> from util.anchor_cluster import cluster_anchors, load_or_cluster_anchors
    >>> anchors = load_or_cluster_anchors(
    ...     ["habbof[train]", "wepdtof[train]"],
    ...     num_clusters=9, input_size=416)
"""

import os
import sys
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ANCHOR_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "pretrain", "anchors"
)


def _wh_iou_matrix(boxes: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Compute IoU between every box and every cluster (wh only).

    Args:
        boxes:    (N, 2)  [w, h] in pixels.
        clusters: (K, 2)  cluster centers.

    Returns:
        (N, K) IoU matrix.
    """
    N = boxes.shape[0]
    K = clusters.shape[0]
    box_w, box_h = boxes[:, 0:1], boxes[:, 1:2]        # (N, 1)
    cluster_w, cluster_h = clusters[:, 0], clusters[:, 1]  # (K,)

    inter_w = np.minimum(box_w, cluster_w[None, :])     # (N, K)
    inter_h = np.minimum(box_h, cluster_h[None, :])
    inter = inter_w * inter_h

    box_area = box_w * box_h                              # (N, 1)
    cluster_area = cluster_w * cluster_h                  # (K,)
    union = box_area + cluster_area[None, :] - inter + 1e-8
    return inter / union


def kmeans_anchors(boxes: np.ndarray, k: int = 9,
                   max_iters: int = 300, tol: float = 1e-6,
                   seed: int = 42) -> np.ndarray:
    """K-means clustering of box sizes using 1-IoU distance.

    Args:
        boxes:     (N, 2) [w, h] in pixels.
        k:         Number of clusters (default 9 = 3 scales × 3 anchors).
        max_iters: Max iterations.
        tol:       Convergence tolerance.
        seed:      Random seed.

    Returns:
        (K, 2) clusters sorted by area (ascending).
    """
    N = boxes.shape[0]
    rng = np.random.default_rng(seed)

    # Initialise centroids by picking random boxes
    indices = rng.choice(N, k, replace=False)
    centroids = boxes[indices].astype(np.float64).copy()

    for it in range(max_iters):
        iou = _wh_iou_matrix(boxes, centroids)            # (N, K)
        dist = 1.0 - iou                                   # (N, K)
        assignments = dist.argmin(axis=1)                  # (N,)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = boxes[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]  # empty cluster → keep old

        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < tol:
            break

    # Sort by area
    areas = centroids[:, 0] * centroids[:, 1]
    centroids = centroids[np.argsort(areas)]
    return centroids


def _collect_boxes(dataset_specs: list, input_size: int) -> np.ndarray:
    """Collect all bounding boxes from the given dataset specs.

    Boxes are scaled to *input_size* and rotated-box w/h are canonicalised
    so that clustering sees a consistent distribution across rotation angles:

        • w → min(w, h)    (short side)
        • h → max(w, h)    (long side)

    Without canonicalisation, a 100×50 box at R=0 and the same box at R=90
    (50×100) would be treated as two different shapes, breaking k-means.
    """
    from datasets import get_dataset

    all_boxes = []
    for spec in dataset_specs:
        try:
            ds = get_dataset(spec)
        except FileNotFoundError:
            print(f"  [anchor_cluster] SKIP {spec} (no file)")
            continue

        # Collect boxes, scale to input_size, and canonicalise w/h
        for img_id, file_name in ds._entries:
            anns = ds._anns_by_image.get(img_id, [])
            for a in anns:
                bbox = a["bbox"]
                if len(bbox) >= 4:
                    img_info = ds._images.get(img_id, {})
                    orig_w = img_info.get("width", input_size)
                    orig_h = img_info.get("height", input_size)
                    sx = input_size / orig_w if orig_w else 1.0
                    sy = input_size / orig_h if orig_h else 1.0
                    bw = float(bbox[2]) * sx
                    bh = float(bbox[3]) * sy
                    # Canonicalise: w = short side, h = long side
                    # This makes rotated boxes comparable regardless of R
                    if bw > bh:
                        bw, bh = bh, bw
                    all_boxes.append((bw, bh))

    boxes = np.array(all_boxes, dtype=np.float32)
    return boxes


def _cache_key(dataset_specs: list, num_clusters: int,
               input_size: int) -> str:
    """Deterministic cache key from specs + params."""
    raw = json.dumps({
        "specs": sorted(dataset_specs),
        "k": num_clusters,
        "size": input_size,
    }, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(key: str) -> str:
    os.makedirs(ANCHOR_CACHE_DIR, exist_ok=True)
    return os.path.join(ANCHOR_CACHE_DIR, f"anchors_{key}.json")


def _scale_anchors(anchors: np.ndarray, input_size: int) -> np.ndarray:
    """Scale anchors from pixel space to INPUT_SIZE.

    The clustering is performed on raw image boxes (original pixels).
    We need to scale them so they match the INPUT_SIZE used in training
    (images are resized during collate).  We use a heuristic: scale by
    the ratio of INPUT_SIZE to a typical fisheye image dimension (≈1920).
    """
    # Collect boxes are in original image pixels.
    # Training resizes images to input_size × input_size.
    # We return anchors in input_size pixel space.
    # Since we don't know the original image size per-box, we compute
    # the average image size from the collected boxes and scale.
    return anchors  # kept in original pixels; scaling handled during training


def cluster_anchors(dataset_specs: list, num_clusters: int = 9,
                    input_size: int = 416, force: bool = False,
                    seed: int = 42) -> np.ndarray:
    """Cluster anchors from dataset(s), with disk cache.

    Args:
        dataset_specs: List of ``"preset[split]"`` specs.
        num_clusters:  Number of clusters (default 9).
        input_size:    Training input size (used for cache key).
        force:         If True, re-cluster even if cache exists.
        seed:          Random seed.

    Returns:
        (K, 2) float32 anchor boxes [w, h] in original-image pixel space.
    """
    key = _cache_key(dataset_specs, num_clusters, input_size)
    path = _cache_path(key)

    if not force and os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        anchors = np.array(data["anchors"], dtype=np.float32)
        print(f"[anchor_cluster] Loaded cached anchors from {path}")
        return anchors

    print(f"[anchor_cluster] Clustering {num_clusters} anchors from "
          f"{dataset_specs} ...")
    boxes = _collect_boxes(dataset_specs, input_size)
    if len(boxes) == 0:
        raise RuntimeError("No boxes collected — check dataset specs.")

    print(f"[anchor_cluster] Collected {len(boxes)} boxes")
    centroids = kmeans_anchors(boxes, k=num_clusters, seed=seed)

    # Save to cache
    data = {
        "key": key,
        "specs": dataset_specs,
        "num_clusters": num_clusters,
        "input_size": input_size,
        "num_boxes": int(len(boxes)),
        "anchors": centroids.astype(float).tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[anchor_cluster] Saved {num_clusters} anchors to {path}")
    print(f"[anchor_cluster] Anchors: {centroids.astype(int).tolist()}")

    return centroids.astype(np.float32)


def load_or_cluster_anchors(dataset_specs: list = None,
                            num_clusters: int = 9,
                            input_size: int = 416,
                            force: bool = False,
                            seed: int = 42):
    """Main entry point: return anchors as the nested list config expects.

    Returns:
        List of 3 groups (one per scale), each with num_clusters//3 anchors:
        ``[[(w,h), ...], [(w,h), ...], [(w,h), ...]]``
    """
    if dataset_specs is None:
        from config import TRAIN_DATASETS
        dataset_specs = TRAIN_DATASETS

    centroids = cluster_anchors(
        dataset_specs, num_clusters=num_clusters,
        input_size=input_size, force=force, seed=seed,
    )
    # Split into 3 scales (small → large), each with k/3 anchors
    assert num_clusters % 3 == 0, "num_clusters must be divisible by 3"
    per_scale = num_clusters // 3
    groups = []
    for s in range(3):
        start = s * per_scale
        end = start + per_scale
        groups.append([
            (float(w), float(h))
            for w, h in centroids[start:end]
        ])
    return groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster anchor boxes from dataset(s)")
    parser.add_argument("--specs", nargs="+",
                        default=["habbof[train]", "wepdtof[train]"],
                        help="Dataset specs, e.g. habbof[train]")
    parser.add_argument("--k", type=int, default=9,
                        help="Number of clusters")
    parser.add_argument("--input-size", type=int, default=416,
                        help="Training input size")
    parser.add_argument("--force", action="store_true",
                        help="Re-cluster even if cache exists")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    anchors = load_or_cluster_anchors(
        args.specs, num_clusters=args.k,
        input_size=args.input_size, force=args.force, seed=args.seed,
    )
    print("\n# Paste this into config.py ANCHORS:")
    print(f"ANCHORS = {anchors}")
