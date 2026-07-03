#!/usr/bin/env python3
"""
Download DarkNet53 pretrained weights (COCO) for SDANet and
extract backbone-only weights into a clean checkpoint file.

Sources (tried in order):
  1. MMDetection YOLOv3 DarkNet53 COCO 320x320  (~406 MB)
  2. MMDetection YOLOv3 DarkNet53 COCO 416x416  (~406 MB)

Output:
  pretrain/darknet53_backbone.pth  — backbone weights with keys matching
                                     our DarkNet53 parameter names.

Usage:
    python pretrain/download_pretrained.py
"""

import os, sys, urllib.request
from collections import OrderedDict

URLS = [
    "http://download.openmmlab.com/mmdetection/v2.0/yolo/"
    "yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth",
    "http://download.openmmlab.com/mmdetection/v2.0/yolo/"
    "yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_CHECKPOINT = os.path.join(OUT_DIR, "darknet53_full.pth")
BACKBONE_FILE   = os.path.join(OUT_DIR, "darknet53_backbone.pth")


def _download(url: str, dest: str) -> bool:
    print(f"  Downloading: {url}")
    print(f"  To:          {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  FAILED: {e}")
        return False
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  OK ({size_mb:.1f} MB)")
    return True


def _extract_backbone(full_path: str, out_path: str):
    """Extract backbone weights and remap keys to match our DarkNet53."""
    import torch
    raw = torch.load(full_path, map_location="cpu", weights_only=True)
    sd = raw.get("state_dict", raw)

    backbone = OrderedDict()
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        nk = k[len("backbone."):]              # strip "backbone." prefix

        # skip running stats and tracking buffers (GN doesn't use them)
        if any(x in nk for x in ("num_batches_tracked", "running_mean", "running_var")):
            continue

        # --- conv1 → stem (now a Sequential [conv, norm, act]) ---
        if nk.startswith("conv1."):
            nk = "stem." + nk[len("conv1."):]
            sub = nk.split(".", 1)[1]       # e.g. "conv.weight" or "bn.weight"
            submap = _cbl_submap(sub)
            nk = "stem." + submap

        # --- conv_res_block{1-5} → stages.{0-4} ---
        elif nk.startswith("conv_res_block"):
            tail = nk[len("conv_res_block"):]         # e.g. "1.conv.conv.weight"
            block_idx = int(tail.split(".")[0]) - 1   # 0-indexed
            rest = tail[len(str(block_idx + 1)) + 1:]  # e.g. "conv.conv.weight"

            if rest.startswith("conv."):
                # Downsample CBL: {"conv.conv.weight": "0.weight", "conv.bn.weight": "1.weight"} etc.
                sub = rest[len("conv."):]              # e.g. "conv.weight" or "bn.weight"
                submap = _cbl_submap(sub)
                nk = f"stages.{block_idx}.ds_{block_idx}.{submap}"

            elif rest.startswith("res"):
                # e.g. "res0.conv1.conv.weight"
                res_idx_str = rest[len("res"):].split(".")[0]  # "0"
                res_idx = int(res_idx_str)
                sub = rest[len("res" + res_idx_str) + 1:]      # "conv1.conv.weight"
                conv_name = sub.split(".")[0]                   # "conv1" or "conv2"
                conv_tail = sub[len(conv_name) + 1:]            # "conv.weight" or "bn.weight"
                cbl_name = {"conv1": "cbl1", "conv2": "cbl2"}[conv_name]
                submap = _cbl_submap(conv_tail)
                nk = f"stages.{block_idx}.res_{block_idx}_{res_idx}.{cbl_name}.{submap}"

        # drop buffers we don't need (already filtered above)
        backbone[nk] = v

    torch.save(backbone, out_path)
    print(f"  Extracted backbone: {len(backbone)} keys → {os.path.abspath(out_path)}")
    return backbone


def _cbl_submap(subkey: str) -> str:
    """Map MMDetection CBL leaf keys to our nn.Sequential index keys.

    MMDet:   "conv.weight" → 0.weight   (0 = Conv2d)
             "bn.weight"   → 1.weight   (1 = Norm2d)
             "bn.bias"     → 1.bias
             "bn.running_mean"  → (skip, GN doesn't use these)
             "bn.running_var"   → (skip)
    """
    if subkey == "conv.weight":
        return "0.weight"
    if subkey == "bn.weight":
        return "1.weight"
    if subkey == "bn.bias":
        return "1.bias"
    # running_mean / running_var → skip (GN has no running stats; BN track
    # buffers are not parameters)
    raise KeyError(subkey)  # should not happen — caller filters before calling


def main():
    # Step 1: download
    downloaded = False
    for url in URLS:
        print("[pretrain] Trying COCO-pretrained URL...")
        if _download(url, FULL_CHECKPOINT):
            downloaded = True
            break

    if not downloaded:
        print("\nAll download sources failed.  Download manually:")
        for u in URLS:
            print(f"  {u}")
        sys.exit(1)

    # Step 2: extract backbone
    print("[pretrain] Extracting backbone weights...")
    bb = _extract_backbone(FULL_CHECKPOINT, BACKBONE_FILE)

    # Step 3: quick sanity check
    print("[pretrain] Sanity check — first 8 keys:")
    for k in list(bb.keys())[:8]:
        print(f"  {k}: {list(bb[k].shape)}")

    # Step 4: remove full checkpoint to save space
    os.remove(FULL_CHECKPOINT)
    print(f"[pretrain] Removed {FULL_CHECKPOINT} (backbone-only saved)")

    print(f"\nDone.  Use in config:  DARKNET53_PRETRAINED = \"{BACKBONE_FILE}\"")
    print(f"  For Jittor backend, run:  python pretrain/convert_to_jittor.py")


if __name__ == "__main__":
    main()
