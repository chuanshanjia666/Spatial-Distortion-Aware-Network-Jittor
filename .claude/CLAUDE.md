# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SDANet (Spatial Distortion-Aware Network) — PyTorch/Jittor fisheye camera object detection. Based on the IEEE TMM 2026 paper *"Towards Better Distortion Feature Learning for Object Detection in Top-View Fisheye Cameras"*.

Dual backend: `config.BACKEND = "pytorch"` or `"jittor"`. The `if BACKEND == "pytorch"` guards appear in `datasets/`, `model/op/sdaconv.py`, `model/backbone/darknet53.py`, `model/neck/fpn.py`, `model/head/sda_head.py`, and `model/sdanet.py`.

## Essential commands

```bash
# Download COCO-pretrained DarkNet53 backbone, extract backbone-only weights
python pretrain/download_pretrained.py

# Convert to Jittor format (if using jittor backend)
python pretrain/convert_to_jittor.py

# Convert raw datasets to unified COCO format
python tools/convert_habbof.py --root datasets/HABBOF
python tools/convert_fisheye8k.py --root datasets/FishEye8k

# Split merged all.json into train/val/test
python tools/split_train_val.py --ann datasets/HABBOF/annotations/all.json

# Smoke-test datasets
python datasets/datasets.py

# Train (all params from config.py, no CLI args)
python tools/train.py
```

## Architecture

```
config.py                    # All hyperparameters (the single source of truth)
datasets/
  pfdaug.py                  # PFDAug core class (barrel distortion + box warping)
  datasets.py                # FisheyeDataset — unified COCO reader, online PFDAug
  __init__.py                # exports PFDAug, FisheyeDataset, get_dataset
model/
  op/sdaconv.py              # SDAConv (dynamic multi-kernel conv), SDAConvBlock, SpatiallySeparateSDA
  backbone/darknet53.py      # DarkNet53 with GN/BN switching
  neck/fpn.py                # FPN neck with SDAConv lateral connections
  head/sda_head.py           # Per-scale SDAHead with oriented bbox prediction (cx,cy,w,h,θ,conf)
  sdanet.py                  # Top-level SDANet: backbone → FPN → head
pretrain/
  download_pretrained.py     # Download + key-remap COCO DarkNet53 weights
  convert_to_jittor.py       # PyTorch .pth ↔ Jittor .pkl
tools/
  convert_habbof.py          # HABBOF .txt → COCO JSON
  convert_fisheye8k.py       # FishEye8k samples.json → COCO JSON
  split_train_val.py         # all.json → train/val/test.json
  merge_cepdof.py            # Merge CEPDOF scene JSONs → all.json
  merge_wepdtof.py           # Merge WEPDTOF scene JSONs → all.json
  pfdaug.py                  # CLI offline PFDAug (standalone, imports core from datasets/)
  train.py                   # Training entry point (no CLI args, all params from config.py)
util/
  __init__.py                # exports metrics, loss, train_utils
  metrics.py                 # Rotated IoU, AP/mAP, NMS, box format conversion
  loss.py                    # SDALoss (YOLOv3-style oriented bbox loss)
  train_utils.py             # decode, collate, build_datasets, build_model, LR scheduling
```

## Key design decisions

### BN vs GN
`config.USE_ACCUMULATION_STEP = True` → all `BatchNorm2d` becomes `GroupNorm` (GN is stable with small micro-batches). Toggle `USE_ACCUMULATION_STEP` to switch; all modules use the `_norm2d()` helper.

### PFDAug — online only
PFDAug is applied in `FisheyeDataset.__getitem__` (controlled by `PFDAUG_ENABLED` in config), not offline. `tools/pfdaug.py` provides an optional offline CLI but it's secondary. Train split gets distortion with probability `PFDAUG_P`; val/test never gets distortion.

### Dataset format — unified COCO with rotated bboxes
All three datasets (HABBOF, FishEye8k, CEPDOF) are converted to the same COCO format with `"bbox": [cx, cy, w, h, R]`. Annotations are split into `annotations/{train,val,test}.json`. Multi-dataset training uses `torch.utils.data.ConcatDataset`.

### Pretrained backbone key remapping
MMDetection DarkNet53 uses keys like `backbone.conv_res_block1.res0.conv1.conv.weight`. Our `DarkNet53` uses `stages.0.res_0_0.cbl1.0.weight` (nn.Sequential indices). `pretrain/download_pretrained.py` remaps during download, yielding 156 matched keys. BN running stats are dropped since `USE_ACCUMULATION_STEP` enables GN.

### SDAConv — output mixing, not weight mixing
The paper's Eq. 2 combines kernel *weights*. Our implementation runs each base conv independently then mixes *outputs* weighted by the learned coefficient vector M — mathematically equivalent but simpler.
