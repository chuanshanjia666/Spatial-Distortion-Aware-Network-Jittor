"""
Spatial Distortion-Aware Network (SDANet).

Full object detection network for top-view fisheye cameras.

Architecture (Fig. 3 in the paper)::

    Input (1024x1024)
      |
      v
    DarkNet53 Backbone
      |
      +-- f8  (128x128, 256ch)
      +-- f16 (64x64,  512ch)
      +-- f32 (32x32, 1024ch)
      |
      v
    FPN Neck (with SDAConv)
      |
      +-- p8  (128x128, 128ch)
      +-- p16 (64x64,  128ch)
      +-- p32 (32x32,  128ch)
      |
      v
    SDA Head (per scale, oriented bbox prediction)
      |
      +-- preds[0] (A*6, 128, 128)
      +-- preds[1] (A*6, 64,  64)
      +-- preds[2] (A*6, 32,  32)

Output per anchor: [cx, cy, w, h, theta, obj_conf] in grid-cell local coords.

Supports both PyTorch and Jittor backends.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (BACKEND, SDA_BASE_KERNELS, NECK_OUT_CHANNELS,
                    NUM_ANCHORS, SSS_ENABLED, SSS_GRID,
                    DARKNET53_PRETRAINED, INPUT_SIZE)

if BACKEND == "pytorch":
    import torch
    import torch.nn as nn
else:
    import jittor.nn as nn

from model.backbone.darknet53 import DarkNet53
from model.neck.fpn import FPNNeck
from model.head.sda_head import SDAHeadMulti


class SDANet(nn.Module):
    """Spatial Distortion-Aware Network for fisheye object detection.

    Args:
        num_classes:   Number of object categories.
        base_kernels:  SDAConv base kernel sizes (default from config).
        sss_enabled:   Enable Spatially Separate Strategy.
        sss_grid:      Grid size for SSS.
        pretrained:    Path to pretrained DarkNet53 weights.
    """

    def __init__(self, num_classes=None, base_kernels=None,
                 sss_enabled=None, sss_grid=None, pretrained=None):
        super().__init__()
        num_classes = num_classes or 1
        base_kernels = base_kernels or SDA_BASE_KERNELS
        sss_enabled = sss_enabled if sss_enabled is not None else SSS_ENABLED
        pretrained = pretrained or DARKNET53_PRETRAINED

        # Backbone
        self.backbone = DarkNet53()
        backbone_channels = self.backbone.out_channels  # [256, 512, 1024]

        # Neck (FPN with SDAConv)
        self.neck = FPNNeck(in_channels=backbone_channels,
                            out_channels=NECK_OUT_CHANNELS,
                            base_kernels=base_kernels,
                            use_sda=True)

        # Head (per-scale detection with SDAConv / SSS)
        self.head = SDAHeadMulti(
            num_scales=3,
            in_channels=NECK_OUT_CHANNELS,
            num_anchors=NUM_ANCHORS,
            base_kernels=base_kernels,
            sss_enabled=sss_enabled,
        )

        self.num_classes = num_classes
        self.num_anchors = NUM_ANCHORS
        self.strides = [8, 16, 32]

        # Load pretrained backbone if available
        if pretrained and os.path.isfile(pretrained):
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path):
        if BACKEND == "pytorch":
            state = torch.load(path, map_location='cpu', weights_only=True)
        else:
            import jittor as jt
            state = jt.load(path)

        # Handle MMDetection-style checkpoints (state_dict wrapper)
        if 'state_dict' in state and len(state) <= 3:
            sd = state['state_dict']
        else:
            sd = state

        # Filter to backbone-only keys
        backbone_state = {}
        for k, v in sd.items():
            # Strip 'backbone.' prefix if present
            if k.startswith('backbone.'):
                nk = k[len('backbone.'):]
                backbone_state[nk] = v
            else:
                # Direct key — try loading as-is (already backbone-only)
                backbone_state[k] = v

        if backbone_state:
            missing, unexpected = self.backbone.load_state_dict(
                backbone_state, strict=False)
            print(f"[SDANet] Loaded pretrained backbone: {len(backbone_state)} keys "
                  f"(missing {len(missing)}, unexpected {len(unexpected)})")
        else:
            print(f"[SDANet] WARNING: no backbone keys found in {path}")

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) input image tensor.

        Returns:
            List of 3 prediction tensors, each:
            (B, num_anchors * box_fields, H_s, W_s) where box_fields = 6.
        """
        features = self.backbone(x)
        fpn_feats = self.neck(features)
        preds = self.head(fpn_feats)
        return preds


# Smoke test
if __name__ == "__main__":
    from config import DEVICE, INPUT_SIZE

    print(f"Backend: {BACKEND}, Device: {DEVICE}")
    print(f"Base kernels: {SDA_BASE_KERNELS}")

    model = SDANet(sss_enabled=False)
    if DEVICE == "cuda":
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_params:.2f}M  |  Trainable: {trainable:.2f}M")

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    if DEVICE == "cuda":
        dummy = dummy.cuda()

    with torch.no_grad():
        preds = model(dummy)

    print(f"\nPredictions ({INPUT_SIZE}x{INPUT_SIZE} input):")
    for i, p in enumerate(preds):
        print(f"  scale {i} (stride {model.strides[i]}): {p.shape}")

    A = model.num_anchors
    F = 6
    for s, p in zip(model.strides, preds):
        H_s = INPUT_SIZE // s
        assert p.shape == (1, A * F, H_s, H_s), (
            f"Unexpected shape: {p.shape} vs {(1, A*F, H_s, H_s)}")
    print("\nAll output shapes correct")
