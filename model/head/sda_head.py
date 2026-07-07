"""
Detection head for SDANet with oriented bounding box prediction.

Uses SDAConv blocks to extract distortion-aware features at each
FPN scale, then predicts oriented bounding boxes:

    [cx, cy, w, h, theta, obj_conf] per anchor x per grid cell.

When ``config.USE_ACCUMULATION_STEP`` is True, GroupNorm replaces
BatchNorm so that small micro-batches remain stable.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (BACKEND, SDA_BASE_KERNELS, NECK_OUT_CHANNELS,
                    NUM_ANCHORS, BOX_FIELDS, SSS_ENABLED, SSS_GRID,
                    SDA_FC_RATIO)

if BACKEND == "pytorch":
    import torch
    import torch.nn as nn
else:
    import jittor as jt
    import jittor.nn as nn

from model.op.sdaconv import SDAConv, SpatiallySeparateSDA, _norm2d


def _CBL(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv → Norm → LeakyReLU."""
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False),
        _norm2d(out_ch),
        nn.LeakyReLU(0.1),
    )


class SDAHead(nn.Module):
    """Oriented bounding box detection head with SDAConv.

    Args:
        in_channels:  Channel count per scale (same as neck output).
        num_anchors:  Anchors per scale.
        num_classes:  Number of classes.
        base_kernels: SDAConv base kernel sizes.
        fc_ratio:     SDAConv FC hidden dim = in_channels / fc_ratio.
        sss_enabled:  Enable Spatially Separate Strategy.
        sss_grid:     Grid size for SSS.
    """

    def __init__(self, in_channels=None, num_anchors=None,
                 base_kernels=None, fc_ratio=None,
                 sss_enabled=None, sss_grid=None):
        super().__init__()
        in_channels = in_channels or NECK_OUT_CHANNELS
        num_anchors = num_anchors or NUM_ANCHORS
        base_kernels = base_kernels or SDA_BASE_KERNELS
        fc_ratio = fc_ratio if fc_ratio is not None else SDA_FC_RATIO
        sss_enabled = sss_enabled if sss_enabled is not None else SSS_ENABLED
        sss_grid = sss_grid or SSS_GRID

        out_per_anchor = BOX_FIELDS  # cx, cy, w, h, theta, conf
        out_channels = num_anchors * out_per_anchor

        if sss_enabled:
            self.sda_conv = nn.Sequential(
                SpatiallySeparateSDA(in_channels, in_channels,
                                     grid=sss_grid, base_kernels=base_kernels,
                                     fc_ratio=fc_ratio),
            )
        else:
            self.sda_conv = SDAConv(in_channels, in_channels,
                                    base_kernels=base_kernels,
                                    fc_ratio=fc_ratio)

        self.cbl = _CBL(in_channels, in_channels, 3)

        self.pred_conv = nn.Conv2d(in_channels, out_channels, 1,
                                   stride=1, padding=0, bias=True)
        self.num_anchors = num_anchors
        self.out_per_anchor = out_per_anchor

    def forward(self, x):
        x = self.sda_conv(x)
        x = self.cbl(x)
        return self.pred_conv(x)

    def execute(self, x):
        """Jittor entry point — delegates to forward."""
        return self.forward(x)


class SDAHeadMulti(nn.Module):
    """Per-scale SDAHead (one independent head per FPN scale).

    Each scale gets its own SDAHead with dedicated distortion learning,
    matching Fig. 3 in the paper.
    """

    def __init__(self, num_scales=3, in_channels=None, num_anchors=None,
                 base_kernels=None, fc_ratio=None, sss_enabled=None):
        super().__init__()
        self.heads = nn.ModuleList([
            SDAHead(in_channels, num_anchors, base_kernels, fc_ratio, sss_enabled)
            for _ in range(num_scales)
        ])

    def forward(self, features):
        return [h(f) for h, f in zip(self.heads, features)]

    def execute(self, features):
        """Jittor entry point — delegates to forward."""
        return self.forward(features)


# Smoke test
if __name__ == "__main__":
    from config import DEVICE

    head = SDAHeadMulti(num_scales=3)
    if DEVICE == "cuda":
        head = head.cuda()

    if BACKEND == "pytorch":
        import torch
        f8  = torch.randn(1, 128, 80, 80)
        f16 = torch.randn(1, 128, 40, 40)
        f32 = torch.randn(1, 128, 20, 20)
    else:
        import jittor as jt
        f8  = jt.random([1, 128, 80, 80])
        f16 = jt.random([1, 128, 40, 40])
        f32 = jt.random([1, 128, 20, 20])
    if DEVICE == "cuda":
        f8, f16, f32 = f8.cuda(), f16.cuda(), f32.cuda()

    preds = head([f8, f16, f32])
    for i, p in enumerate(preds):
        print(f"  pred {i}: {p.shape}  (A={p.shape[1] // 6})")
    print(f"  total params: {sum(p.numel() for p in head.parameters())/1e6:.2f}M")
