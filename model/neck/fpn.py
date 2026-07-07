"""
FPN neck for SDANet.

Takes DarkNet53 backbone outputs at strides 8, 16, 32 and applies
top-down feature pyramid with lateral connections, then uses SDAConv
to extract distortion-aware features at each scale.

Top-down structure (from smallest scale to largest)::

    f32 → lat[2] ──→ (CBL+SDA)×2 → CBL ──────── p1 ───→ head[2] (stride 32)
                                              /
                        ┌─ upsample → CBL ──┘
    f16 → lat[1] ──────┤
                        └─ concat ──→ (CBL+SDA)×2 → CBL ──────── p2 ───→ head[1] (stride 16)
                                                              /
                            ┌─ upsample → CBL ───────────────┘
    f8  → lat[0] ──────────┤
                            └─ concat ──→ (CBL+SDA)×2 → CBL ── p3 ───→ head[0] (stride 8)

When ``config.USE_ACCUMULATION_STEP`` is True, GroupNorm replaces
BatchNorm so that small micro-batches remain stable.

Supports both PyTorch and Jittor backends.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import BACKEND, SDA_BASE_KERNELS, NECK_OUT_CHANNELS, USE_ACCUMULATION_STEP, GN_NUM_GROUPS, SDA_FC_RATIO

if BACKEND == "pytorch":
    import torch.nn as nn
else:
    import jittor.nn as nn

from model.op.sdaconv import SDAConv, _norm2d
from model.op.functional import interpolate, cat


def _CBL(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv → Norm → LeakyReLU."""
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False),
        _norm2d(out_ch),
        nn.LeakyReLU(0.1),
    )


class FPNNeck(nn.Module):
    """Feature Pyramid Network neck with SDAConv.

    Top-down pathway with concatenation-based fusion.  Each scale
    applies a (CBL + SDA) × 2 + CBL processing block to extract
    distortion-aware features before passing them to the head.

    Args:
        in_channels:  List of backbone output channels (from large to small scale
                      i.e. stride 8, 16, 32).
        out_channels: Output channel count per scale (default from config).
        base_kernels: SDAConv base kernel sizes.
        fc_ratio:     SDAConv FC hidden dim = in_channels / fc_ratio.
        use_sda:      If True, use SDAConv in processing blocks; else use
                      standard 3×3 CBL (for ablation experiments).
    """

    def __init__(self, in_channels=None, out_channels=None,
                 base_kernels=None, fc_ratio=None, use_sda=True):
        super().__init__()
        in_channels = in_channels or [256, 512, 1024]
        out_channels = out_channels or NECK_OUT_CHANNELS
        base_kernels = base_kernels or SDA_BASE_KERNELS
        fc_ratio = fc_ratio if fc_ratio is not None else SDA_FC_RATIO

        self.out_channels = out_channels
        self.num_scales = len(in_channels)
        self.use_sda = use_sda
        self.base_kernels = base_kernels

        self.fc_ratio = fc_ratio

        # ---- Lateral connections: 1×1 CBL to unify channel count ----
        # in_channels  order: [stride_8, stride_16, stride_32]
        # laterals[0] = stride  8  (largest,  used as p3 lateral)
        # laterals[1] = stride 16  (medium,   used as p2 lateral)
        # laterals[2] = stride 32  (smallest, used as p1 lateral)
        self.laterals = nn.ModuleList([
            _CBL(c, out_channels, 1) for c in in_channels
        ])

        # ---- p1 (stride 32, smallest scale): process lateral directly ----
        self.p1_process = self._make_process_block(out_channels, out_channels)

        # ---- p2 (stride 16): upsample_CBL(p1_out) + concat(lateral) → process ----
        self.p2_up_cbl = _CBL(out_channels, out_channels, 3)
        self.p2_process = self._make_process_block(out_channels * 2, out_channels)

        # ---- p3 (stride 8, largest scale): upsample_CBL(p2_out) + concat(lateral) → process ----
        self.p3_up_cbl = _CBL(out_channels, out_channels, 3)
        self.p3_process = self._make_process_block(out_channels * 2, out_channels)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _make_sda(self, in_ch: int, out_ch: int):
        """SDAConv when ``use_sda`` is True, else a fallback 3×3 CBL."""
        if self.use_sda:
            return SDAConv(in_ch, out_ch, base_kernels=self.base_kernels,
                           fc_ratio=self.fc_ratio)
        else:
            return _CBL(in_ch, out_ch, 3)

    def _make_process_block(self, in_ch: int, out_ch: int):
        """Build the per-scale processing block: (CBL + SDA) × 2 + CBL."""
        return nn.Sequential(
            _CBL(in_ch, out_ch, 3),
            self._make_sda(out_ch, out_ch),
            _CBL(out_ch, out_ch, 3),
            self._make_sda(out_ch, out_ch),
            _CBL(out_ch, out_ch, 3),
        )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, features):
        """
        Args:
            features: List of backbone feats [f8, f16, f32] (large to small scale).

        Returns:
            List of distortion-aware FPN features [p3, p2, p1]
            (large to small scale), each with ``out_channels`` channels.
        """
        # Lateral 1×1 CBL on each backbone feature
        lat = [l(f) for l, f in zip(self.laterals, features)]
        # lat[0] = stride  8  → p3 lateral
        # lat[1] = stride 16  → p2 lateral
        # lat[2] = stride 32  → p1 lateral

        # ---- p1: stride 32 (smallest) ----
        p1 = self.p1_process(lat[2])                     # (CBL+SDA)×2 + CBL

        # ---- p2: stride 16 (medium) ----
        p1_up = interpolate(p1, scale_factor=2, mode='nearest')
        p1_up = self.p2_up_cbl(p1_up)                    # upsample → CBL
        p2_cat = cat([p1_up, lat[1]], dim=1)             # concat with p2 lateral
        p2 = self.p2_process(p2_cat)                     # (CBL+SDA)×2 + CBL

        # ---- p3: stride 8 (largest) ----
        p2_up = interpolate(p2, scale_factor=2, mode='nearest')
        p2_up = self.p3_up_cbl(p2_up)                    # upsample → CBL
        p3_cat = cat([p2_up, lat[0]], dim=1)             # concat with p3 lateral
        p3 = self.p3_process(p3_cat)                     # (CBL+SDA)×2 + CBL

        return [p3, p2, p1]  # large → small spatial size

    def execute(self, features):
        """Jittor entry point — delegates to forward."""
        return self.forward(features)


# Smoke test
if __name__ == "__main__":
    from config import DEVICE

    if BACKEND == "pytorch":
        import torch

    neck = FPNNeck(in_channels=[256, 512, 1024], out_channels=128)
    if DEVICE == "cuda":
        neck = neck.cuda()

    if BACKEND == "pytorch":
        f8  = torch.randn(1, 256, 80, 80)
        f16 = torch.randn(1, 512, 40, 40)
        f32 = torch.randn(1, 1024, 20, 20)
    else:
        import jittor as jt
        f8  = jt.random([1, 256, 80, 80])
        f16 = jt.random([1, 512, 40, 40])
        f32 = jt.random([1, 1024, 20, 20])
    if DEVICE == "cuda":
        f8, f16, f32 = f8.cuda(), f16.cuda(), f32.cuda()

    outs = neck([f8, f16, f32])
    for i, o in enumerate(outs):
        print(f"  neck out {i}: {o.shape}")
    print(f"  total params: {sum(p.numel() for p in neck.parameters())/1e6:.2f}M")
