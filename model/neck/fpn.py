"""
FPN neck for SDANet.

Takes DarkNet53 backbone outputs at strides 8, 16, 32 and applies
top-down feature pyramid with lateral connections, then uses SDAConv
to extract distortion-aware features at each scale.

When ``config.USE_ACCUMULATION_STEP`` is True, GroupNorm replaces
BatchNorm so that small micro-batches remain stable.

Supports both PyTorch and Jittor backends.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import BACKEND, SDA_BASE_KERNELS, NECK_OUT_CHANNELS, USE_ACCUMULATION_STEP, GN_NUM_GROUPS

if BACKEND == "pytorch":
    import torch.nn as nn
else:
    import jittor.nn as nn

from model.op.sdaconv import SDAConv, _norm2d
from model.op.functional import interpolate


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

    Args:
        in_channels:  List of backbone output channels (from small to large scale).
        out_channels: Output channel count per scale (default from config).
        base_kernels: SDAConv base kernel sizes.
        use_sda:      If True, use SDAConv in lateral/top-down; else use
                      standard 1x1 CBL (for ablation experiments).
    """

    def __init__(self, in_channels=None, out_channels=None,
                 base_kernels=None, use_sda=True):
        super().__init__()
        in_channels = in_channels or [256, 512, 1024]
        out_channels = out_channels or NECK_OUT_CHANNELS
        base_kernels = base_kernels or SDA_BASE_KERNELS

        self.num_scales = len(in_channels)

        # Lateral connections: 1x1 to unify channel count
        self.laterals = nn.ModuleList([
            _CBL(c, out_channels, 1) for c in in_channels
        ])

        # Top-down: after upsampling fusion, conv to smooth
        self.td_convs = nn.ModuleList()
        for _ in range(self.num_scales - 1):
            self.td_convs.append(_CBL(out_channels, out_channels, 3))

        # SDAConv at each scale to extract distortion features
        if use_sda:
            self.sda_convs = nn.ModuleList([
                SDAConv(out_channels, out_channels, base_kernels=base_kernels)
                for _ in range(self.num_scales)
            ])
        else:
            self.sda_convs = nn.ModuleList([
                _CBL(out_channels, out_channels, 3)
                for _ in range(self.num_scales)
            ])

    def forward(self, features):
        """
        Args:
            features: List of backbone feats [f8, f16, f32] (small to large scale).

        Returns:
            List of distortion-aware FPN features, same length, channels=out_channels.
        """
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        for i in range(self.num_scales - 1, 0, -1):
            up = interpolate(laterals[i], scale_factor=2, mode='nearest')
            laterals[i - 1] = laterals[i - 1] + up

        for i in range(self.num_scales - 1):
            laterals[i] = self.td_convs[i](laterals[i])

        outs = [sda(l) for sda, l in zip(self.sda_convs, laterals)]
        return outs

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
