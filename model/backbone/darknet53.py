"""
DarkNet53 backbone for fisheye object detection.

Adapted from YOLOv3's DarkNet53.  Outputs multi-scale features at
strides 8, 16, 32 (used by the FPN neck and detection head).

When ``config.USE_ACCUMULATION_STEP`` is True, GroupNorm replaces
BatchNorm so that small micro-batches remain stable.

Supports both PyTorch and Jittor backends.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import BACKEND, USE_ACCUMULATION_STEP, GN_NUM_GROUPS

if BACKEND == "pytorch":
    import torch.nn as nn
else:
    import jittor.nn as nn

from model.op.functional import constant_, normal_, kaiming_normal_


# ------------------------------------------------------------------
#  Norm helper
# ------------------------------------------------------------------

def _norm2d(num_channels: int):
    if USE_ACCUMULATION_STEP:
        groups = min(GN_NUM_GROUPS, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


# ------------------------------------------------------------------
#  Basic building blocks
# ------------------------------------------------------------------

def _CBL(in_ch, out_ch, kernel_size=3, stride=1, **kwargs):
    """Conv → Norm → LeakyReLU."""
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False, **kwargs),
        _norm2d(out_ch),
        nn.LeakyReLU(0.1),
    )


class ResBlock(nn.Module):
    """Standard residual block used in DarkNet53 (CBL → CBL + shortcut)."""

    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.cbl1 = _CBL(channels, mid, 1)
        self.cbl2 = _CBL(mid, channels, 3)

    def forward(self, x):
        return x + self.cbl2(self.cbl1(x))

    def execute(self, x):
        """Jittor entry point — delegates to forward."""
        return self.forward(x)


# ------------------------------------------------------------------
#  DarkNet53
# ------------------------------------------------------------------

class DarkNet53(nn.Module):
    """DarkNet53 feature extractor.

    Outputs three scale features for FPN::

        out0  — stride  8,  channels[2]
        out1  — stride 16,  channels[3]
        out2  — stride 32,  channels[4]

    Args:
        filters:     Base channel sizes per stage.
                     Default [32, 64, 128, 256, 512, 1024].
        repeats:     ResBlocks per stage.
                     Default [1, 2, 8, 8, 4].
        out_indices: Indices of stage outputs to return.
                     Default (2, 3, 4) → strides 8, 16, 32.
    """

    def __init__(self, filters=None, repeats=None, out_indices=(2, 3, 4)):
        super().__init__()
        filters = filters or [32, 64, 128, 256, 512, 1024]
        repeats = repeats or [1, 2, 8, 8, 4]
        self.out_indices = out_indices

        # Stem: Conv → Norm → LeakyReLU (matches MMDetection's backbone.conv1)
        self.stem = nn.Sequential(
            nn.Conv2d(3, filters[0], 3, stride=1, padding=1, bias=False),
            _norm2d(filters[0]),
            nn.LeakyReLU(0.1),
        )

        # Stages
        in_ch = filters[0]
        self.stages = nn.ModuleList()
        for i, (f, r) in enumerate(zip(filters[1:], repeats)):
            stage = nn.Sequential()
            stage.add_module(f"ds_{i}", _CBL(in_ch, f, 3, stride=2))
            for j in range(r):
                stage.add_module(f"res_{i}_{j}", ResBlock(f))
            self.stages.append(stage)
            in_ch = f

        # Output channels for each returned stage
        self.out_channels = [filters[i + 1] for i in out_indices]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out',
                                nonlinearity='leaky_relu')
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        """Returns list of feature maps at each ``out_indices`` stage."""
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def execute(self, x):
        """Jittor entry point — delegates to forward."""
        return self.forward(x)


# Smoke test
if __name__ == "__main__":
    from config import DEVICE, INPUT_SIZE

    if BACKEND == "pytorch":
        import torch

    model = DarkNet53()
    if DEVICE == "cuda":
        model = model.cuda()

    if BACKEND == "pytorch":
        x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    else:
        import jittor as jt
        x = jt.random([2, 3, INPUT_SIZE, INPUT_SIZE])
    if DEVICE == "cuda":
        x = x.cuda()
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"  out {i}: {o.shape}")
    print(f"  out_channels: {model.out_channels}")
