"""
Spatial Distortion-Aware Convolution (SDABlock).

From "Towards Better Distortion Feature Learning for Object Detection
in Top-View Fisheye Cameras" (Guo et al., IEEE TMM 2026).

The SDABlock predefines several base convolution kernels of different
sizes and, based on the input's distortion characteristics, learns
aggregation coefficients to dynamically generate the most suitable
convolution kernel.

Supports both PyTorch and Jittor backends (controlled by config.BACKEND).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import BACKEND, USE_ACCUMULATION_STEP, GN_NUM_GROUPS

from model.op.functional import relu, pad, cat, normal_, constant_

if BACKEND == "pytorch":
    import torch.nn as nn
    Module = nn.Module
else:
    import jittor.nn as nn
    Module = nn.Module


# ------------------------------------------------------------------
#  Norm helpers — GroupNorm when accumulating gradients, else BatchNorm
# ------------------------------------------------------------------

def _norm2d(num_channels: int):
    """Return a 2D normalization layer.

    When ``USE_ACCUMULATION_STEP`` is True, small per-step batch sizes
    make BatchNorm unstable, so we use GroupNorm instead.
    """
    if USE_ACCUMULATION_STEP:
        groups = min(GN_NUM_GROUPS, num_channels)
        # Ensure groups divides channels
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


class SDAConv(Module):
    """Spatial Distortion-Aware Convolution (SDABlock).

    Predefines ``num_kernels`` base convolution kernels of different sizes.
    A small "coefficient predict" sub-network (GAP → FC → ReLU → FC → softmax)
    learns per-input aggregation weights, producing a dynamic kernel tailored
    to the current spatial distortion.

    Args:
        in_channels:   Input channels.
        out_channels:  Output channels.
        base_kernels:  Tuple of kernel sizes, e.g. ``(1, 3, 5)``.
        stride:        Convolution stride.
        padding:        Per-kernel padding (auto-computed from kernel size
                        if ``None``).
        fc_ratio:      FC hidden dim = in_channels / fc_ratio.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 base_kernels: tuple = (1, 3, 5), stride: int = 1,
                 padding: list | None = None, fc_ratio: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_kernels = base_kernels
        self.stride = stride
        self.num_kernels = len(base_kernels)

        # Padding: same by default
        if padding is None:
            self.padding = [k // 2 for k in base_kernels]
        else:
            self.padding = padding

        # ---- Coefficient Predict Part (Eq. 1 in the paper) ----
        hidden_dim = max(1, in_channels // fc_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_kernels)

        # ---- Base kernels (conv weights for each kernel size) ----
        self.base_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, ks, stride=stride,
                      padding=pad, bias=False)
            for ks, pad in zip(base_kernels, self.padding)
        ])

        # ---- BatchNorm / GroupNorm + Activation ----
        self.bn = _norm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            if hasattr(m, 'weight'):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, out_channels, H, W) distortion-aware features.
        """
        B = x.shape[0]

        # ---- 1. Coefficient prediction (Eq. 1) ----
        # M ∈ R^{B × num_kernels}
        pooled = self.gap(x).view(B, -1)
        M = relu(self.fc1(pooled))
        M = self.fc2(M)
        M = nn.Softmax(dim=1)(M)  # (B, num_kernels)

        # ---- 2. Kernel generation (Eq. 2) ----
        # Aggregate base conv outputs via iterative weighted sum:
        #   F̂ = Σ m_i * out_i                                          (Eq. 3)
        #
        # We accumulate in-place instead of stacking to avoid holding
        # num_kernels large intermediate tensors simultaneously.
        F_hat = None
        for i, conv in enumerate(self.base_convs):
            out_i = conv(x)                          # (B, C_out, H, W)
            w = M[:, i].view(-1, 1, 1, 1)            # broadcast weight
            if F_hat is None:
                F_hat = out_i * w
            else:
                F_hat = F_hat + out_i * w

        # ---- 3. BatchNorm + Activation ----
        F_hat = self.bn(F_hat)
        return self.act(F_hat)

    def execute(self, x):
        """Jittor entry point — delegates to forward."""
        return self.forward(x)

    def __repr__(self):
        return (f"SDAConv(in={self.in_channels}, out={self.out_channels}, "
                f"base_kernels={self.base_kernels})")


class SDAConvBlock(Module):
    """Conv-BN-ReLU → SDAConv-BN-ReLU residual block (CBL + SDABlock).

    Replaces the standard ResBlock with a distortion-aware variant.
    Follows the paper's Fig. 4 structure.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 base_kernels: tuple = (1, 3, 5), fc_ratio: int = 4):
        super().__init__()
        mid_channels = out_channels // 2

        self.cbl1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            _norm2d(mid_channels),
            nn.LeakyReLU(0.1),
        )
        self.sda = SDAConv(mid_channels, out_channels,
                           base_kernels=base_kernels, fc_ratio=fc_ratio)

        # Residual connection: if channel dims mismatch, project with 1×1 conv
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        identity = x
        out = self.cbl1(x)
        out = self.sda(out)
        if self.use_residual:
            out = out + identity
        else:
            out = out + self.residual_proj(identity)
        return out

    def execute(self, x):
        """Jittor entry point — delegates to forward."""
        return self.forward(x)


class SpatiallySeparateSDA(Module):
    """Spatially Separate Strategy (SSS) wrapper.

    Splits the input feature map into an N×N grid and applies an
    independent SDAConv to each cell.  Results are concatenated back
    into the full spatial layout.

    The paper demonstrates this strategy improves performance without
    adding FLOPS, since the per-cell convolutions can share the same
    runtime cost when implemented with grouped convolution.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 grid: int = 3, base_kernels: tuple = (1, 3, 5),
                 fc_ratio: int = 4):
        super().__init__()
        self.grid = grid
        self.in_channels = in_channels
        self.out_channels = out_channels

        # One SDABlock per grid cell
        self.cell_convs = nn.ModuleList([
            SDAConv(in_channels, out_channels // (grid * grid),
                    base_kernels=base_kernels, fc_ratio=fc_ratio)
            for _ in range(grid * grid)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.grid
        # Pad if needed
        pad_h = (g - H % g) % g
        pad_w = (g - W % g) % g
        if pad_h or pad_w:
            x = pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape
        cell_h, cell_w = Hp // g, Wp // g

        cells = []
        for i in range(g):
            for j in range(g):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = x[:, :, y1:y2, x1:x2]
                cell = self.cell_convs[i * g + j](cell)
                cells.append(cell)

        # Concatenate along channel dimension
        # Reshape spatially: cells → rows → hstack → vstack
        rows = []
        for i in range(g):
            row = cat(cells[i * g:(i + 1) * g], dim=3)  # concat W
            rows.append(row)
        out = cat(rows, dim=2)  # concat H

        # Crop back to original size
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out

    def execute(self, x):
        """Jittor entry point — delegates to forward."""
        return self.forward(x)
