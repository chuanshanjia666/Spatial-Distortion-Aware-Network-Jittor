"""
Unified functional operators — abstracts PyTorch / Jittor backend differences.

Usage::

    from model.op.functional import relu, pad, interpolate, cat

Both backends are supported transparently.
"""

from config import BACKEND

if BACKEND == "pytorch":
    import torch
    import torch.nn.functional as F
else:
    import jittor as jt
    import jittor.nn as jnn


# ------------------------------------------------------------------
#  Activation
# ------------------------------------------------------------------

def relu(x):
    if BACKEND == "pytorch":
        return F.relu(x)
    return jnn.relu(x)


# ------------------------------------------------------------------
#  Padding
# ------------------------------------------------------------------

def pad(x, pad):
    """Pad with constant zero. ``pad`` is (left, right, top, bottom, …)."""
    if BACKEND == "pytorch":
        return F.pad(x, pad)
    return jnn.pad(x, pad)


# ------------------------------------------------------------------
#  Upsampling
# ------------------------------------------------------------------

def interpolate(x, scale_factor=None, size=None, mode='nearest'):
    if BACKEND == "pytorch":
        return F.interpolate(x, scale_factor=scale_factor, size=size, mode=mode)
    return jnn.interpolate(x, scale_factor=scale_factor, size=size, mode=mode)


# ------------------------------------------------------------------
#  Concatenation
# ------------------------------------------------------------------

def cat(tensors, dim=0):
    if BACKEND == "pytorch":
        return torch.cat(tensors, dim=dim)
    return jt.cat(tensors, dim=dim)


# ------------------------------------------------------------------
#  Weight initialisation
# ------------------------------------------------------------------

def normal_(tensor, mean=0, std=1):
    """In-place normal fill."""
    if BACKEND == "pytorch":
        import torch.nn.init as init
        return init.normal_(tensor, mean, std)
    return jt.init.gauss_(tensor, mean, std)


def constant_(tensor, val):
    """In-place constant fill."""
    if BACKEND == "pytorch":
        import torch.nn.init as init
        return init.constant_(tensor, val)
    return jt.init.constant_(tensor, val)


def kaiming_normal_(tensor, mode='fan_out', nonlinearity='leaky_relu'):
    """In-place Kaiming normal initialisation."""
    if BACKEND == "pytorch":
        import torch.nn.init as init
        return init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)
    return jt.init.relu_invariant_gauss_(tensor, mode=mode)
