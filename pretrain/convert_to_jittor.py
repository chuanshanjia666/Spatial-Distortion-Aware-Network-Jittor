#!/usr/bin/env python3
"""
Convert pretrained weights between PyTorch and Jittor formats.

Supported conversions:
  - PyTorch (.pth)  →  Jittor (.pkl)
  - Jittor  (.pkl)  →  PyTorch (.pth)

Usage:
    # PyTorch → Jittor
    python pretrain/convert_to_jittor.py

    # Jittor → PyTorch
    python pretrain/convert_to_jittor.py --reverse

    # Custom paths
    python pretrain/convert_to_jittor.py \\
        --src pretrain/darknet53_backbone.pth \\
        --dst pretrain/darknet53_backbone.pkl
"""

import os
import argparse
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))


def _pth_to_pkl(src_path: str, dst_path: str):
    """PyTorch (.pth) → Jittor (.pkl)."""
    import torch
    assert os.path.isfile(src_path), f"PyTorch checkpoint not found: {src_path}"

    print(f"Loading PyTorch checkpoint: {src_path}")
    state = torch.load(src_path, map_location="cpu", weights_only=True)

    # Flatten MMDetection state_dict wrapper if present
    if isinstance(state, dict) and len(state) <= 3 and "state_dict" in state:
        state = state["state_dict"]

    jt_state = {}
    for k, v in state.items():
        jt_state[k] = v.detach().cpu().numpy()

    print(f"Writing Jittor checkpoint ({len(jt_state)} keys): {dst_path}")
    with open(dst_path, "wb") as f:
        pickle.dump(jt_state, f)

    print(f"  {os.path.getsize(dst_path) / 1024 / 1024:.1f} MB")


def _pkl_to_pth(src_path: str, dst_path: str):
    """Jittor (.pkl) → PyTorch (.pth)."""
    import torch
    assert os.path.isfile(src_path), f"Jittor checkpoint not found: {src_path}"

    print(f"Loading Jittor checkpoint: {src_path}")
    with open(src_path, "rb") as f:
        jt_state = pickle.load(f)

    pt_state = {}
    for k, v in jt_state.items():
        pt_state[k] = torch.from_numpy(v)

    print(f"Writing PyTorch checkpoint ({len(pt_state)} keys): {dst_path}")
    torch.save(pt_state, dst_path)

    print(f"  {os.path.getsize(dst_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pretrained weights between PyTorch (.pth) and Jittor (.pkl)")
    parser.add_argument("--src", default=None, help="Source checkpoint path")
    parser.add_argument("--dst", default=None, help="Destination checkpoint path")
    parser.add_argument("--reverse", action="store_true",
                        help="Convert Jittor .pkl → PyTorch .pth (default: .pth → .pkl)")
    args = parser.parse_args()

    # Auto-detect paths
    default_pth = os.path.join(HERE, "darknet53_backbone.pth")
    default_pkl = os.path.join(HERE, "darknet53_backbone.pkl")

    if args.reverse:
        src = args.src or default_pkl
        dst = args.dst or default_pth
        _pkl_to_pth(src, dst)
    else:
        src = args.src or default_pth
        dst = args.dst or default_pkl
        _pth_to_pkl(src, dst)

    print("Done.")


if __name__ == "__main__":
    main()
