#!/usr/bin/env python3
"""
绘制所有训练日志的 loss 曲线对比图 (2x2 布局)
"""
import re
import os
import matplotlib.pyplot as plt

LOG_DIR = "log"

# 配对的 log 文件列表
LOG_PAIRS = [
    ("train_torch_ha.log", "train_log_jittor_ha.log", "HA Dataset"),
    ("train_torch_ce.log", "train_log_jittor_ce.log", "CE Dataset"),
    ("train_torch_we.log", "train_log_jittor_we.log", "WE Dataset"),
    ("train_torch_ce_we_ha.log", "train_log_jittor_ce_we_ha.log", "CE+WE+HA Dataset"),
]


def parse_log(filepath):
    """解析训练日志，提取 iter 和 loss"""
    pattern = r"epoch\s+(\d+)\s+\|\s+iter\s+(\d+)/(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e-]+)"

    iterations = []
    losses = []

    with open(filepath, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iterations.append(int(match.group(2)))
                losses.append(float(match.group(4)))

    return iterations, losses


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    max_iter = 2000  # 只统计 2000 轮

    for idx, (torch_log, jittor_log, title) in enumerate(LOG_PAIRS):
        ax = axes[idx]

        # 绘制 PyTorch
        torch_path = os.path.join(LOG_DIR, torch_log)
        if os.path.exists(torch_path):
            iters, losses = parse_log(torch_path)
            # 截取到 max_iter
            filtered = [(i, l) for i, l in zip(iters, losses) if i <= max_iter]
            if filtered:
                iters, losses = zip(*filtered)
            ax.plot(iters, losses, 'r-', linewidth=1.2, alpha=0.8, label='PyTorch')
            print(f"{torch_log}: {len(iters)} iterations, final loss: {losses[-1]:.4f}")
        else:
            print(f"{torch_log}: not found")

        # 绘制 Jittor
        jittor_path = os.path.join(LOG_DIR, jittor_log)
        if os.path.exists(jittor_path):
            iters, losses = parse_log(jittor_path)
            # 截取到 max_iter
            filtered = [(i, l) for i, l in zip(iters, losses) if i <= max_iter]
            if filtered:
                iters, losses = zip(*filtered)
            ax.plot(iters, losses, 'b-', linewidth=1.2, alpha=0.8, label='Jittor')
            print(f"{jittor_log}: {len(iters)} iterations, final loss: {losses[-1]:.4f}")
        else:
            print(f"{jittor_log}: not found")

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=max_iter)

    plt.suptitle('PyTorch vs Jittor Training Loss Comparison (First 2000 iterations)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("log/training_curves_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: log/training_curves_comparison.png")


if __name__ == "__main__":
    main()