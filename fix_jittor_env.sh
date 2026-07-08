#!/bin/bash
# Jittor CUDA 环境修复脚本
# 解决 conda cuda 环境缺失的头文件和库文件问题

set -e

CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base 2>/dev/null)/envs/sdanet}"
echo "Jittor CUDA 环境修复 - CONDA_PREFIX: $CONDA_PREFIX"

# 1. 创建 lib64 目录并链接 CUDA 运行时库
echo "[1/4] 链接 CUDA 运行时库 (lib -> lib64)..."
mkdir -p "$CONDA_PREFIX/lib64"
ln -sf "$CONDA_PREFIX/lib/libcudart.so" "$CONDA_PREFIX/lib64/libcudart.so" 2>/dev/null || true

# 2. 链接所有 cuDNN 库到 lib64
echo "[2/4] 链接 cuDNN 库 (lib -> lib64)..."
for f in "$CONDA_PREFIX/lib/libcudnn"*; do
    [ -f "$f" ] && ln -sf "$f" "$CONDA_PREFIX/lib64/$(basename "$f")" 2>/dev/null || true
done

# 3. 链接 CUDA 头文件到 include 目录
echo "[3/4] 链接 CUDA 头文件 (targets -> include)..."
CUDA_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
for f in "$CUDA_INCLUDE"/*; do
    name=$(basename "$f")
    if [ ! -e "$CONDA_PREFIX/include/$name" ]; then
        ln -sf "$f" "$CONDA_PREFIX/include/$name"
    fi
done


echo ""
echo "修复完成！请运行: python -m jittor.test.test_example"