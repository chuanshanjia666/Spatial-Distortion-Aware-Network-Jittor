# SDABlock 实现说明

## 概述

本文档说明 SDABlock（Spatial Distortion-Aware Block）两种等价格式化方案的实现差异，以及代码中采用的简化方案及其与论文的对应关系。

## 论文 Fig. 4 的描述

论文 Fig. 4 (Kernel Generate Part) 描述的 SDABlock 结构如下：

```
Input ──GAP──► FC₁ ──ReLU──► FC₂ ──softmax──► M = (m₁, m₂, m₃)
                                                         │
                               ┌─────────────────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   New Kernel Generation  │
                    │  w_new = Σ mᵢ · wᵢ      │
                    │  (w_i = base kernels)   │
                    └──────────┬──────────────┘
                               │
                               ▼
                         conv(x, w_new)
                               │
                               ▼
                            Output
```

**严格按论文的方式**：先生成新核（加权 base kernel weights），再做一次卷积。

## 代码中的实现

当前代码（SDAConv）采用的是**输出混合（Output Mixing）**方案：

```python
def forward(self, x):
    # 1. 系数预测（M）
    pooled = self.gap(x).view(B, -1)
    M = relu(self.fc1(pooled))
    M = self.fc2(M)
    M = nn.Softmax(dim=1)(M)

    # 2. 多核独立卷积后加权输出（Eq. 3）
    F_hat = None
    for i, conv in enumerate(self.base_convs):
        out_i = conv(x)                          # 3 次独立 conv
        w = M[:, i].view(-1, 1, 1, 1)
        if F_hat is None:
            F_hat = out_i * w
        else:
            F_hat = F_hat + out_i * w

    # 3. Norm + Activation
    F_hat = self.bn(F_hat)
    return self.act(F_hat)
```

## 两种方案的数学等价性

假设 $k_i$ 是第 $i$ 个 base kernel，$x$ 是输入：

**论文方案（权重混合）**：
```
w_new = Σ mᵢ · kᵢ
y = conv(x, w_new)
```

**代码方案（输出混合）**：
```
y = Σ mᵢ · conv(x, kᵢ)
```

**关键性质**：卷积运算对加权系数是**线性**的：
```
conv(x, Σ mᵢ · kᵢ) = Σ mᵢ · conv(x, kᵢ)
```

因此两种方案在数学上**严格等价**，前提是所有 base kernel 的 padding 使输出空间尺寸一致。

## 实现对比

| 维度 | 论文方案（权重混合） | 代码方案（输出混合） |
|------|---------------------|---------------------|
| 卷积次数 | 1 次 | 多次（等于 base kernel 数） |
| 计算量 | 略低 | 略高 |
| 显存占用 | 较小（单一 tensor） | 较高（多个中间结果） |
| 代码复杂度 | 高（需 pad weights、重新合成） | 低（直接循环即可） |
| ONNX 导出 | 需额外处理 | 直接支持 |

## 选择理由

代码选择**输出混合**方案的原因：

1. **实现简洁**：仅需循环执行多个卷积，无需复杂的权重 pad 和重组
2. **ONNX 兼容**：直接可导出，无需额外算子支持
3. **数学等价**：已证明两种方案在卷积运算上等价
4. **调试方便**：可独立查看每个 base kernel 的输出

## 论文与代码对照表

| 论文描述 | 代码实现 | 状态 |
|---------|---------|------|
| Eq. (1) 系数预测 M = softmax(FC₂(FC₁(GAP(F)))) | `gap → fc1 → relu → fc2 → softmax` | ✅ 一致 |
| Eq. (2) 新核生成 w_new = Σ mᵢ · wᵢ | 未直接实现（输出混合替代） | ⚠️ 等价简化 |
| Eq. (3) F̂ = conv(x, w_new) | 加权求和各核输出 | ⚠️ 等价简化 |

## 注意事项

- 本简化不影响训练 loss 收敛和最终 mAP 指标
- 推理速度略慢于论文严格方案，但差值在可接受范围内
- 如需完全对齐论文 Fig. 4，可将 `model/op/sdaconv.py` 中的 forward 改写为权重混合版本