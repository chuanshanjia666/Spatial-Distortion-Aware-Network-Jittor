# PFDAug: Prominent Fisheye Distortion Augmentation

## 概述

**PFDAug**（Prominent Fisheye Distortion Augmentation）是论文 *Towards Better Distortion Feature Learning for Object Detection in Top-View Fisheye Cameras* 中提出的一种专门针对鱼眼图像的**数据增强方法**。

## 动机

在真实的鱼眼图像数据集中，存在两个关键问题：

1. **数据稀少**：鱼眼图像数据集整体规模较小
2. **空间分布不均**：大多数物体位于图像中心附近（畸变轻微），边缘区域（畸变严重）的样本极少

论文图2的统计显示，CEPDOF 数据集中距离中心 $\gt 800$ 像素的外围区域样本非常稀少。这导致模型缺乏对**严重畸变特征**的学习，无法有效检测外围物体。

PFDAug 通过对鱼眼图像**额外引入鱼眼畸变**，增加严重畸变样本的数量，从而增强模型对畸变特征的泛化能力。

## 核心原理：桶形畸变模型

PFDAug 使用**桶形畸变（Barrel Distortion）**来模拟鱼眼镜头的畸变效果，其数学模型为：

$$r_u = r_d \cdot (1 + k \cdot r_d^2)$$

其中：
- $r_u$：畸变后像素到畸变中心的距离
- $r_d$：畸变前像素到畸变中心的距离
- $k$：畸变系数（$k \gt 0$ 时产生桶形畸变，值越大畸变越强）

## 算法细节

### Algorithm 1: PFDAug 坐标变换算法

**输入**：有向边界框 $\text{obbox} = (x, y, w, h, \theta)$，畸变系数 $k$

**输出**：畸变后的有向边界框 $(x', y', w', h', \theta')$

**步骤1：计算边界框的四个角点坐标**

以中心坐标 $(x, y)$、宽 $w$、高 $h$、旋转角 $\theta$ 计算四个角点在旋转后的坐标：

$$
\begin{aligned}
x_i' &= x + dx_i \cdot \cos\theta - dy_i \cdot \sin\theta \\
y_i' &= y + dx_i \cdot \sin\theta + dy_i \cdot \cos\theta
\end{aligned}
$$

其中 $(dx_i, dy_i)$ 为以中心为原点的四个象限点：

$$
\left(-\frac{w}{2}, -\frac{h}{2}\right),\;
\left(+\frac{w}{2}, -\frac{h}{2}\right),\;
\left(-\frac{w}{2}, +\frac{h}{2}\right),\;
\left(+\frac{w}{2}, +\frac{h}{2}\right)
$$

**步骤2：对每个角点应用点畸变算法（Algorithm 2）**

**步骤3：计算最小外接旋转矩形**

使用 OpenCV 的 `cv2.minAreaRect` 从四个畸变角点计算新的有向边界框。

### Algorithm 2: 点畸变算法

**输入**：像素坐标 $(x, y)$，图像宽高 $(W, H)$，畸变系数 $k$

**输出**：畸变后的坐标 $(x_\text{dis}, y_\text{dis})$

**步骤1：归一化到图像中心坐标系**

$$
x' = \frac{x - W/2}{W/2},\quad y' = \frac{y - H/2}{H/2}
$$

**步骤2：计算极坐标**

$$
r = \sqrt{x'^2 + y'^2},\quad \theta = \operatorname{atan2}(y', x')
$$

**步骤3：应用桶形畸变**

$$
r_\text{dis} = r \cdot (1 + k \cdot r^2)
$$

**步骤4：反归一化回像素坐标**

$$
\begin{aligned}
x_\text{dis} &= r_\text{dis} \cdot \cos\theta \cdot \frac{W}{2} + \frac{W}{2} \\[4pt]
y_\text{dis} &= r_\text{dis} \cdot \sin\theta \cdot \frac{H}{2} + \frac{H}{2}
\end{aligned}
$$

## 图像畸变（Image Warp）

对整张图像使用**逆向映射（Backward Mapping）**：

对于输出图像中每个像素（距离 $r_u$），求解其在原始图像中对应位置的距离 $r_d$：

$$k \cdot r_d^{\,3} + r_d - r_u = 0$$

使用牛顿法求解，迭代公式：

$$r_d \leftarrow r_d - \frac{k \cdot r_d^{\,3} + r_d - r_u}{3k \cdot r_d^{\,2} + 1}$$

收敛后将 $r_d$ 映射回原始图像对应像素位置。

## 参数选择

- $k = 0$：无额外畸变
- $0 \lt k \le 0.5$：适中畸变，性能提升显著
- $k = 0.8$：过度畸变，常规场景性能下降（困难场景仍有提升）

论文推荐 **$k = 0.5$** 作为默认值。

## 效果

- CEPDOF 数据集：$\text{AP}_{50}$ 提升 $3.6\%$（RAPiD）/ $2.9\%$（SDANet）
- all-off 子数据集（最困难场景）：提升高达 $12.8\%$ / $12.4\%$
- 使小模型 + 少数据性能超过大模型 + 多数据

## 引用

```bibtex
@article{guo2026sdanet,
  title={Towards Better Distortion Feature Learning for Object Detection
         in Top-View Fisheye Cameras},
  author={Guo, Pengbo and Liu, Chengxu and Hou, Xingsong and Qian, Xueming},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  volume={28},
  pages={2106--2118}
}
```
