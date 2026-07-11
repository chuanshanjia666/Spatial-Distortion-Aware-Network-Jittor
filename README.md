# SDANet (Jittor/PyTorch)

**Spatial Distortion-Aware Network** — 畸变感知适应网络

IEEE TMM 2026 论文 _"Towards Better Distortion Feature Learning for Object Detection in Top-View Fisheye Cameras"_ 复现，支持 **PyTorch** 与 **Jittor** 双后端。

## 论文摘要

SDANet 提出了一种动态卷积 + 空间分离策略的畸变感知适应网络，利用鱼眼镜头的畸变特性来提升目标检测性能。针对顶视鱼眼相机的畸变特性，SDANet 设计了 SDAConv 模块，该模块能够自适应地学习不同空间位置的卷积核组合，从而有效地捕捉畸变图像中的特征信息。通过在 FPN 颈部引入 SDAConv 横向连接，SDANet 能够在多尺度上进行有向框检测，显著提升了在畸变图像上的检测性能。此外，作者还提出一种在线畸变增强方法 PFDAug，用于在训练过程中模拟鱼眼的严重畸变，从而进一步提高模型的泛化能力。实验结果表明，SDANet 在多个鱼眼数据集上均取得了优异的性能，验证了其在顶视鱼眼相机目标检测任务中的有效性。

![sdablk](img/sdablk.png)

![sss](img/sss.png)

![pfgaug](img/pfgaug.png)

## 📐 网络结构

网络基于YOLOv3改进
![网络结构](img/网络结构.png)

## 快速开始

### 1. 环境准备

pytorch后端使用任意安装方式即可，可以使用系统包管理器，pip，conda等方式安装。
可以使用提供的conda环境文件创建环境：

```bash
conda env create -f environment.yml
conda activate sdanet-pytorch
```

jittor的兼容性一般，推荐使用docker镜像，来保证最佳兼容性，jittor官方镜像最后一次更新在五年前，所以我们提供了一个基于Nvidia镜像的自定义镜像，包含了jittor和其他依赖。

```bash
docker build -t sdanet-jittor -f Dockerfile .
```

### 2. 下载预训练骨干网络

```bash
python pretrain/download_pretrained.py        # 下载并提取 DarkNet53 backbone
python pretrain/convert_to_jittor.py          # 如使用 Jittor 后端，转换权重格式
```

### 3. 准备数据集

使用HABBOF CEPDOF WEPDTOF 数据集，你可以从这里下载 https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/

fisheye8k数据集可以huggingface直接下载

数据集需要转化为统一的coco格式，使用提供的脚本进行转换和拆分：

```bash
# HABBOF: .txt → COCO JSON
python tools/convert_habbof.py --root datasets/HABBOF

# FishEye8k: samples.json → COCO JSON
python tools/convert_fisheye8k.py --root datasets/FishEye8k

# 拆分 train/val/test
python tools/split_train_val.py --ann datasets/HABBOF/annotations/all.json

# CEPDOF / WEPDTOF: 合并多场景 JSON
python tools/merge_cepdof.py
python tools/merge_wepdtof.py

```

### 4. 训练

所有超参数均在 `config.py` 中配置，无需命令行参数：

```bash
python tools/train.py
```

关键配置：

```python
BACKEND = "pytorch"
TRAIN_DATASETS = ["habbof[train]"]
VAL_DATASETS   = ["habbof[val]"]
TEST_DATASETS  = ["habbof[test]"]

BATCH_SIZE = 64
INPUT_SIZE = 416
USE_ACCUMULATION_STEP = True
EPOCHS = 50
WARMUP_ITERS = 1000
LR = 0.0001
```

### 5. 测试与评估

```bash
python tools/test.py
```

自动加载 `output/` 目录下最新的权重，在全测试集上推理并输出：

- **COCO 风格指标**：AP@50 / AP@75 / mAP@[.50:.95] / AP_small / AP_medium / AP_large
- **VOC 风格指标**：AP@IoU=0.50, 每类 AP, mAP
- **可视化结果**：保存到 `output/vis_results/`（前 NUM_VIS 张）

## 支持的数据集

| 数据集        | 类别                          | 场景          | 说明                 |
| ------------- | ----------------------------- | ------------- | -------------------- |
| **HABBOF**    | person                        | 实验室/会议室 | 室内顶视鱼眼，仅人物 |
| **CEPDOF**    | person                        | 多活动场景    | 各类室内活动，仅人物 |
| **WEPDTOF**   | person                        | 商店/办公室   | 多场景监控，仅人物   |
| **FishEye8k** | Bike/Bus/Car/Pedestrian/Truck | 室外街景      | 多类别鱼眼检测       |

所有数据集统一转换为带旋转框 `bbox: [cx, cy, w, h, R]` 的 COCO 格式。

## 📝 引用

```bibtex
@article{sdanet2026,
  title={Towards Better Distortion Feature Learning for Object Detection in Top-View Fisheye Cameras},
  journal={IEEE Transactions on Multimedia (TMM)},
  year={2026}
}
```

## 📄 License

见 [LICENSE](LICENSE)。
