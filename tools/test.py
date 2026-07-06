#!/usr/bin/env python3
"""
SDANet 测试脚本 - 加载训练好的权重进行推理并可视化结果。

所有超参数从 config.py 读取。

使用方法:
    python tools/test.py
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BACKEND, DEVICE, TEST_DATASETS, OUTPUT_DIR,
    CONF_THRESH, NMS_IOU_THRESH,
)

# 测试相关配置
TEST_CHECKPOINT = None  # 权重文件路径，如 "output/sdanet_epoch050.pth"
VIS_DIR = "output/vis_results"  # 可视化结果保存目录
NUM_VIS = 16  # 可视化图片数量

if BACKEND == "pytorch":
    import torch
else:
    import jittor as jt

from util import decode_predictions, build_datasets, build_model, oriented_nms


def find_latest_checkpoint():
    """自动查找最新的训练权重"""
    if not os.path.exists(OUTPUT_DIR):
        return None
    checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pth')]
    if not checkpoints:
        return None
    # 按epoch排序
    checkpoints.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]) if 'epoch' in x else 0)
    return os.path.join(OUTPUT_DIR, checkpoints[-1])


def load_checkpoint(checkpoint_path, model):
    """加载训练好的权重"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    if BACKEND == "pytorch":
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    else:
        ckpt = jt.load(checkpoint_path)

    # 尝试加载模型权重
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    print(f"成功加载权重: {checkpoint_path}")

    epoch = ckpt.get('epoch', 'unknown')
    print(f"训练轮次: {epoch}")
    return model


def visualize_detection(image, boxes, scores, labels, gt_boxes=None,
                        class_names=None, conf_thresh=0.3):
    """可视化检测结果

    Args:
        image: (H, W, 3) uint8 RGB图像
        boxes: (N, 5) [cx, cy, w, h, R]
        scores: (N,) 置信度
        labels: (N,) 类别ID
        gt_boxes: (M, 5) 真值框 (可选)
        class_names: 类别名称列表
        conf_thresh: 置信度阈值
    """
    vis_img = image.copy()

    # 颜色映射: 不同类别不同颜色
    colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 紫色
        (0, 255, 255),    # 青色
        (255, 128, 0),    # 橙色
        (128, 0, 255),    # 紫罗兰
    ]

    # 绘制真值框 (绿色)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            cx, cy, w, h, R = box
            rect = ((float(cx), float(cy)), (float(w), float(h)), float(R))
            pts = cv2.boxPoints(rect)
            pts = np.intp(pts)
            cv2.drawContours(vis_img, [pts], 0, (0, 255, 0), 2)  # 绿色实线

    # 绘制预测框 (根据置信度调整颜色深浅)
    for box, score, label in zip(boxes, scores, labels):
        if score < conf_thresh:
            continue

        cx, cy, w, h, R = box
        color = colors[int(label) % len(colors)]

        # 绘制旋转矩形
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(R))
        pts = cv2.boxPoints(rect)
        pts = np.intp(pts)

        # 置信度越高线条越粗
        thickness = max(1, int(score * 4))
        cv2.drawContours(vis_img, [pts], 0, color, thickness)

        # 绘制类别标签和置信度
        class_name = class_names[int(label)] if class_names else f"cls{int(label)}"
        text = f"{class_name}: {score:.2f}"

        # 文字背景
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img,
                     (int(cx) - text_w//2, int(cy) - text_h//2 - 15),
                     (int(cx) + text_w//2, int(cy) - text_h//2),
                     color, -1)

        # 文字
        cv2.putText(vis_img, text,
                   (int(cx) - text_w//2, int(cy) - text_h//2 - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_img


def main():
    print("=" * 60)
    print("SDANet 推理测试")
    print("=" * 60)
    print(f"Backend: {BACKEND}")
    print(f"数据集: {TEST_DATASETS}")
    print(f"置信度阈值: {CONF_THRESH}")
    print(f"NMS IoU阈值: {NMS_IOU_THRESH}")
    print("=" * 60)

    # 自动查找最新权重
    if TEST_CHECKPOINT is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise RuntimeError(f"未找到训练权重文件，请先将权重放入 {OUTPUT_DIR} 目录，"
                             f"或设置 TEST_CHECKPOINT 变量指定权重路径")
    else:
        checkpoint_path = TEST_CHECKPOINT

    print(f"使用权重: {checkpoint_path}")

    # 创建可视化目录
    os.makedirs(VIS_DIR, exist_ok=True)

    # 构建数据集
    _, test_ds, _ = build_datasets([], TEST_DATASETS)

    if test_ds is None:
        raise RuntimeError(f"无法加载测试数据集: {TEST_DATASETS}")

    print(f"\n测试数据集: {len(test_ds)} 样本")

    # 获取类别名称
    if hasattr(test_ds, 'class_names'):
        class_names = test_ds.class_names
        num_classes = len(class_names)
    elif hasattr(test_ds, 'datasets'):
        # ConcatDataset
        for ds in test_ds.datasets:
            if hasattr(ds, 'class_names'):
                class_names = ds.class_names
                num_classes = len(class_names)
                break
        else:
            class_names = None
            num_classes = 1
    else:
        class_names = None
        num_classes = 1

    print(f"类别数: {num_classes}")
    if class_names:
        print(f"类别名称: {class_names}")

    # 构建模型
    model = build_model(num_classes)
    print("\n模型已构建")

    # 加载权重
    model = load_checkpoint(checkpoint_path, model)
    model.eval()

    # 移动到设备
    if DEVICE == "cuda":
        if BACKEND == "pytorch":
            model = model.cuda()

    # 推理和可视化
    print(f"\n开始推理 (最多可视化 {NUM_VIS} 张图)...")

    num_vis = min(NUM_VIS, len(test_ds))
    stats = {
        'total_detections': 0,
        'images_with_detections': 0,
    }

    for i in range(num_vis):
        # 获取原始图像和标注
        entry = test_ds[i]
        if BACKEND == "pytorch":
            img_tensor, boxes_tensor, _ = entry
            img_np = img_tensor.cpu().numpy()
            boxes = boxes_tensor.cpu().numpy()
        else:
            img_np = entry[0].numpy()
            boxes = entry[1].numpy() if hasattr(entry[1], 'numpy') else entry[1]

        # 转换为 (H, W, 3) 图像
        img = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8).copy()

        # 模型推理
        if BACKEND == "pytorch":
            with torch.no_grad():
                input_tensor = torch.from_numpy(img_np).unsqueeze(0)
                if DEVICE == "cuda":
                    input_tensor = input_tensor.cuda()
                preds = model(input_tensor)
        else:
            input_tensor = jt.array(img_np).unsqueeze(0)
            preds = model(input_tensor)

        # 解码预测
        dets = decode_predictions(preds, conf_thresh=CONF_THRESH)
        pred_boxes, pred_scores, pred_labels = dets[0]

        # 应用 NMS
        if len(pred_boxes) > 0:
            keep = oriented_nms(pred_boxes, pred_scores, iou_thresh=NMS_IOU_THRESH)
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        # 统计
        stats['total_detections'] += len(pred_boxes)
        if len(pred_boxes) > 0:
            stats['images_with_detections'] += 1

        # 可视化
        vis_img = visualize_detection(
            img, pred_boxes, pred_scores, pred_labels,
            gt_boxes=None,
            class_names=class_names,
            conf_thresh=CONF_THRESH,
        )

        # 保存结果
        output_path = os.path.join(VIS_DIR, f"detection_{i:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        # 打印统计信息
        gt_count = len(boxes) if boxes is not None else 0
        pred_count = len(pred_boxes)
        print(f"  [{i+1:3d}/{num_vis}] GT:{gt_count:2d} | Pred:{pred_count:2d} | "
              f"输出: {output_path}")

    # 打印汇总统计
    print("\n" + "=" * 60)
    print("推理统计汇总")
    print("=" * 60)
    print(f"处理图像数: {num_vis}")
    print(f"有检测的图像数: {stats['images_with_detections']} ({100*stats['images_with_detections']/max(1,num_vis):.1f}%)")
    print(f"总检测数: {stats['total_detections']}")
    print(f"平均每图检测数: {stats['total_detections']/max(1,num_vis):.2f}")
    print(f"结果保存在: {VIS_DIR}/")
    print("=" * 60)

    print("\n✅ 推理完成!")


if __name__ == "__main__":
    main()