import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BACKEND, DEVICE, TEST_DATASETS, OUTPUT_DIR, INPUT_SIZE,
    CONF_THRESH, NMS_IOU_THRESH, IOU_THRESH,
)

# 测试相关配置
TEST_CHECKPOINT = None  # 权重文件路径，如 "output/sdanet_epoch050.pth"
VIS_DIR = "output/vis_results"  # 可视化结果保存目录
NUM_VIS = 16  # 可视化图片数量（设为0则不可视化）

if BACKEND == "pytorch":
    import torch
else:
    import jittor as jt

from util import (
    decode_predictions, build_datasets, build_model, oriented_nms,
    compute_map, compute_map_coco,
)
from util.anchor_cluster import load_or_cluster_anchors


def find_latest_checkpoint():
    """自动查找最新的训练权重

    优先级：
        1. ``latest.pth`` — train.py 每次保存时同步更新的快捷入口。
        2. 扫描 ``sdanet_iter*.pth``，取 iter 号最大的。
    """
    if not os.path.exists(OUTPUT_DIR):
        return None

    # 优先：latest.pth
    latest_path = os.path.join(OUTPUT_DIR, "latest.pth")
    if os.path.isfile(latest_path):
        return latest_path

    # 回退：扫描 iter 最大的 sdanet_iterXXXXXX.pth
    checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("sdanet_iter") and f.endswith('.pth')]
    if not checkpoints:
        return None
    best_iter = -1
    best_f = None
    for f in checkpoints:
        try:
            it = int(f.replace("sdanet_iter", "").replace(".pth", ""))
        except ValueError:
            continue
        if it > best_iter:
            best_iter = it
            best_f = f
    if best_f:
        return os.path.join(OUTPUT_DIR, best_f)
    return None


def load_checkpoint(checkpoint_path, model):
    """加载训练好的权重"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    if BACKEND == "pytorch":
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    else:
        ckpt = jt.load(checkpoint_path)

    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    print(f"成功加载权重: {checkpoint_path}")

    epoch = ckpt.get('epoch', 'unknown')
    print(f"训练轮次: {epoch}")
    return model


def visualize_detection(image, boxes, scores, labels, gt_boxes=None,
                        class_names=None, conf_thresh=0.3):
    """可视化检测结果"""
    vis_img = image.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    ]

    # 真值框 (绿色)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            cx, cy, w, h, R = box
            rect = ((float(cx), float(cy)), (float(w), float(h)), float(R))
            pts = cv2.boxPoints(rect)
            cv2.drawContours(vis_img, [np.intp(pts)], 0, (0, 255, 0), 2)

    # 预测框
    for box, score, label in zip(boxes, scores, labels):
        if score < conf_thresh:
            continue
        cx, cy, w, h, R = box
        color = colors[int(label) % len(colors)]
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(R))
        pts = cv2.boxPoints(rect)
        thickness = max(1, int(score * 4))
        cv2.drawContours(vis_img, [np.intp(pts)], 0, color, thickness)

        class_name = class_names[int(label)] if class_names else f"cls{int(label)}"
        text = f"{class_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img,
                     (int(cx) - text_w//2, int(cy) - text_h//2 - 15),
                     (int(cx) + text_w//2, int(cy) - text_h//2),
                     color, -1)
        cv2.putText(vis_img, text,
                   (int(cx) - text_w//2, int(cy) - text_h//2 - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_img


def main():
    print("=" * 60)
    print("SDANet 测试评估")
    print("=" * 60)
    print(f"Backend: {BACKEND}")
    print(f"数据集: {TEST_DATASETS}")
    print(f"置信度阈值: {CONF_THRESH}")
    print(f"NMS IoU阈值: {NMS_IOU_THRESH}")
    print(f"评估IoU阈值: {IOU_THRESH}")
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
    if NUM_VIS > 0:
        os.makedirs(VIS_DIR, exist_ok=True)

    # 构建数据集
    _, test_ds, _ = build_datasets([], TEST_DATASETS)

    if test_ds is None:
        raise RuntimeError(f"无法加载测试数据集: {TEST_DATASETS}")

    print(f"\n测试数据集: {len(test_ds)} 样本")

    # 获取类别信息
    if hasattr(test_ds, 'class_names'):
        class_names = test_ds.class_names
        num_classes = len(class_names)
    elif hasattr(test_ds, 'datasets'):
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

    # 加载聚类锚框（与训练时一致）
    resolved_anchors = load_or_cluster_anchors(
        TEST_DATASETS, num_clusters=9, input_size=INPUT_SIZE)

    # 移动到设备
    if DEVICE == "cuda":
        if BACKEND == "pytorch":
            model = model.cuda()

    # 全测试集推理
    print(f"\n开始全测试集推理...")

    all_predictions = []
    all_targets = []
    stats = {
        'total_detections': 0,
        'images_with_detections': 0,
    }

    start_time = time.time()

    for i in range(len(test_ds)):
        # 获取数据
        entry = test_ds[i]
        if BACKEND == "pytorch":
            img_tensor, boxes_tensor, labels_tensor = entry
            img_np = img_tensor.cpu().numpy()
            boxes = boxes_tensor.cpu().numpy()
            labels = labels_tensor.cpu().numpy()
        else:
            img_np = entry[0].numpy()
            boxes = entry[1].numpy() if hasattr(entry[1], 'numpy') else entry[1]
            labels = entry[2].numpy() if hasattr(entry[2], 'numpy') else entry[2]

        # 推理
        if BACKEND == "pytorch":
            with torch.no_grad():
                input_tensor = torch.from_numpy(img_np).unsqueeze(0)
                if DEVICE == "cuda":
                    input_tensor = input_tensor.cuda()
                preds = model(input_tensor)
        else:
            input_tensor = jt.array(img_np).unsqueeze(0)
            preds = model(input_tensor)

        # 解码预测（使用聚类锚框，与训练时一致）
        dets = decode_predictions(preds, conf_thresh=CONF_THRESH, anchors=resolved_anchors)
        pred_boxes, pred_scores, pred_labels = dets[0]

        # NMS
        if len(pred_boxes) > 0:
            keep = oriented_nms(pred_boxes, pred_scores, iou_thresh=NMS_IOU_THRESH)
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        all_predictions.append((pred_boxes, pred_scores, pred_labels))

        # GT
        if boxes is None or len(boxes) == 0:
            boxes = np.empty((0, 5), dtype=np.float32)
        if labels is None or len(labels) == 0:
            labels = np.empty((0,), dtype=np.int64)
        all_targets.append((boxes, labels))

        # 统计
        stats['total_detections'] += len(pred_boxes)
        if len(pred_boxes) > 0:
            stats['images_with_detections'] += 1

        # 进度显示
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:4d}/{len(test_ds)}] 速度: {speed:.1f} img/s")

        # 可视化前 NUM_VIS 张
        if NUM_VIS > 0 and i < NUM_VIS:
            img = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
            vis_img = visualize_detection(
                img, pred_boxes, pred_scores, pred_labels,
                gt_boxes=boxes,
                class_names=class_names,
                conf_thresh=CONF_THRESH,
            )
            output_path = os.path.join(VIS_DIR, f"detection_{i:04d}.png")
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    inference_time = time.time() - start_time
    speed = len(test_ds) / inference_time

    print(f"\n推理完成! 耗时: {inference_time:.2f}s, 速度: {speed:.1f} img/s")

    # 计算 mAP
    print("\n" + "=" * 60)
    print("计算 mAP...")
    print("=" * 60)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # AP@50 (标准)
    results_50 = compute_map(all_predictions, all_targets, class_names, iou_thresh=IOU_THRESH)

    # COCO 风格 mAP
    results_coco = compute_map_coco(all_predictions, all_targets, class_names)

    # 打印结果
    print("\n" + "=" * 60)
    print("测试评估结果")
    print("=" * 60)

    print(f"\n【COCO 风格评估】")
    print(f"  AP@50:     {results_coco['AP50']*100:.2f}%")
    print(f"  AP@75:     {results_coco['AP75']*100:.2f}%")
    print(f"  mAP:       {results_coco['mAP']*100:.2f}%")
    print(f"  AP_small:  {results_coco['AP_small']*100:.2f}%")
    print(f"  AP_medium: {results_coco['AP_medium']*100:.2f}%")
    print(f"  AP_large:  {results_coco['AP_large']*100:.2f}%")

    print(f"\n【AP@IoU={IOU_THRESH:.2f}】")
    for cls_name, ap in results_50.items():
        if cls_name not in ['mAP', 'GT_count', 'Det_count']:
            if not np.isnan(ap):
                print(f"  {cls_name}: {ap*100:.2f}%")
    print(f"  mAP:       {results_50['mAP']*100:.2f}%")

    print(f"\n【统计信息】")
    print(f"  GT框总数:      {results_50['GT_count']}")
    print(f"  检测框总数:    {results_50['Det_count']}")
    print(f"  有检测的图像:  {stats['images_with_detections']} "
          f"({100*stats['images_with_detections']/max(1,len(test_ds)):.1f}%)")
    print(f"  平均每图检测:  {stats['total_detections']/max(1,len(test_ds)):.2f}")
    print(f"  推理速度:      {speed:.1f} img/s")

    if NUM_VIS > 0:
        print(f"\n【可视化结果】")
        print(f"  保存目录: {VIS_DIR}/")
        print(f"  可视化数量: {NUM_VIS}")

    print("=" * 60)
    print("✅ 测试完成!")


if __name__ == "__main__":
    main()