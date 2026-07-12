#!/usr/bin/env python3
"""
Test all trained models on val / test / val+test splits.
Computes COCO-style metrics and F-score.

Usage:
    python tools/test_all.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BACKEND, DEVICE, INPUT_SIZE, CONF_THRESH, NMS_IOU_THRESH, IOU_THRESH, RANDOM_SEED,
    IMAGENET_MEAN, IMAGENET_STD,
)

if BACKEND == "pytorch":
    import torch
else:
    import jittor as jt
    jt.flags.lazy_execution = 1
    jt.flags.use_threading = 1

from util import (
    decode_predictions, build_model, oriented_nms,
    compute_map, compute_map_coco,
)
from util.anchor_cluster import load_or_cluster_anchors

# 模型和数据集配置
MODELS = {
    "HA": {
        "checkpoint": "output/habbof.pth",
        "train_datasets": ["habbof[train]"],
        "test_datasets": ["habbof[test]"],
        "val_datasets": ["habbof[val]"],
    },
    "CE": {
        "checkpoint": "output/cepdof.pth",
        "train_datasets": ["cepdof[train]"],
        "test_datasets": ["cepdof[test]"],
        "val_datasets": ["cepdof[val]"],
    },
    "WE": {
        "checkpoint": "output/wepdtof.pth",
        "train_datasets": ["wepdtof[train]"],
        "test_datasets": ["wepdtof[test]"],
        "val_datasets": ["wepdtof[val]"],
    },
    "CE+WE+HA": {
        "checkpoint": "output/cepdof-wepdtof-habbof.pth",
        "train_datasets": ["cepdof[train]", "wepdtof[train]", "habbof[train]"],
        # 混合数据集训练后在各数据集分别测试 (test+val 合并)
        "test_datasets_ce": ["cepdof[test]", "cepdof[val]"],
        "test_datasets_we": ["wepdtof[test]", "wepdtof[val]"],
        "test_datasets_ha": ["habbof[test]", "habbof[val]"],
    },
}


def load_checkpoint(checkpoint_path, model):
    """加载训练好的权重

    支持 PyTorch (.pth) 和 Jittor (.pkl) 格式。
    根据文件后缀自动选择加载方式。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ext = os.path.splitext(checkpoint_path)[-1].lower()

    if ext == ".pth" or ext == ".pt":
        # PyTorch 格式
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    else:
        # Jittor 格式 (.pkl)
        try:
            import jittor as jt
        except ImportError:
            raise ImportError(
                f"无法加载 {checkpoint_path}，需要 jittor 后端。"
                f"请切换到 Jittor 后端 (BACKEND='jittor') 或使用 .pth 格式模型。"
            )
        ckpt = jt.load(checkpoint_path)

    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    return model


def evaluate(model, dataset_specs, resolved_anchors, device="cuda"):
    """在指定数据集上评估模型"""
    from datasets import get_dataset

    test_datasets = []
    for spec in dataset_specs:
        try:
            ds = get_dataset(spec)
            test_datasets.append(ds)
        except FileNotFoundError:
            continue

    if not test_datasets:
        raise RuntimeError(f"无法加载测试数据集: {dataset_specs}")

    if BACKEND == "pytorch":
        from torch.utils.data import ConcatDataset as PyTorchConcatDataset
        test_ds = PyTorchConcatDataset(test_datasets)
    else:
        from jittor.dataset import Dataset as JittorDatasetBase

        class ConcatDataset(JittorDatasetBase):
            def __init__(self, datasets):
                super().__init__()
                self.datasets = datasets
                self._lengths = [len(d) for d in datasets]
                self._cumsum = [0]
                for l in self._lengths:
                    self._cumsum.append(self._cumsum[-1] + l)
                self.total = self._cumsum[-1]
                self.set_attrs(total_len=self.total, keep_numpy_array=True)

            def __len__(self):
                return self.total

            def __getitem__(self, idx):
                for i, (cs, cl) in enumerate(zip(self._cumsum[:-1], self._cumsum[1:])):
                    if cs <= idx < cl:
                        return self.datasets[i][idx - cs]
                return self.datasets[-1][idx - self._cumsum[-2]]

        test_ds = ConcatDataset(test_datasets)

    ds_length = getattr(test_ds, 'total', None) or len(test_ds)

    # 获取类别信息
    if hasattr(test_ds, 'class_names'):
        class_names = test_ds.class_names
    elif hasattr(test_ds, 'datasets'):
        for ds in test_ds.datasets:
            if hasattr(ds, 'class_names'):
                class_names = ds.class_names
                break
        else:
            class_names = ["person"]
    else:
        class_names = ["person"]

    model.eval()

    # Jittor 后端：先预热模型
    if BACKEND == "jittor":
        warmup_iters = 50
        print(f"    Jittor 预热中 ({warmup_iters} 次推理)...")
        for i_warmup in range(warmup_iters):
            entry = test_ds[i_warmup % ds_length]
            img_np = entry[0].numpy()
            input_tensor = jt.array(img_np).unsqueeze(0)
            if device == "cuda":
                input_tensor = input_tensor.cuda()
            _ = model(input_tensor)
            if (i_warmup + 1) % 10 == 0:
                print(f"    预热进度: {i_warmup + 1}/{warmup_iters}")
        jt.sync_all()
        jt.gc()
        print(f"    预热完成，开始评估...")

    all_predictions = []
    all_targets = []

    start_time = time.time()
    with (jt.no_grad() if BACKEND == "jittor" else torch.no_grad()):
        for i in range(ds_length):
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

            input_tensor = jt.array(img_np).unsqueeze(0) if BACKEND == "jittor" else torch.from_numpy(img_np).unsqueeze(0)
            if device == "cuda":
                input_tensor = input_tensor.cuda()

            preds = model(input_tensor)
            dets = decode_predictions(preds, conf_thresh=CONF_THRESH, anchors=resolved_anchors)
            pred_boxes, pred_scores, pred_labels = dets[0]

            if len(pred_boxes) > 0:
                keep = oriented_nms(pred_boxes, pred_scores, iou_thresh=NMS_IOU_THRESH)
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

            all_predictions.append((pred_boxes, pred_scores, pred_labels))

            if boxes is None or len(boxes) == 0:
                boxes = np.empty((0, 5), dtype=np.float32)
            if labels is None or len(labels) == 0:
                labels = np.empty((0,), dtype=np.int64)
            all_targets.append((boxes, labels))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                print(f"    [{i+1}/{ds_length}] {speed:.1f} img/s")

    # 计算指标
    results_coco = compute_map_coco(all_predictions, all_targets, class_names)
    results_50 = compute_map(all_predictions, all_targets, class_names, iou_thresh=IOU_THRESH)
    fps = ds_length / (time.time() - start_time) if ds_length > 0 else 0

    # 计算 F-score
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for (pred_boxes, pred_scores, pred_labels), (gt_boxes, gt_labels) in zip(all_predictions, all_targets):
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
        elif len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
        else:
            # 简单的匹配（只用于 F-score）
            matched_gt = set()
            for p_idx in range(len(pred_boxes)):
                matched = False
                for g_idx in range(len(gt_boxes)):
                    if g_idx not in matched_gt:
                        from util.metrics import rotated_iou_single
                        iou = rotated_iou_single(pred_boxes[p_idx], gt_boxes[g_idx])
                        if iou >= IOU_THRESH:
                            matched = True
                            matched_gt.add(g_idx)
                            total_tp += 1
                            break
                if not matched:
                    total_fp += 1
            total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "coco": results_coco,
        "ap50": results_50.get("mAP", 0),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "fps": fps,
        "num_samples": ds_length,
    }


def main():
    print("=" * 70)
    print("SDANet All Models Evaluation")
    print("=" * 70)
    print(f"Backend: {BACKEND}, Device: {DEVICE}")
    print()

    all_results = {}

    for model_name, model_config in MODELS.items():
        print("-" * 70)
        print(f"Model: {model_name} ({os.path.basename(model_config['checkpoint'])})")
        print("-" * 70)

        if not os.path.exists(model_config["checkpoint"]):
            print(f"  [SKIP] Checkpoint not found: {model_config['checkpoint']}")
            continue

        # 使用对应模型的训练集聚类锚框
        model_anchors = load_or_cluster_anchors(
            model_config["train_datasets"], num_clusters=9, input_size=INPUT_SIZE)

        # 加载模型
        num_classes = 1
        model = build_model(num_classes)
        model = load_checkpoint(model_config["checkpoint"], model)
        # Jittor 的 cuda 移动已在 build_model 中处理
        # PyTorch 需要显式移动
        if DEVICE == "cuda" and BACKEND == "pytorch":
            model = model.cuda()
        model.eval()

        all_results[model_name] = {}

        # 判断是否为混合数据集模型
        is_mixed = "test_datasets_ce" in model_config

        if is_mixed:
            # 混合数据集模型：在各个原始数据集上分别测试
            for subset_name, test_specs in [
                ("CEPDOF", model_config["test_datasets_ce"]),
                ("WEPDTOF", model_config["test_datasets_we"]),
                ("HABBOF", model_config["test_datasets_ha"]),
            ]:
                print(f"\n  [{subset_name} Set]")
                results = evaluate(model, test_specs, model_anchors, DEVICE)
                all_results[model_name][subset_name.lower()] = results
                print(f"    COCO mAP: {results['coco']['mAP']*100:.2f}%")
                print(f"    AP@50:    {results['coco']['AP50']*100:.2f}%")
                print(f"    AP@75:    {results['coco']['AP75']*100:.2f}%")
                print(f"    AP_small: {results['coco']['AP_small']*100:.2f}%")
                print(f"    AP_medium:{results['coco']['AP_medium']*100:.2f}%")
                print(f"    AP_large: {results['coco']['AP_large']*100:.2f}%")
                print(f"    F-score:  {results['f_score']*100:.2f}% (P: {results['precision']*100:.2f}%, R: {results['recall']*100:.2f}%)")
                print(f"    FPS:      {results['fps']:.1f} img/s")
        else:
            # 单数据集模型：测试 test / val / val+test
            print(f"\n  [Test Set]")
            test_results = evaluate(model, model_config["test_datasets"], model_anchors, DEVICE)
            all_results[model_name]["test"] = test_results
            print(f"    COCO mAP: {test_results['coco']['mAP']*100:.2f}%")
            print(f"    AP@50:    {test_results['coco']['AP50']*100:.2f}%")
            print(f"    AP@75:    {test_results['coco']['AP75']*100:.2f}%")
            print(f"    AP_small: {test_results['coco']['AP_small']*100:.2f}%")
            print(f"    AP_medium:{test_results['coco']['AP_medium']*100:.2f}%")
            print(f"    AP_large: {test_results['coco']['AP_large']*100:.2f}%")
            print(f"    F-score:  {test_results['f_score']*100:.2f}% (P: {test_results['precision']*100:.2f}%, R: {test_results['recall']*100:.2f}%)")
            print(f"    FPS:      {test_results['fps']:.1f} img/s")

            print(f"\n  [Val Set]")
            val_results = evaluate(model, model_config["val_datasets"], model_anchors, DEVICE)
            all_results[model_name]["val"] = val_results
            print(f"    COCO mAP: {val_results['coco']['mAP']*100:.2f}%")
            print(f"    AP@50:    {val_results['coco']['AP50']*100:.2f}%")
            print(f"    AP@75:    {val_results['coco']['AP75']*100:.2f}%")
            print(f"    AP_small: {val_results['coco']['AP_small']*100:.2f}%")
            print(f"    AP_medium:{val_results['coco']['AP_medium']*100:.2f}%")
            print(f"    AP_large: {val_results['coco']['AP_large']*100:.2f}%")
            print(f"    F-score:  {val_results['f_score']*100:.2f}% (P: {val_results['precision']*100:.2f}%, R: {val_results['recall']*100:.2f}%)")
            print(f"    FPS:      {val_results['fps']:.1f} img/s")

            print(f"\n  [Val + Test Set]")
            val_test_results = evaluate(model, model_config["val_datasets"] + model_config["test_datasets"], model_anchors, DEVICE)
            all_results[model_name]["val_test"] = val_test_results
            print(f"    COCO mAP: {val_test_results['coco']['mAP']*100:.2f}%")
            print(f"    AP@50:    {val_test_results['coco']['AP50']*100:.2f}%")
            print(f"    AP@75:    {val_test_results['coco']['AP75']*100:.2f}%")
            print(f"    AP_small: {val_test_results['coco']['AP_small']*100:.2f}%")
            print(f"    AP_medium:{val_test_results['coco']['AP_medium']*100:.2f}%")
            print(f"    AP_large: {val_test_results['coco']['AP_large']*100:.2f}%")
            print(f"    F-score:  {val_test_results['f_score']*100:.2f}% (P: {val_test_results['precision']*100:.2f}%, R: {val_test_results['recall']*100:.2f}%)")
            print(f"    FPS:      {val_test_results['fps']:.1f} img/s")

        print()

    # 汇总表格
    print("=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Model':<12} {'Split':<12} {'mAP':<10} {'AP@50':<10} {'AP@75':<10} {'F-score':<10} {'FPS':<8}")
    print("-" * 70)
    for model_name in ["HA", "CE", "WE", "CE+WE+HA"]:
        if model_name not in all_results:
            continue
        for split_name, r in all_results[model_name].items():
            print(f"{model_name:<12} {split_name:<12} {r['coco']['mAP']*100:>8.2f}% {r['coco']['AP50']*100:>8.2f}% {r['coco']['AP75']*100:>8.2f}% {r['f_score']*100:>8.2f}% {r['fps']:>6.1f}")
        print()


if __name__ == "__main__":
    main()