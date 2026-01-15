import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from src.dataset import SharedBikeDataset, get_transform
from src.model import get_baseline_model, get_finetune_model

# 引入终极绘图模块
from src.plotting_ultra import (
    plot_1_training_loss,
    plot_2_radar_comparison,
    plot_3_class_map_bar,
    plot_4_confusion_matrix,
    plot_5_iou_histogram,
    plot_6_pr_curve,
    plot_7_f1_curve,
    plot_8_visual_mosaic
)

DEVICE = torch.device('cuda')
DATA_ROOT = './cycledata'
OUTPUT_DIR = './output'
PLOT_DIR = './plots/ultra_analysis' # 终极文件夹
os.makedirs(PLOT_DIR, exist_ok=True)

CLASSES = ['Background', 'Bicycle', 'Shared-Bicycle']

def get_batch_stats(preds, targets, iou_thresh=0.5):
    """提取混淆矩阵和IoU数据"""
    y_true, y_pred, all_ious = [], [], []
    
    for p, t in zip(preds, targets):
        pred_boxes = p['boxes']
        pred_lbls = p['labels']
        pred_sc = p['scores']
        gt_boxes = t['boxes']
        gt_lbls = t['labels']
        
        # 1. 计算 IoU 分布 (只算由于 GT 匹配上的)
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(gt_boxes, pred_boxes)
            # 取每个 GT 对应的最大 IoU
            max_ious, _ = ious.max(dim=1) 
            all_ious.extend(max_ious.cpu().numpy())
            
        # 2. 计算混淆矩阵标签
        if len(gt_boxes) == 0:
            for pl, sc in zip(pred_lbls, pred_sc):
                if sc > 0.5: y_true.append(0); y_pred.append(pl.item())
            continue
            
        if len(pred_boxes) == 0:
            for gl in gt_lbls: y_true.append(gl.item()); y_pred.append(0)
            continue
            
        ious_mat = box_iou(gt_boxes, pred_boxes)
        for i in range(len(gt_boxes)):
            gt_l = gt_lbls[i].item()
            # 找到匹配最好的预测框
            max_iou, max_idx = ious_mat[i].max(dim=0)
            
            if max_iou > iou_thresh:
                # 进一步检查：这个预测框是否也被其他 GT 抢走了？(简化版不查重)
                pred_l = pred_lbls[max_idx].item()
                y_true.append(gt_l); y_pred.append(pred_l)
            else:
                y_true.append(gt_l); y_pred.append(0) # 漏检
                
    return y_true, y_pred, all_ious

def main():
    print("========== 启动全方位评估系统 (Ultra Evaluation) ==========")
    
    # 1. 绘制训练图 (Chart 1)
    print(">>> [1/8] Generating Training Chart...")
    plot_1_training_loss(os.path.join(OUTPUT_DIR, 'training_log.csv'), PLOT_DIR)
    
    # 准备数据与模型
    val_dataset = SharedBikeDataset(DATA_ROOT, split='val', transforms=get_transform())
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    print(">>> Loading Models...")
    # Baseline (MobileNetV3)
    base_model = get_baseline_model().to(DEVICE).eval()
    # Fine-tuned (MobileNetV3)
    ft_model = get_finetune_model(num_classes=3).to(DEVICE).eval()
    if os.path.exists(os.path.join(OUTPUT_DIR, 'best_model.pth')):
        ft_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    else:
        print("⚠️ Warning: best_model.pth not found!")

    # 3. 运行推理统计
    print(">>> Running Inference Loop...")
    metric_ft = MeanAveragePrecision(class_metrics=True, extended_summary=True)
    metric_base = MeanAveragePrecision(class_metrics=True)
    
    all_yt, all_yp, all_ious = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # --- Fine-tuned ---
            ft_out = ft_model(images)
            metric_ft.update(ft_out, targets)
            
            # 收集详细统计数据
            yt, yp, ious = get_batch_stats(ft_out, targets)
            all_yt.extend(yt); all_yp.extend(yp); all_ious.extend(ious)
            
            # --- Baseline ---
            base_out = base_model(images)
            metric_base.update(base_out, targets)

    print(">>> Computing Metrics...")
    ft_res = metric_ft.compute()
    base_res = metric_base.compute()
    
    # 4. 生成剩余图表
    print(">>> Generating Charts [2-8]...")
    
    # 图2: 雷达对比
    plot_2_radar_comparison(base_res, ft_res, PLOT_DIR)
    
    # 图3: 分科成绩 (Ours)
    plot_3_class_map_bar(ft_res['map_per_class'], CLASSES, PLOT_DIR)
    
    # 图4: 混淆矩阵
    plot_4_confusion_matrix(all_yt, all_yp, CLASSES, PLOT_DIR)
    
    # 图5: IoU 分布
    plot_5_iou_histogram(all_ious, PLOT_DIR)
    
    # 图6: PR 曲线
    plot_6_pr_curve(ft_res['precision'], CLASSES, PLOT_DIR)
    
    # 图7: F1 曲线
    plot_7_f1_curve(ft_res['precision'], ft_res['recall'], CLASSES, PLOT_DIR)
    
    # 图8: 可视化马赛克 (需要 dataset)
    plot_8_visual_mosaic(val_dataset, ft_model, DEVICE, PLOT_DIR)
    
    print(f"\n✅ 恭喜！8张全维度分析图表已生成至: {PLOT_DIR}")
    print("1. 训练动态 (Training Dynamics)")
    print("2. 能力雷达 (Radar Chart)")
    print("3. 类别mAP (Class mAP)")
    print("4. 混淆矩阵 (Confusion Matrix)")
    print("5. 定位质量 (IoU Histogram)")
    print("6. PR曲线 (PR Curve)")
    print("7. F1曲线 (F1 Curve)")
    print("8. 实测预览 (Visual Mosaic)")

if __name__ == '__main__':
    main()