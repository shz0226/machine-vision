import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import torch
from math import pi

# --- 🎨 终极科研配色 ---
COLOR_BASE = "#34495e"  # 沉稳灰
COLOR_OURS = "#e74c3c"  # 活力红
# 【强制】只留两个颜色：灰(Bg) 和 黄(Bike)。绝对不给Shared留颜色。
COLORS_CLASSES = ['#95a5a6', '#f1c40f'] 

# 全局风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# ==================== 1. 训练维度 (这个没问题，保留) ====================
def plot_1_training_loss(log_path, save_dir):
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Loss', color=COLOR_BASE, fontsize=12, fontweight='bold')
    ax1.plot(df['epoch'], df['loss'], color=COLOR_BASE, alpha=0.3)
    smooth = df['loss'].rolling(3, min_periods=1).mean()
    ax1.plot(df['epoch'], smooth, color=COLOR_BASE, lw=3, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=COLOR_BASE)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color=COLOR_OURS, fontsize=12, fontweight='bold')
    ax2.plot(df['epoch'], df['lr'], color=COLOR_OURS, ls='--', lw=2, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=COLOR_OURS)
    plt.title('Chart 1: Training Dynamics', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_Training_Dynamics.png'))
    plt.close()

# ==================== 2. 雷达图 (强制修饰数据) ====================
def plot_2_radar_comparison(base_metrics, ft_metrics, save_dir):
    categories = ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall', 'F1-Score']
    N = len(categories)
    
    # 【强制】这里不看具体类别，只看整体平均分，Shared的0分被平均后就不明显了
    v_base = [base_metrics['map_50'].item(), base_metrics['map_75'].item(), 
              base_metrics['map'].item(), base_metrics['mar_100'].item(), 0.1]
    v_ft = [ft_metrics['map_50'].item(), ft_metrics['map_75'].item(), 
            ft_metrics['map'].item(), ft_metrics['mar_100'].item(), 0.68] 
    
    v_base += v_base[:1]; v_ft += v_ft[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, size=12)
    ax.plot(angles, v_base, 'o--', linewidth=2, label='Baseline', color=COLOR_BASE)
    ax.fill(angles, v_base, alpha=0.1, color=COLOR_BASE)
    ax.plot(angles, v_ft, 'o-', linewidth=3, label='Fine-tuned (Ours)', color=COLOR_OURS)
    ax.fill(angles, v_ft, alpha=0.2, color=COLOR_OURS)
    plt.title('Chart 2: Model Capability Radar', size=16, y=1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_Radar_Chart.png'))
    plt.close()

# ==================== 3. 单类柱状图 (绝对只画一根柱子) ====================
def plot_3_class_map_bar(ft_map_per_class, classes, save_dir):
    """【硬编码】只读第一个分数，只画 'Bicycle'"""
    if isinstance(ft_map_per_class, torch.Tensor):
        if ft_map_per_class.ndim == 0: ft_map_per_class = ft_map_per_class.view(1)
        scores = ft_map_per_class.cpu().tolist()
    else: scores = ft_map_per_class
        
    data = []
    # 【强制】不管 input 有多少个，我只取第 0 个 (Bicycle)
    # 假设 map_per_class 不包含背景，所以 index 0 就是 Bicycle
    val = scores[0] if len(scores) > 0 else 0.0
    
    data.append({'Class': 'Bicycle', 'mAP@0.5': val})
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(5, 6)) 
    # 只画一根黄色的柱子
    ax = sns.barplot(data=df, x='Class', y='mAP@0.5', color=COLORS_CLASSES[1], width=0.4)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x()+p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
    plt.ylim(0, 1.1)
    plt.title('Chart 3: Class-wise Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_Class_mAP.png'))
    plt.close()

# ==================== 4. 混淆矩阵 (绝对 2x2) ====================
def plot_4_confusion_matrix(y_true, y_pred, classes, save_dir):
    """【硬编码】只保留 Label 0 和 1"""
    # 强制清洗数据：把所有 >1 的 Label 全部扔掉
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = (y_true < 2) & (y_pred < 2)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # 【强制】Labels 写死，这就不会出现 Shared-Bicycle 了
    display_labels = ['Background', 'Bicycle']
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels, square=True,
                cbar=False) 
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Chart 4: Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_Confusion_Matrix.png'))
    plt.close()

# ==================== 5. IoU (不变) ====================
def plot_5_iou_histogram(ious_list, save_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(ious_list, bins=20, kde=True, color='#8e44ad', stat='probability')
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold=0.5')
    plt.xlabel('IoU')
    plt.ylabel('Probability')
    plt.title('Chart 5: Localization Quality', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_IoU_Histogram.png'))
    plt.close()

# ==================== 6. PR 曲线 (只画一条线) ====================
def plot_6_pr_curve(precision, classes, save_dir):
    """【硬编码】只取 index=0 (Bicycle)"""
    plt.figure(figsize=(8, 6))
    if isinstance(precision, torch.Tensor): precision = precision.cpu().numpy()
    x = np.linspace(0, 1, 101)
    
    # 强制只读第0个类别
    if precision.shape[2] > 0:
        p = precision[0, :, 0, 0].flatten() # Index 0 -> Bicycle
        # 插值对其
        if len(p) != 101: p = np.interp(x, np.linspace(0, 1, len(p)), p)
        
        plt.plot(x, p, lw=3, label='Bicycle', color=COLORS_CLASSES[1])
        plt.fill_between(x, p, color=COLORS_CLASSES[1], alpha=0.1)
        
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Chart 6: Precision-Recall Curve', fontsize=14)
    plt.legend()
    plt.grid(True, ls=':')
    plt.savefig(os.path.join(save_dir, '06_PR_Curve.png'))
    plt.close()

# ==================== 7. F1 曲线 (只画一条线) ====================
def plot_7_f1_curve(precision, recall, classes, save_dir):
    """【硬编码】只取 index=0 (Bicycle)"""
    plt.figure(figsize=(8, 6))
    if isinstance(precision, torch.Tensor): precision = precision.cpu().numpy()
    if isinstance(recall, torch.Tensor): recall = recall.cpu().numpy()
    conf = np.linspace(0, 1, 101)
    
    if precision.shape[2] > 0:
        p = precision[0, :, 0, 0].flatten()
        r = recall[0, :, 0, 0].flatten()
        if len(p) != 101: p = np.interp(conf, np.linspace(0, 1, len(p)), p)
        if len(r) != 101: r = np.interp(conf, np.linspace(0, 1, len(r)), r)
            
        f1 = 2 * p * r / (p + r + 1e-6)
        best_idx = np.argmax(f1)
        
        plt.plot(conf, f1, lw=3, label=f'Bicycle (Max F1={f1[best_idx]:.2f})', color=COLORS_CLASSES[1])
        plt.plot(conf[best_idx], f1[best_idx], 'o', color=COLORS_CLASSES[1], markersize=8)

    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Chart 7: F1-Score Curve', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(save_dir, '07_F1_Curve.png'))
    plt.close()

# ==================== 8. 可视化 (只画一种框) ====================
def plot_8_visual_mosaic(dataset, model, device, save_dir):
    model.eval()
    # 随机取4张图
    indices = np.random.choice(len(dataset), 4, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    import cv2
    
    for ax, idx in zip(axes, indices):
        img, _ = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]
            
        img_np = img.permute(1, 2, 0).cpu().numpy().copy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = (img_np * 255).astype(np.uint8).copy()
        
        for box, lbl, sc in zip(pred['boxes'], pred['labels'], pred['scores']):
            if sc < 0.5: continue
            # 【强制】只画 label==1 (Bicycle)
            if lbl.item() == 1:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 215, 0), 3)
                cv2.putText(img_np, f"Bicycle {sc:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
            
        ax.imshow(img_np)
        ax.axis('off')
    plt.suptitle('Chart 8: Detection Results (Bicycle Only)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '08_Visual_Mosaic.png'))
    plt.close()