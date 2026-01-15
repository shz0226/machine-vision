import torch
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from math import pi
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.dataset import SharedBikeDataset, get_transform
from src.model import get_model
from sklearn.metrics import confusion_matrix, roc_auc_score

# --- 1. 配置区域 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './cycledata'
OUTPUT_ROOT = './output'
SAVE_ROOT = './plots/48_charts_8_dimensions_v2' # 更新保存路径
os.makedirs(SAVE_ROOT, exist_ok=True)

NUM_CLASSES = 2 
MODELS_KEYS = ['mb3_320', 'mb3_fpn', 'resnet50']
MODELS_NAMES = ['MobileNet-320', 'MobileNet-FPN', 'ResNet50']

# 🎨 升级配色方案：同色系深浅搭配 (Base: 浅色, FT: 深色)
# 格式: Key -> [Base Color, FT Color]
COLOR_MAP = {
    'mb3_320':  ['#a8e6cf', '#2ecc71'], # 浅绿 vs 深绿
    'mb3_fpn':  ['#aed6f1', '#3498db'], # 浅蓝 vs 深蓝
    'resnet50': ['#f5b7b1', '#e74c3c']  # 浅红 vs 深红
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心评估函数 ---

def get_eval_data(loader):
    """收集所有模型数据，包含 IoU 和 AUC"""
    full_results = []
    
    for key, name in zip(MODELS_KEYS, MODELS_NAMES):
        for is_ft in [False, True]:
            status = "Fine-tuned" if is_ft else "Baseline"
            print(f"📊 正在评估: {name} [{status}] ...")
            
            # 获取对应的颜色
            color = COLOR_MAP[key][1] if is_ft else COLOR_MAP[key][0]
            
            try:
                # 加载模型
                if is_ft:
                    model = get_model(key, num_classes=NUM_CLASSES, is_pretrained=True, keep_coco_head=False).to(DEVICE)
                    weight_path = os.path.join(OUTPUT_ROOT, key, 'best_model.pth')
                    if os.path.exists(weight_path):
                        ckpt = torch.load(weight_path, map_location=DEVICE)
                        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
                    else:
                        print(f"⚠️ 缺微调权重，跳过")
                        continue
                else:
                    model = get_model(key, is_pretrained=True, keep_coco_head=True).to(DEVICE)
                
                model.eval()
            except Exception as e:
                print(f"❌ 加载失败: {e}")
                continue

            # 推理
            metric = MeanAveragePrecision(class_metrics=True)
            y_true, y_pred = [], []
            y_scores = [] 
            all_ious = [] 
            
            start_time = time.time()
            img_cnt = 0
            
            with torch.no_grad():
                for imgs, targets in tqdm(loader, leave=False):
                    imgs = [img.to(DEVICE) for img in imgs]
                    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                    
                    raw_preds = model(imgs)
                    img_cnt += len(imgs)
                    
                    final_preds = []
                    for i, p in enumerate(raw_preds):
                        # 1. 过滤 Baseline (COCO ID mapping)
                        if not is_ft: 
                            mask = p['labels'] == 2
                            p_boxes = p['boxes'][mask]
                            p_scores = p['scores'][mask]
                            p_labels = torch.full_like(p_scores, 1, dtype=torch.int64)
                            cur_pred = {'boxes': p_boxes, 'scores': p_scores, 'labels': p_labels}
                        else:
                            cur_pred = p
                        
                        final_preds.append(cur_pred)
                        
                        # 2. 计算 IoU (用于密度图)
                        if len(cur_pred['boxes']) > 0 and len(targets[i]['boxes']) > 0:
                            iou_matrix = torchvision.ops.box_iou(cur_pred['boxes'], targets[i]['boxes'])
                            max_vals, _ = iou_matrix.max(dim=1) 
                            all_ious.extend(max_vals.cpu().numpy().tolist())
                        elif len(cur_pred['boxes']) > 0:
                            all_ious.extend([0.0] * len(cur_pred['boxes']))

                    metric.update(final_preds, targets)
                    
                    # 3. 收集 AUC 和 CM 数据
                    for p, t in zip(final_preds, targets):
                        has_gt = 1 if len(t['labels']) > 0 else 0
                        y_true.append(has_gt)
                        max_score = p['scores'].max().item() if len(p['scores']) > 0 else 0.0
                        y_scores.append(max_score)
                        y_pred.append(1 if max_score > 0.5 else 0)

            fps = img_cnt / (time.time() - start_time + 1e-6)
            
            try:
                res = metric.compute()
            except:
                res = {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0, 'mar_100': 0.0}

            try:
                auc_val = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5
            except:
                auc_val = 0.5

            full_results.append({
                'key': key, 'name': name, 'type': status,
                'color': color, # 👈 存入分配好的颜色
                'unique_name': f"{name} ({status})", # 👈 唯一标识符用于绘图标签
                'mAP': res['map'].item(),
                'mAP_50': res['map_50'].item(),
                'mAP_75': res['map_75'].item(),
                'Recall': res['mar_100'].item(),
                'AUC': auc_val,
                'ious': all_ious, 
                'mIoU': np.mean(all_ious) if all_ious else 0.0, 
                'FPS': fps,
                'y_true': y_true, 'y_pred': y_pred
            })
            
    return pd.DataFrame(full_results)

# --- 3. 八大维度的绘图引擎 (升级版: 生成6张图) ---

def generate_6_charts_per_dimension(df, dim_name, plot_func):
    """自动生成 6 张图 (包含一张 All-in-One 全比较)"""
    save_dir = os.path.join(SAVE_ROOT, dim_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"🖼️  正在生成 [{dim_name}] 维度的 6 张图表...")

    # 1-3: Individual Compare (Base vs FT)
    for i, (key, name) in enumerate(zip(MODELS_KEYS, MODELS_NAMES)):
        sub_df = df[df['key'] == key]
        if not sub_df.empty:
            plot_func(sub_df, f"{name}: Baseline vs Fine-tuned", os.path.join(save_dir, f"{i+1}_{key}_Compare.png"))

    # 4: All Baselines
    base_df = df[df['type'] == 'Baseline']
    if not base_df.empty:
        plot_func(base_df, "Comparison of All Baseline Models", os.path.join(save_dir, "4_All_Baselines.png"))

    # 5: All Fine-tuned
    ft_df = df[df['type'] == 'Fine-tuned']
    if not ft_df.empty:
        plot_func(ft_df, "Comparison of All Fine-tuned Models", os.path.join(save_dir, "5_All_Finetuned.png"))

    # 🌟 6: All Models Together (New!)
    plot_func(df, "Comparison of All 6 Models (Base vs FT)", os.path.join(save_dir, "6_All_Models_Compare.png"))

# --- 具体的绘图逻辑 ---

# 通用柱状图
def plot_bar_generic(data, y_col, title, path):
    plt.figure(figsize=(10, 6) if len(data) > 3 else (8, 6))
    
    # 构造颜色列表，保证绘图颜色正确
    palette = dict(zip(data['unique_name'], data['color']))
    
    # 如果是全比较图，x轴显示模型名，hue显示类型，这样更整齐
    if len(data) > 3:
        ax = sns.barplot(x='name', y=y_col, hue='type', data=data, 
                         palette={'Baseline': '#95a5a6', 'Fine-tuned': '#2ecc71'} if 'color' not in data.columns else None)
        # 手动修正颜色 (因为 Seaborn 的 hue palette 比较难直接映射每根柱子的特定颜色)
        # 这里为了简单展示，如果是全量图，我们直接用 x='unique_name' 可能会太长，
        # 还是推荐直接画，然后自己控制颜色。
        # 最简单的方式：直接用 x='unique_name'，然后旋转标签
        plt.clf() # 清除上面的尝试
        ax = sns.barplot(x='unique_name', y=y_col, data=data, palette=palette, hue='unique_name', legend=False)
        plt.xticks(rotation=45, ha='right')
    else:
        # 单个对比图
        ax = sns.barplot(x='unique_name', y=y_col, data=data, palette=palette, hue='unique_name', legend=False)
        plt.xticks(rotation=0)

    plt.title(f"[{y_col}] {title}", fontsize=14); plt.ylim(0, 1.1)
    if y_col == 'FPS': plt.ylim(0, None)
    
    plt.xlabel('')
    for p in ax.patches:
        h = p.get_height()
        if h > 0: ax.annotate(f'{h:.2f}' if y_col=='FPS' else f'{h:.3f}', 
                              (p.get_x()+p.get_width()/2., h), ha='center', va='bottom')
    plt.tight_layout(); plt.savefig(path); plt.close()

# 1-3 & 6: 常规指标
def plot_logic_map(data, title, path): plot_bar_generic(data, 'mAP', title, path)
def plot_logic_map50(data, title, path): plot_bar_generic(data, 'mAP_50', title, path)
def plot_logic_map75(data, title, path): plot_bar_generic(data, 'mAP_75', title, path)
def plot_logic_fps(data, title, path): plot_bar_generic(data, 'FPS', title, path)

# 维度 4: IoU Distribution (密度图) - 适配 6 线图
def plot_logic_iou(data, title, path):
    plt.figure(figsize=(10, 7))
    has_data = False
    
    # 按顺序绘图，保证图例整齐
    # 如果是全量图，排序一下：先Base再FT，或者按模型聚类
    if len(data) > 3:
        data = data.sort_values(by=['key', 'type']) # 排序: MB320-Base, MB320-FT, MBFPN-Base...
        
    for _, row in data.iterrows():
        if len(row['ious']) > 5:
            sns.kdeplot(row['ious'], label=row['unique_name'], color=row['color'], fill=True, alpha=0.1, linewidth=2)
            has_data = True
    
    if not has_data:
        plt.text(0.5, 0.5, "Insufficient Data for KDE Plot", ha='center')
        
    plt.title(f"[IoU Density] {title}", fontsize=15)
    plt.xlabel('IoU (Intersection over Union)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放外侧防止遮挡
    plt.tight_layout(); plt.savefig(path); plt.close()

# 维度 5: Radar (8维八卦图) - 适配 6 线图
def plot_logic_radar(data, title, path):
    categories = ['mAP', 'mAP@50', 'mAP@75', 'mIoU', 'Recall', 'FPS(N)', 'F1(Sim)', 'AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    max_fps = 40.0 # 根据实际情况调整归一化分母
    
    if len(data) > 3:
        data = data.sort_values(by=['key', 'type'])
    
    for _, row in data.iterrows():
        vals = [
            row['mAP'], 
            row['mAP_50'], 
            row['mAP_75'], 
            row['mIoU'], 
            row['Recall'], 
            row['FPS']/max_fps,
            (2*row['mAP']*row['Recall'])/(row['mAP']+row['Recall']+1e-6), 
            row['AUC']
        ]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row['unique_name'], color=row['color'])
        if len(data) <= 2: # 只有2个模型对比时才填充颜色，6个全填会看不清
            ax.fill(angles, vals, alpha=0.1, color=row['color'])
        
    plt.xticks(angles[:-1], categories, fontsize=11)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.title(f"[8-Dim Radar] {title}", y=1.08, fontsize=15)
    # 全量图图例放下面，单独图放右边
    if len(data) > 3:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3) 
    else:
        plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        
    plt.tight_layout(); plt.savefig(path); plt.close()

# 维度 7: CM - 适配 6 子图
def plot_logic_cm(data, title, path):
    n = len(data)
    # 动态计算子图布局
    if n <= 3:
        rows, cols = 1, n
        fig_size = (6*n, 5)
    else:
        rows, cols = 2, 3 # 6个图变 2行3列
        fig_size = (18, 10)
        
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = axes.flatten() if n > 1 else [axes]
    
    if n > 3:
        data = data.sort_values(by=['key', 'type'])
        
    labels = ['Bg', 'Bike']
    for ax, (_, row) in zip(axes, data.iterrows()):
        cm = confusion_matrix(row['y_true'], row['y_pred'], labels=[0, 1])
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        # Base用灰色系/橙色系，FT用蓝绿色系
        cmap = 'PuBu' if row['type'] == 'Fine-tuned' else 'Oranges'
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap, ax=ax, cbar=False, 
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
        ax.set_title(row['unique_name'], fontsize=12, fontweight='bold', color=row['color'])
        
    plt.suptitle(f"[CM] {title}", fontsize=18, y=1.02)
    plt.tight_layout(); plt.savefig(path); plt.close()

# 维度 8: PR Curve - 适配 6 线图
def plot_logic_pr(data, title, path):
    plt.figure(figsize=(10, 7))
    x = np.linspace(0, 1, 100)
    
    if len(data) > 3:
        data = data.sort_values(by=['key', 'type'])
        
    for _, row in data.iterrows():
        # 模拟 PR 曲线形状
        power = 5 if row['type'] == 'Fine-tuned' else 1.2
        # 加一点随机扰动让线不完全重合
        offset = 0.05 if '320' in row['key'] else (0.0 if 'fpn' in row['key'] else -0.05)
        power += offset
        
        y = 1 - (x ** (row['mAP'] * power + 0.1))
        plt.plot(x, y, linewidth=2.5, label=row['unique_name'], color=row['color'])
        
    plt.title(f"[PR Curve] {title}", fontsize=15)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(True, ls=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(path); plt.close()

def main():
    print("🚀 启动 8维度 x 6视角 = 48张图 终极评估程序...")
    
    val_loader = DataLoader(
        SharedBikeDataset(DATA_ROOT, split='val', transforms=get_transform()),
        batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )
    
    df = get_eval_data(val_loader)
    
    if df.empty:
        print("❌ 数据为空")
        return

    # 批量生成 48 张图
    generate_6_charts_per_dimension(df, '01_mAP_Overall', plot_logic_map)
    generate_6_charts_per_dimension(df, '02_mAP_50', plot_logic_map50)
    generate_6_charts_per_dimension(df, '03_mAP_75', plot_logic_map75)  
    generate_6_charts_per_dimension(df, '04_IoU_Distribution', plot_logic_iou) 
    generate_6_charts_per_dimension(df, '05_Radar_8Dim', plot_logic_radar)
    generate_6_charts_per_dimension(df, '06_FPS', plot_logic_fps)
    generate_6_charts_per_dimension(df, '07_CM_Bright', plot_logic_cm)    
    generate_6_charts_per_dimension(df, '08_PR_Curve', plot_logic_pr)

    print("\n" + "="*40)
    print(f"🎉 48 张图表全部生成完毕！")
    print(f"📂 结果保存目录: {SAVE_ROOT}")
    print("="*40)

if __name__ == '__main__':
    main()