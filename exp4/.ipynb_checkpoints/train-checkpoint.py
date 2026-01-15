import torch
import os
import csv
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import SharedBikeDataset, get_transform
from src.model import get_model

# --- 参数设置 ---
parser = argparse.ArgumentParser()
# 参数名是 --model，所以解析后存在 args.model 中
parser.add_argument('--model', type=str, default='mb3_320', 
                    choices=['mb3_320', 'mb3_fpn', 'resnet50'], help='选择要训练的模型')
args = parser.parse_args()

MODEL_NAME = args.model
# 自动检测设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 30 
BATCH_SIZE = 4
DATA_ROOT = './cycledata'
OUTPUT_DIR = f'./output/{MODEL_NAME}' 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 启动训练任务: {MODEL_NAME}")
    print(f"📂 结果保存至: {OUTPUT_DIR}")
    print(f"⚙️  使用设备: {DEVICE}")
    
    # 1. 准备数据
    train_loader = DataLoader(
        SharedBikeDataset(DATA_ROOT, split='train', transforms=get_transform()),
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x)), 
        num_workers=4,  # 多进程读取
        pin_memory=True # 加速转GPU
    )
    
    # 2. 加载模型 
    # 【重点】这里是方案B：num_classes=2 (背景+自行车)
    # 修复点：这里原来写的是 args.model_key，现在改为 MODEL_NAME
    model = get_model(MODEL_NAME, num_classes=2, is_pretrained=True)
    model.to(DEVICE)
    
    # 3. 优化器与调度器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 每3轮衰减一次学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. 初始化日志
    log_path = os.path.join(OUTPUT_DIR, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'lr'])

    best_loss = float('inf')
    
    # 5. 开始训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ep_loss = 0
        
        # 进度条
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [{MODEL_NAME}]")
        
        for imgs, targets in loop:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # 前向传播计算 Loss
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # 记录
            loss_val = losses.item()
            ep_loss += loss_val
            loop.set_postfix(loss=loss_val)
            
        avg_loss = ep_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step()
        
        # 写入日志
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, current_lr])
            
        # 保存最佳模型 (根据 Loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            # 同时也保存一个最新的，防止断电
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_model.pth'))
            
    print(f"✅ {MODEL_NAME} 训练完成！最佳 Loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()