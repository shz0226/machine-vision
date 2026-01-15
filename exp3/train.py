import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import ConvNet
from dataset import get_dataloaders
import time
import sys
from tqdm import tqdm


def train():
    # 1. 初始化
    train_loader, val_loader, _ = get_dataloaders()
    model = ConvNet().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print(f"🚀 开始训练，使用设备: {Config.DEVICE}")
    print(f"📊 训练集: {len(train_loader.dataset)} | 验证集: {len(val_loader.dataset)}")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # --- 训练循环 ---
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, ncols=100, desc="Training", file=sys.stdout) as train_bar:
            for data, target in train_bar:
                data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0

        with tqdm(val_loader, ncols=100, desc="Validating", leave=False, file=sys.stdout) as val_bar:
            with torch.no_grad():
                for data, target in val_bar:
                    data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

        # 结果计算
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # 打印总结（此时进度条已强制关闭，不会再错位）
        print(f"Configs: Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"🏆 恭喜！模型准确率提升，已保存为 {Config.MODEL_PATH}")

    print("\n" + "=" * 30)
    print(f"🏁 训练结束，总耗时: {time.time() - start_time:.1f}s")
    print(f"🌟 最佳验证集准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    train()