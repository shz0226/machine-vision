import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# --- 第一部分：卷积输出维度计算 ---
def calculate_conv_output(n, m, k, s, padding_method="valid"):
    """
    计算卷积层输出维度
    n, m: 输入的高和宽
    k: 卷积核大小 (k x k)
    s: 步长 (stride)
    padding_method: "valid" 或 "same" (假设 same padding 时 s=1)
    """
    p = 0
    if padding_method == "same":
        p = (k - 1) // 2

    h_out = int(np.floor((n + 2 * p - k) / s) + 1)
    w_out = int(np.floor((m + 2 * p - k) / s) + 1)

    return h_out, w_out


# --- 第二部分：反向传播算法演示 ---
def run_backpropagation_demo():
    # 1. 准备数据: y = 2x + 1
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

    # 2. 定义简单网络 (线性层)
    # 包含权重 W 和 偏置 b
    model = nn.Linear(1, 1)

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print(f"【初始状态】 Weights: {model.weight.item():.4f}, Bias: {model.bias.item():.4f}")

    loss_history = []

    # 4. 训练循环 (模拟 BP 过程)
    epochs = 500
    for epoch in range(epochs):
        # A. 前向传播 (Forward)
        y_pred = model(X)

        # B. 计算损失 (Compute Loss)
        loss = criterion(y_pred, Y)
        loss_history.append(loss.item())

        # C. 反向传播 (Backward) -> 核心步骤
        optimizer.zero_grad()  # 清空旧梯度
        loss.backward()  # 自动计算 dLoss/dW 和 dLoss/db

        # D. 参数更新 (Update)
        optimizer.step()  # W = W - lr * grad

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}: Loss = {loss.item():.6f}, W = {model.weight.item():.4f}, b = {model.bias.item():.4f}")

    return loss_history


if __name__ == '__main__':
    # 1. 测试卷积维度
    print("--- 卷积维度计算结果 ---")
    inputs = [(32, 32, 3, 1, 'valid'), (32, 32, 3, 1, 'same'), (32, 32, 5, 2, 'valid')]
    for (n, m, k, s, p) in inputs:
        out = calculate_conv_output(n, m, k, s, p)
        print(f"输入: {n}x{m}, 核: {k}x{k}, 步长: {s}, Padding: {p} -> 输出: {out}")

    # 2. 运行反向传播
    print("\n--- 反向传播训练过程 ---")
    losses = run_backpropagation_demo()

    # 3. 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Backpropagation Training Process (Loss Descent)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()