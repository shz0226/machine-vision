import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 卷积层 1: 输入1通道，输出32通道，核大小3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层 2: 输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout层：防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        # 经过两次2x2池化，28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10个数字分类

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.dropout1(x)

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x