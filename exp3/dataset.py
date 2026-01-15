import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import Config


def get_dataloaders():
    # 定义预处理：转Tensor + 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    # 下载/加载完整的训练集 (60000张)
    full_train_dataset = datasets.MNIST(root=Config.DATA_PATH, train=True,
                                        download=True, transform=transform)

    # 下载/加载测试集 (10000张)
    test_dataset = datasets.MNIST(root=Config.DATA_PATH, train=False,
                                  download=True, transform=transform)

    # 【关键步骤】将60000张训练集划分为：54000训练 + 6000验证
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"数据加载完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
    return train_loader, val_loader, test_loader