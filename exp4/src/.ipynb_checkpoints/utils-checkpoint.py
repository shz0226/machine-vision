# src/utils.py
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    固定所有随机种子，确保实验结果可复现。
    这是科研实验中非常重要的一步，保证各种对比是公平的。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 全局随机种子已固定为: {seed}")

def collate_fn(batch):
    """
    自定义的 batch 整理函数。
    Faster R-CNN 需要的输入是一个 tuple (images, targets)。
    """
    return tuple(zip(*batch))