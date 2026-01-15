import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob

class SharedBikeDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        """
        root_dir: 指向 'cycledata' 目录
        split: 'train' 或 'val'
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # 1. 确定图片和标签文件夹路径
        # 根据截图，结构是 cycledata/images/train 和 cycledata/labels/train
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)
        
        # 2. 获取所有图片文件
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        # 3. 定义类别 (注意：YOLO txt中 0通常是第一类，1是第二类)
        # 根据经验和你的描述: 0 -> bicycle, 1 -> shared-bicycle
        # Faster R-CNN 需要 0 作为背景，所以我们要把 ID + 1
        self.class_mapping = {
            0: 1,  # 原来的 0 -> 现在是 1 (Bicycle)
            1: 1,  # 原来的 1 -> 现在也是 1 (Bicycle, 防御性映射)
            2: 1   # 原来的 2 -> 现在也是 1 (Bicycle, 防御性映射)
        }

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size # 获取原图尺寸，用于反归一化
        
        # 构造对应的 label 文件路径
        # 例如: .../images/train/IMG_123.jpg -> .../labels/train/IMG_123.txt
        file_name = os.path.basename(img_path)
        label_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                x_center, y_center, width, height = parts[1:]
                
                # --- YOLO 格式转 Faster R-CNN 格式 ---
                # YOLO: x_center, y_center, w, h (都是 0-1 归一化值)
                # Faster R-CNN: x_min, y_min, x_max, y_max (绝对像素坐标)
                
                x_c = x_center * w
                y_c = y_center * h
                box_w = width * w
                box_h = height * h
                
                x_min = x_c - box_w / 2
                y_min = y_c - box_h / 2
                x_max = x_c + box_w / 2
                y_max = y_c + box_h / 2
                
                # 稍微修正一下边界，防止超出图片
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                # 过滤无效框
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    # 映射类别 ID
                    labels.append(self.class_mapping.get(cls_id, 0))
        
        # 转换为 Tensor
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # 处理没有目标的负样本 (虽然训练集通常都有目标)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.img_files)

def get_transform():
    return T.Compose([T.ToTensor()])