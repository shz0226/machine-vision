import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import SharedBikeDataset, get_transform

# 配置
DATA_ROOT = './cycledata'

def check_dataloader_labels():
    print("🕵️‍♂️ 正在核查验证集标签分布...")
    
    # 加载你的验证集
    dataset = SharedBikeDataset(DATA_ROOT, split='val', transforms=get_transform())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    label_counts = {0: 0, 1: 0, 2: 0, 'others': 0}
    total_boxes = 0
    images_with_shared = 0
    
    for _, targets in tqdm(loader):
        for t in targets:
            labels = t['labels']
            for lbl in labels:
                l = lbl.item()
                total_boxes += 1
                if l in label_counts:
                    label_counts[l] += 1
                else:
                    label_counts['others'] += 1
            
            # 检查这张图里有没有 2 (Shared)
            if 2 in labels:
                images_with_shared += 1

    print("\n" + "="*40)
    print(f"📊 标签统计结果 (Total Boxes: {total_boxes})")
    print(f"   Label 0 (Background): {label_counts[0]}")
    print(f"   Label 1 (Bicycle)   : {label_counts[1]}")
    print(f"   Label 2 (Shared)    : {label_counts[2]}  <-- 重点看这里！")
    print(f"   Label Others        : {label_counts['others']}")
    print("-" * 40)
    print(f"🖼️ 包含共享单车的图片数量: {images_with_shared} / {len(dataset)}")
    print("="*40)

    if label_counts[2] == 0:
        print("❌ 致命错误：验证集中根本没有读取到 Label=2 的数据！")
        print("   原因可能是：")
        print("   1. XML/TXT 标注文件里，共享单车的名字不是 'shared_bike' (可能是 'shared' 或其他)")
        print("   2. Dataset 代码里的 class_dict 映射写错了")
    else:
        print("✅ 数据集里有共享单车标签。")
        print("   如果依然跑分是0，那就是严重的类别不平衡或特征太像了。")

if __name__ == '__main__':
    check_dataloader_labels()