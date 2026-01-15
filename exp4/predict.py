import torch
import cv2
import glob
import os
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from src.model import get_finetune_model

INPUT_DIR = './my_campus_photos'
SAVE_DIR = './plots/predictions'
MODEL_PATH = './output/best_model.pth'
CLASSES = ['BG', 'Bicycle', 'Shared-Bike'] # 0, 1, 2
COLORS = [(0,0,0), (0, 255, 0), (255, 165, 0)] # 绿=私有, 橙=共享

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')
    
    model = get_finetune_model(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    
    transform = T.Compose([T.ToTensor()])
    img_paths = glob.glob(os.path.join(INPUT_DIR, '*'))
    
    print(f"开始推理 {len(img_paths)} 张图片...")
    for img_path in tqdm(img_paths):
        try:
            # 读取
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = transform(pil_img).to(device)
            
            # 预测
            with torch.no_grad():
                pred = model([img_tensor])[0]
            
            # 绘图
            cv_img = cv2.imread(img_path)
            if cv_img is None: continue
            
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                if score < 0.5: continue
                
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                cls_id = label.item()
                color = COLORS[cls_id] if cls_id < len(COLORS) else (255,255,255)
                
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                text = f"{CLASSES[cls_id]} {score:.2f}"
                cv2.putText(cv_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            save_path = os.path.join(SAVE_DIR, 'pred_' + os.path.basename(img_path))
            cv2.imwrite(save_path, cv_img)
            
        except Exception as e:
            print(f"Error {img_path}: {e}")
            
    print("预测完成，结果在 plots/predictions")

if __name__ == '__main__':
    main()