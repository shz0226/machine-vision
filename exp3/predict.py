import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from model import ConvNet
from config import Config
from utils import pad_resize_digit


def predict_image(image_path):
    # --- 1. 加载模型 ---
    device = Config.DEVICE
    model = ConvNet().to(device)
    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"❌ 模型文件未找到: {Config.MODEL_PATH}")
        return
    model.eval()

    # --- 2. 读取图片 ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 为了统一处理标准，先把图片高度 Resize 到 1000 像素
    h, w = img.shape[:2]
    scale_ratio = 1000 / h
    new_w = int(w * scale_ratio)
    img = cv2.resize(img, (new_w, 1000))
    img_display = img.copy() 

    # --- 3. 图像预处理 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊：去除纸张的噪点颗粒
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值：应对光照不均匀
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 10)

    # 形态学操作：闭运算（连接断开的笔画）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- 4. 轮廓提取与筛选 ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 计算宽高比
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)

        # 更严格的筛选条件
        if area > 400 and h > 30 and aspect_ratio < 1.5:
            digit_rects.append((x, y, w, h))

    # 从左到右排序
    digit_rects.sort(key=lambda x: x[0])

    # --- 5. 预测 ---
    result_str = ""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    print(f"🔍 过滤噪点后，检测到 {len(digit_rects)} 个有效数字...")

    for i, (x, y, w, h) in enumerate(digit_rects):
        roi = thresh[y:y + h, x:x + w]

        # 针对细长字体进行加粗，防止 resize 后特征消失
        kernel_dilate = np.ones((2, 2), np.uint8)
        roi = cv2.dilate(roi, kernel_dilate, iterations=1)

        roi_processed = pad_resize_digit(roi)

        img_tensor = transform(roi_processed).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1).item()
            result_str += str(prediction)

        # 在图片上绘制结果
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 3)  # 加粗边框
        cv2.putText(img_display, str(prediction), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)  # 加大字体

    # --- 6. 显示和保存结果 ---
    print("\n" + "=" * 40)
    print(f"📸 原始文件: {image_path}")
    print(f"🔢 识别结果: {result_str}")
    print("=" * 40)

    # --- 7. 自动保存结果图片到同目录 ---
    # 生成保存路径（在原文件同目录，文件名添加_result后缀）
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)
    save_path = os.path.join(dir_name, f"{file_name}_result{ext}")
    
    # 在图片顶部添加识别结果文本
    result_text = f"Result: {result_str}"
    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (img_display.shape[1] - text_size[0]) // 2
    text_y = 50
    
    # 添加半透明背景
    overlay = img_display.copy()
    cv2.rectangle(overlay, (text_x - 10, text_y - 40), 
                 (text_x + text_size[0] + 10, text_y + 10), (200, 200, 200), -1)
    alpha = 0.7
    img_display = cv2.addWeighted(overlay, alpha, img_display, 1 - alpha, 0)
    
    # 添加文字
    cv2.putText(img_display, result_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # 保存结果图片
    cv2.imwrite(save_path, img_display)
    print(f"✅ 结果图片已保存至: {save_path}")

    # --- 8. 显示最终结果 ---
    display_h = 600
    display_ratio = display_h / img_display.shape[0]
    display_w = int(img_display.shape[1] * display_ratio)
    final_show = cv2.resize(img_display, (display_w, display_h))

    # 添加窗口标题
    cv2.imshow(f"识别结果: {result_str} | 按任意键关闭", final_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_image('test.jpg') 