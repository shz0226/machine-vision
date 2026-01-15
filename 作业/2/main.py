import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # 用于检查文件是否存在

def sift_detector(image_path):
    # 0. 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"【错误】找不到文件: {image_path}")
        print("请检查路径，或者使用绝对路径 (例如: C:/Users/name/Desktop/1.jpg)")
        return

    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"【错误】OpenCV无法读取图像，可能是格式不支持: {image_path}")
        return

    print(f"成功读取图像，尺寸: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 创建SIFT对象
    sift = cv2.SIFT_create()

    # 3. 检测关键点
    kp, des = sift.detectAndCompute(gray, None)
    print(f"检测到 {len(kp)} 个关键点")

    # 4. 绘制关键点
    img_sift = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 5. 显示结果
    plt.figure(figsize=(10, 8))
    # OpenCV读入是BGR，Matplotlib显示需要RGB，这里必须转换
    plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Features (Total: {len(kp)})')
    plt.axis('off')
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    image_file = 'image.jpg'
    sift_detector(image_file)