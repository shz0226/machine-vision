import cv2
import numpy as np


def pad_resize_digit(img_gray, target_size=28):

    h, w = img_gray.shape

    # 1. 保持比例缩放
    scale = 20.0 / max(h, w)  # 将最大边缩放到20像素(留出边距)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. 创建黑色背景
    final_img = np.zeros((target_size, target_size), dtype=np.uint8)

    # 3. 将数字居中放置
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    final_img[top:top + new_h, left:left + new_w] = resized

    return final_img