import cv2
import numpy as np
import matplotlib.pyplot as plt



def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]  # 图片底部
    y2 = int(y1 * 0.45)  # 延伸高度

    # 防止极端情况下斜率为0导致的计算错误
    if abs(slope) < 0.001: return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None

    # 获取图像中心点，用于物理分割左右区域
    midpoint = image.shape[1] // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # 1. 过滤掉太平坦的线（路面噪点通常斜率很小）
        if abs(slope) < 0.4:
            continue

        # 2. 通过“中线”和“斜率方向”双重判定
        # 左侧车道线：位于图像左半边 且 斜率为负
        if slope < 0 and x1 < midpoint and x2 < midpoint:
            left_fit.append((slope, intercept))
        # 右侧车道线：位于图像右半边 且 斜率为正
        elif slope > 0 and x1 > midpoint and x2 > midpoint:
            right_fit.append((slope, intercept))

    left_line = None
    right_line = None

    if len(left_fit) > 0:
        left_avg = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_avg)

    if len(right_fit) > 0:
        right_avg = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_avg)

    return left_line, right_line



def final_lane_detection_extended(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None

    # 1. HLS 颜色空间
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 2. 颜色提取
    white_mask = cv2.inRange(hls, np.array([0, 130, 0]), np.array([180, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([15, 0, 80]), np.array([45, 255, 255]))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 3. 边缘检测
    blur = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    # 4. ROI 范围
    height, width = edges.shape
    roi_top = 0.45
    roi_v = np.array([[(0, height),
                       (int(width * 0.1), int(height * roi_top)),
                       (int(width * 0.9), int(height * roi_top)),
                       (width, height)]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_v, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 5. 霍夫变换
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 20,
                            minLineLength=40, maxLineGap=250)

    # 6. 计算拟合
    left_line, right_line = average_slope_intercept(img, lines)

    # 7. 绘制
    line_img = np.zeros_like(img)
    if left_line is not None:
        cv2.line(line_img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 15)
    if right_line is not None:
        cv2.line(line_img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 15)

    weighted_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return weighted_img, masked_edges


# 运行
img_path = 'testx.jpg'
final_res, roi_debug = final_lane_detection_extended(img_path)

if final_res is not None:
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(roi_debug, cmap='gray'), plt.title('ROI Edges')
    plt.subplot(122), plt.imshow(cv2.cvtColor(final_res, cv2.COLOR_BGR2RGB)), plt.title('Final Result')
    plt.show()