import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")


def load_image(path):

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img


def display_image_info(img, title="Image Info"):

    print(f"{title}: Shape = {img.shape}")


def convert_to_rgb(img):

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compare_images(img1, img2, titles=("Original Image", "Processed Image"), figsize=(16, 5)):

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.tight_layout()

    # 显示第一幅图像
    axes[0].imshow(convert_to_rgb(img1))
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    # 显示第二幅图像
    axes[1].imshow(convert_to_rgb(img2))
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    plt.show()

#将图像转换为灰度图
def convert_to_gray(img):
    temp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)


# 高斯模糊
def gaussian_blur(image, kernel_size):

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#边缘检测
def canny(image,low_threshold,high_threshold):
    return cv2.Canny(image,low_threshold,high_threshold)


# 生成mask掩膜，提取ROI
# ROI即感兴趣区域，英文：Region of interest

def region_of_interest(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:

    # 创建与输入图像相同大小的黑色掩模
    mask = np.zeros_like(image)

    # 根据图像的通道数确定掩模的颜色
    ignore_mask_color = (255,) * image.shape[2] if len(image.shape) > 2 else 255

    # 填充多边形区域为白色，以创建掩模
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # 将原图像与掩模进行按位与操作，以保留兴趣区域
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def timed_function(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        print(f"{func.__name__} took {end:.4f}s")
        return result

    return wrapper


@timed_function
def hough_lines(img: np.ndarray, rho: float, theta: float, threshold: int, min_line_length: int,
                max_line_gap: int) -> np.ndarray:
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines


# 车道线斜率显示
def draw_lines(image: np.ndarray, lines: np.ndarray) -> np.ndarray:

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 计算直线的斜率
            fit = np.polyfit((x1, x2), (y1, y2), deg=1)
            slope = fit[0]
            slope_str = f"{slope:.2f}"  # 格式化斜率字符串

            # 绘制直线
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)

            # 在终点附近绘制斜率文本
            cv2.putText(image, slope_str, (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    return image


def draw_lines_detection(image: np.ndarray, lines: np.ndarray) -> tuple:

    middle_x = image.shape[1] // 2
    left_slopes = []
    right_slopes = []
    center_slopes = []

    left_biases = []
    right_biases = []
    center_biases = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), deg=1)
            slope = fit[0]
            bias = fit[1]

            if -0.41 < slope < -0.30:  # 左边线
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
                left_slopes.append(slope)
                left_biases.append(bias)
            elif 0.38 < slope < 0.42:  # 右边线
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
                right_slopes.append(slope)
                right_biases.append(bias)
            elif slope > 1:  # 中心线
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                center_slopes.append(slope)
                center_biases.append(bias)

    # 计算平均值
    left_slope = np.mean(left_slopes) if left_slopes else 0
    left_bias = np.mean(left_biases) if left_biases else 0
    right_slope = np.mean(right_slopes) if right_slopes else 0
    right_bias = np.mean(right_biases) if right_biases else 0
    center_slope = np.mean(center_slopes) if center_slopes else 0
    center_bias = np.mean(center_biases) if center_biases else 0

    return image, left_slope, left_bias, right_slope, right_bias, center_slope, center_bias


def draw_all_line(img, slope, bias, color, y2=230):
    # 计算起点
    x1 = 0 if slope == 'left' else img.shape[1] if slope == 'right' else int((img.shape[0] - bias) / slope)
    y1 = int(slope * x1 + bias) if isinstance(slope, (int, float)) else img.shape[0]

    # 计算终点
    x2 = int((y2 - bias) / slope) if isinstance(slope, (int, float)) else x1

    # 绘制线条
    line_img = cv2.line(img.copy(), (x1, y1), (x2, y2), color, 10)

    return line_img


def draw_line(blank, slope, bias, color, y2=230):
    # 计算起点
    if slope == 'left':
        x1 = 0
        y1 = int(slope * x1 + bias)
    elif slope == 'right':
        x1 = blank.shape[1]
        y1 = int(slope * x1 + bias)
    else:
        y1 = blank.shape[0]
        x1 = int((y1 - bias) / slope)

    # 计算终点
    x2 = int((y2 - bias) / slope) if slope != 'left' and slope != 'right' else x1

    # 绘制线条
    cv2.line(blank, (x1, y1), (x2, y2), color, 20)


def draw_lines_and_fuse(img, left_slope, left_bias, right_slope, right_bias, center_slope, center_bias):
    # 创建空白图像
    blank = np.zeros_like(img)

    # 绘制左车道线
    draw_line(blank, left_slope, left_bias, (0, 255, 0))

    # 绘制右车道线
    draw_line(blank, right_slope, right_bias, (0, 0, 255))

    # 绘制中线
    draw_line(blank, center_slope, center_bias, (255, 0, 0))

    # 图像融合
    fusion = cv2.addWeighted(img, 0.8, blank, 1, 0)

    return fusion


# 主程序
if __name__ == "__main__":
    # 加载图像
    image_path = 'test.jpg'
    img = load_image(image_path)

    # 显示图像信息
    display_image_info(img)

    # 对比显示原始图像和处理后的图像
    compare_images(img, img, titles=("Original Image", "RGB Image"))

    gray_img = convert_to_gray(img)

    # 对比显示原始图像和灰度图像
    compare_images(img, gray_img, titles=("Original Image", "Grayscale Image"))

    # 对灰度图像进行高斯模糊处理
    gauss_img = gaussian_blur(gray_img, 5)

    # 对比显示灰度图像和高斯模糊后的图像
    compare_images(gray_img, gauss_img, titles=("Grayscale Image", "Gaussian Blurred Image"))

    # 灰度图直方图
    plt.hist(gauss_img.ravel(), 256, [0, 256])
    plt.title('hist of gray pixel')
    plt.show()

    cannyd_img = canny(gauss_img, 110, 140)
    compare_images(gauss_img, cannyd_img, titles=("Grayscale Image", "Canny Image"))

    # 获取图像尺寸
    imshape = img.shape

    # 设置掩膜区域
    temp_v = np.array([[(0, 350), (350, 225), (530, 245), (800, 355), (950, 1000), (0, 700)]], dtype=np.int32)

    # 调用函数裁剪图像
    crop_img = region_of_interest(img, temp_v)

    # 显示对比图
    compare_images(img, crop_img, titles=("Grayscale Image", "ROI Image"))

    crop_mask = region_of_interest(cannyd_img, temp_v)
    compare_images(cannyd_img, crop_mask, titles=("Canny Image", "ROI Canny Image"))

    # 参数设置
    rho = 1
    theta = np.pi / 180
    hof_threshold = 20
    min_line_len = 30
    max_line_gap = 100

    # 调用函数
    lines_hough = hough_lines(crop_mask, rho, theta, hof_threshold, min_line_len, max_line_gap)

    # 调用函数
    lined_image = draw_lines(img.copy(), lines_hough)

    # 显示对比图
    compare_images(img, lined_image, titles=("Original Image", "Lane line slope Image"))

    # 调用函数
    lined_image, left_slope, left_bias, right_slope, right_bias, center_slope, center_bias = draw_lines_detection(
        img.copy(), lines_hough)

    # 显示对比图
    compare_images(img, lined_image, titles=("Original Image", "Lane markings detection Image"))

    fusion = draw_lines_and_fuse(img, left_slope, left_bias, right_slope, right_bias, center_slope, center_bias)
    compare_images(img, fusion, titles=("Original Image", "Image blending--Lane markings detection results"))





