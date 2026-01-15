import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import warnings

# 忽略警告
warnings.filterwarnings("ignore")


def load_image(path):
    """读取图像，若不存在则报错"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img


def display_image_info(img, title="Image Info"):
    """打印图像尺寸信息"""
    print(f"{title}: Shape = {img.shape}")


def convert_to_rgb(img):
    """将 BGR 图像转换为 RGB 用于 Matplotlib 显示"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compare_images(img1, img2, titles=("Original Image", "Processed Image"), figsize=(16, 5)):
    """并排显示两张图像进行对比"""
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


def gaussian_blur(image, kernel_size):
    """高斯模糊去噪"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, low_threshold, high_threshold):
    """Canny 边缘检测"""
    return cv2.Canny(image, low_threshold, high_threshold)


def timed_function(func):
    """装饰器：计算函数运行时间"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        print(f"{func.__name__} took {end:.4f}s")
        return result

    return wrapper


# --- 核心修改 1: 宽视野 ROI 函数 ---
def get_wide_roi_mask(image):
    """
    生成宽视野 ROI 掩膜。
    为了防止切掉两侧的车道线，我们将梯形顶部大幅拉宽。
    """
    height = image.shape[0]
    width = image.shape[1]

    # 定义梯形顶点：
    # 顶部宽度范围：图像宽度的 10% 到 90%
    # 高度截断：只保留图像底部 40% 的区域 (height * 0.6 以下)
    polygons = np.array([
        [
            (0, height),  # 左下
            (int(width * 0.1), int(height * 0.6)),  # 左上 (极宽)
            (int(width * 0.9), int(height * 0.6)),  # 右上 (极宽)
            (width, height)  # 右下
        ]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


@timed_function
def hough_lines(img: np.ndarray, rho: float, theta: float, threshold: int, min_line_length: int,
                max_line_gap: int) -> np.ndarray:
    """霍夫变换检测直线"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines


# --- 核心修改 2: 优化的绘制函数 ---
def draw_lanes_optimized(image: np.ndarray, lines: np.ndarray) -> np.ndarray:

    # 创建一个纯黑背景用于画线
    line_image = np.zeros_like(image)
    height = image.shape[0]

    if lines is None:
        print("警告：未检测到任何车道线！请检查 Canny 阈值或 ROI 区域。")
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 1. 长度过滤 (可选，防止极短噪点)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 20:
                continue

            # 2. 计算斜率
            if x2 - x1 == 0:
                slope = 999.0
            else:
                slope = (y2 - y1) / (x2 - x1)

            # 3. 核心过滤逻辑
            # 斜率：去掉水平线 (abs < 0.2)
            # 位置：去掉出现在图像上半部分的线 (y < height * 0.5) <- 这是为了防止宽 ROI 把树当成路
            if abs(slope) > 0.2:
                if y1 > height * 0.5 and y2 > height * 0.5:
                    # 绘制线段：绿色，线宽 5
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # 融合
    fusion = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return fusion


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 加载图像
    image_path = 'test3.jpg'
    try:
        img = load_image(image_path)
    except FileNotFoundError as e:
        print(e)
        exit()

    display_image_info(img)

    # 2. 预处理
    # 使用较大的高斯核 (7) 来抑制树木纹理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss_img = gaussian_blur(gray_img, 7)

    # 3. Canny 边缘检测
    # 适当提高低阈值以减少噪声
    cannyd_img = canny(gauss_img, 50, 150)
    compare_images(cv2.cvtColor(gauss_img, cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cannyd_img, cv2.COLOR_GRAY2RGB),
                   titles=("Blurred", "Canny Edges"))

    # 4. ROI 提取
    crop_mask = get_wide_roi_mask(cannyd_img)

    compare_images(cv2.cvtColor(cannyd_img, cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2RGB),
                   titles=("Canny", "Wide ROI Canny"))

    # 5. 霍夫变换
    # 参数调整：threshold 降低以检测断线，gap 增大以连接虚线
    rho = 1
    theta = np.pi / 180
    hof_threshold = 20
    min_line_len = 40
    max_line_gap = 150

    lines_hough = hough_lines(crop_mask, rho, theta, hof_threshold, min_line_len, max_line_gap)

    # 6. 绘制结果
    final_result = draw_lanes_optimized(img.copy(), lines_hough)

    # 7. 显示最终结果
    compare_images(img, final_result, titles=("Original Image", "All Detected Lanes"))