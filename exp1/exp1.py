import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os


# ============ 中文字体设置 ============
# 方法1: 检查系统中可用的中文字体
def setup_chinese_font():
    """设置中文字体，避免警告"""
    # 获取系统字体列表
    font_dirs = [
        'C:/Windows/Fonts/',  # Windows字体目录
        '/System/Library/Fonts/',  # macOS字体目录
        '/usr/share/fonts/'  # Linux字体目录
    ]

    # 常见中文字体名称
    chinese_fonts = [
        'msyh.ttc', 'simhei.ttf', 'simsun.ttc',  # Windows
        'PingFang.ttc', 'Hei.ttc', 'Song.ttc',  # macOS
        'wqy-microhei.ttc', 'wqy-zenhei.ttc'  # Linux
    ]

    # 临时解决方案：如果找不到中文字体，就使用英文
    try:
        # 尝试找到第一个可用的中文字体
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for chinese_font in chinese_fonts:
                    font_path = os.path.join(font_dir, chinese_font)
                    if os.path.exists(font_path):
                        matplotlib.font_manager.fontManager.addfont(font_path)
                        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                        matplotlib.rcParams['font.sans-serif'] = [font_name]
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        print(f"使用中文字体: {font_name}")
                        return
    except:
        pass

    # 如果找不到中文字体，使用默认设置
    print("未找到中文字体，使用英文显示")
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


# 设置中文字体
setup_chinese_font()


# ============ 图像处理函数 ============

def manual_convolution(image, kernel):
    """手动实现卷积运算"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    img_h, img_w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded_img = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w,
                                    cv2.BORDER_REPLICATE)

    output = np.zeros((img_h, img_w), dtype=np.float32)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    output = np.abs(output)

    output_min = output.min()
    output_max = output.max()
    if output_max > output_min:
        output = ((output - output_min) / (output_max - output_min) * 255).astype(np.uint8)
    else:
        output = np.zeros_like(output, dtype=np.uint8)

    return output


def get_histogram(image, bins=256, normalize=False):
    """手动计算颜色直方图"""
    if len(image.shape) == 3:
        h, w, c = image.shape
        histograms = []

        for channel in range(c):
            hist = np.zeros(bins)
            channel_data = image[:, :, channel].ravel()

            for pixel in channel_data:
                bin_idx = min(int(pixel * bins / 256), bins - 1)
                hist[bin_idx] += 1

            if normalize:
                hist = hist / (h * w)

            histograms.append(hist)

        return histograms
    else:
        h, w = image.shape
        hist = np.zeros(bins)

        for pixel in image.ravel():
            bin_idx = min(int(pixel * bins / 256), bins - 1)
            hist[bin_idx] += 1

        if normalize:
            hist = hist / (h * w)

        return hist


def get_lbp_texture(image):
    """手动提取简易LBP纹理特征"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    h, w = image.shape
    if h < 3 or w < 3:
        return np.array([], dtype=np.uint8)

    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = image[i, j]
            code = 0
            code |= (image[i - 1, j - 1] >= center) << 7
            code |= (image[i - 1, j] >= center) << 6
            code |= (image[i - 1, j + 1] >= center) << 5
            code |= (image[i, j + 1] >= center) << 4
            code |= (image[i + 1, j + 1] >= center) << 3
            code |= (image[i + 1, j] >= center) << 2
            code |= (image[i + 1, j - 1] >= center) << 1
            code |= (image[i, j - 1] >= center) << 0
            lbp[i - 1, j - 1] = code

    return lbp


def calculate_histogram_statistics(histogram):
    """计算直方图的统计特征"""
    indices = np.arange(len(histogram))

    total = np.sum(histogram)
    if total == 0:
        return {
            'mean': 0,
            'variance': 0,
            'std_dev': 0,
            'mode': 0,
            'entropy': 0
        }

    mean = np.sum(indices * histogram) / total
    variance = np.sum(histogram * (indices - mean) ** 2) / total
    std_dev = np.sqrt(variance)
    mode = np.argmax(histogram)

    prob = histogram / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))

    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'mode': mode,
        'entropy': entropy
    }


def visualize_results_without_chinese(img, gray, sobel_filtered, custom_filtered,
                                      hist_rgb, hist_gray):
    """可视化结果"""
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Sobel滤波结果
    plt.subplot(2, 3, 2)
    plt.title("Sobel Filter Result")
    if sobel_filtered.size > 0:
        plt.imshow(sobel_filtered, cmap='gray')
    plt.axis('off')

    # 自定义核滤波结果
    plt.subplot(2, 3, 3)
    plt.title("Custom Kernel Result")
    if custom_filtered.size > 0:
        plt.imshow(custom_filtered, cmap='gray')
    plt.axis('off')

    # 灰度直方图
    plt.subplot(2, 3, 4)
    plt.bar(range(len(hist_gray)), hist_gray, width=1.0, color='gray', alpha=0.7)
    plt.title("Gray Histogram (Normalized)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)

    # RGB通道直方图
    plt.subplot(2, 3, 5)
    colors = ['red', 'green', 'blue']
    for i, (hist, color) in enumerate(zip(hist_rgb, colors)):
        plt.plot(hist, color=color, alpha=0.7, label=['R', 'G', 'B'][i])
    plt.title("RGB Channel Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 统计摘要（英文）
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = "Histogram Statistics:\n"
    for i, (hist, name) in enumerate(zip(hist_rgb, ['R', 'G', 'B'])):
        stats = calculate_histogram_statistics(hist)
        stats_text += f"\n{name} Channel:\n"
        stats_text += f"Mean: {stats['mean']:.1f}\n"
        stats_text += f"Std Dev: {stats['std_dev']:.1f}\n"
        stats_text += f"Mode: {stats['mode']}\n"
        stats_text += f"Entropy: {stats['entropy']:.2f}\n"
    plt.text(0.1, 0.5, stats_text, fontsize=10,
             verticalalignment='center', family='monospace')

    plt.tight_layout()
    plt.savefig('experiment1_results.png', dpi=300, bbox_inches='tight')
    print("Summary image saved: experiment1_results.png")
    plt.show()


# ============ 主程序 ============

def main():
    print("=" * 60)
    print("Machine Vision Experiment 1: Image Filtering and Feature Extraction")
    print("=" * 60)

    # 1. 读取图像
    img_path = 'my_photo.jpg'
    print(f"\n[1] Loading image: {img_path}")

    if not os.path.exists(img_path):
        print("Warning: my_photo.jpg not found, creating test image...")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img[i, j] = [i % 256, j % 256, (i + j) % 256]
        cv2.imwrite('my_photo.jpg', img)
    else:
        img = cv2.imread(img_path)

    if img is None:
        print("Error: Failed to read image, creating default image...")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img[i, j] = [i % 256, j % 256, (i + j) % 256]

    print(f"Image shape: {img.shape}")
    print(f"Image data type: {img.dtype}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Gray image shape: {gray.shape}")

    # 2. Sobel滤波
    print("\n[2] Applying Sobel filter...")
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_filtered = manual_convolution(gray, sobel_x)
    print(f"Sobel result shape: {sobel_filtered.shape}")
    print(f"Sobel result range: [{sobel_filtered.min()}, {sobel_filtered.max()}]")

    # 3. 自定义核滤波
    print("\n[3] Applying custom kernel filter...")
    custom_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]], dtype=np.float32)
    custom_filtered = manual_convolution(gray, custom_kernel)
    print(f"Custom filter result range: [{custom_filtered.min()}, {custom_filtered.max()}]")

    # 4. 直方图计算
    print("\n[4] Calculating histograms...")
    hist_rgb = get_histogram(img, bins=256, normalize=False)
    hist_gray = get_histogram(gray, bins=256, normalize=True)
    print(f"RGB histogram shapes: R={hist_rgb[0].shape}, G={hist_rgb[1].shape}, B={hist_rgb[2].shape}")
    print(f"Gray histogram shape: {hist_gray.shape}")

    # 5. 统计特征
    print("\n[5] Calculating histogram statistics...")
    for i, (hist, name) in enumerate(zip(hist_rgb, ['Red', 'Green', 'Blue'])):
        stats = calculate_histogram_statistics(hist)
        print(f"{name} channel:")
        print(f"  Mean: {stats['mean']:.2f}, Std Dev: {stats['std_dev']:.2f}")
        print(f"  Mode: {stats['mode']}, Entropy: {stats['entropy']:.2f}")

    # 6. 纹理特征提取
    print("\n[6] Extracting texture features...")
    texture_feat = get_lbp_texture(gray)
    print(f"Texture feature shape: {texture_feat.shape}")
    np.save('texture_features.npy', texture_feat)
    print("Texture features saved: texture_features.npy")

    # 7. 保存图像
    print("\n[7] Saving output images...")
    cv2.imwrite('original_image.jpg', img)
    print("Original image saved: original_image.jpg")

    if sobel_filtered.size > 0:
        cv2.imwrite('sobel_filtered.jpg', sobel_filtered)
        print("Sobel filtered image saved: sobel_filtered.jpg")

    if custom_filtered.size > 0:
        cv2.imwrite('custom_filtered.jpg', custom_filtered)
        print("Custom filtered image saved: custom_filtered.jpg")

    # 8. 可视化
    print("\n[8] Generating visualizations...")
    visualize_results_without_chinese(img, gray, sobel_filtered, custom_filtered,
                                      hist_rgb, hist_gray)

    # 9. 保存更多直方图视图
    plt.figure(figsize=(12, 4))

    # RGB叠加直方图
    plt.subplot(1, 3, 1)
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']
    for i, (hist, color, label) in enumerate(zip(hist_rgb, colors, labels)):
        plt.plot(hist, color=color, alpha=0.7, label=label)
    plt.title("RGB Histograms (Overlay)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 灰度直方图
    plt.subplot(1, 3, 2)
    plt.bar(range(len(hist_gray)), hist_gray, width=1.0, color='gray', alpha=0.7)
    plt.title("Gray Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)

    # 直方图统计条形图
    plt.subplot(1, 3, 3)
    channel_names = ['Red', 'Green', 'Blue']
    means = [calculate_histogram_statistics(hist)['mean'] for hist in hist_rgb]
    std_devs = [calculate_histogram_statistics(hist)['std_dev'] for hist in hist_rgb]

    x = np.arange(len(channel_names))
    width = 0.35

    plt.bar(x - width / 2, means, width, label='Mean', color='skyblue')
    plt.bar(x + width / 2, std_devs, width, label='Std Dev', color='lightcoral')

    plt.title("Histogram Statistics")
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.xticks(x, channel_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=300, bbox_inches='tight')
    print("Histogram analysis saved: histogram_analysis.png")

    # 10. 生成实验报告摘要
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE!")
    print("=" * 60)

    print("\nGenerated files:")
    print("1. original_image.jpg - Original image")
    print("2. sobel_filtered.jpg - Sobel filtered image")
    print("3. custom_filtered.jpg - Custom kernel filtered image")
    print("4. texture_features.npy - Texture features (LBP)")
    print("5. experiment1_results.png - Main results summary")
    print("6. histogram_analysis.png - Detailed histogram analysis")

    print("\nImage Statistics:")
    print(f"Original image size: {img.shape[1]} x {img.shape[0]}")
    print(f"Gray image size: {gray.shape[1]} x {gray.shape[0]}")
    print(f"Texture features size: {texture_feat.shape[1]} x {texture_feat.shape[0]}")

    print("\nResults Summary:")
    print("- Sobel filter: Edge detection in horizontal direction")
    print("- Custom kernel: Custom edge detection filter")
    print("- Histograms: RGB and gray channel distributions")
    print("- Texture: Local Binary Pattern (LBP) features extracted")


if __name__ == "__main__":
    main()