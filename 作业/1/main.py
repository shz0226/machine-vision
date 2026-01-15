import numpy as np
import cv2
import random
import time


class TextureSynthesizer:
    def __init__(self, sample_path, window_size, sigma_ratio=6.4):
        self.window_size = window_size
        self.half_w = window_size // 2

        # 读取样本并归一化
        self.sample = cv2.imread(sample_path) / 255.0
        if self.sample is None:
            raise ValueError(f"无法读取图片 {sample_path}，请检查路径是否正确")
        self.h, self.w, self.c = self.sample.shape

        # 预计算高斯核 (Gaussian Kernel)
        self.sigma = window_size / sigma_ratio
        kernel_1d = cv2.getGaussianKernel(window_size, self.sigma)
        self.kernel_2d = np.outer(kernel_1d, kernel_1d)

    def get_neighbors(self, filled_map):
        """找到所有未填充但与已知像素相邻的像素"""
        dilated = cv2.dilate(filled_map, np.ones((3, 3), np.uint8))
        boundary = dilated - filled_map
        y_idxs, x_idxs = np.where(boundary == 1)

        coords = list(zip(y_idxs, x_idxs))
        random.shuffle(coords)
        return coords

    def compute_distance(self, target_window, valid_mask, metric='ssd'):
        rows = self.h - self.window_size + 1
        cols = self.w - self.window_size + 1

        # 加权掩膜
        weights = self.kernel_2d * valid_mask
        total_weight = np.sum(weights)
        if total_weight == 0: total_weight = 1.0
        weights /= total_weight

        dists = np.zeros((rows, cols))

        # 全图遍历搜索 (Python原生较慢，但逻辑最准确)
        for sy in range(rows):
            for sx in range(cols):
                sample_patch = self.sample[sy:sy + self.window_size, sx:sx + self.window_size]

                # 计算差异
                diff = sample_patch - target_window

                if metric == 'ssd':
                    dist_pixel = np.sum(diff ** 2, axis=2)
                elif metric == 'l1':
                    dist_pixel = np.sum(np.abs(diff), axis=2)

                dists[sy, sx] = np.sum(dist_pixel * weights)

        return dists

    def synthesize(self, out_shape, metric='ssd', shape_type='square'):
        out_h, out_w = out_shape
        synth_img = np.zeros((out_h, out_w, self.c))
        filled_map = np.zeros((out_h, out_w), dtype=np.uint8)

        # 初始化种子 (Seed)
        seed_size = 3
        rand_y = random.randint(0, self.h - seed_size)
        rand_x = random.randint(0, self.w - seed_size)

        cy, cx = out_h // 2, out_w // 2
        synth_img[cy:cy + seed_size, cx:cx + seed_size] = \
            self.sample[rand_y:rand_y + seed_size, rand_x:rand_x + seed_size]
        filled_map[cy:cy + seed_size, cx:cx + seed_size] = 1

        total_pixels = out_h * out_w
        start_time = time.time()

        # 如果是圆形窗口，修改 kernel_2d
        # 注意：这里需要深拷贝一份 kernel，避免影响后续调用
        original_kernel = self.kernel_2d.copy()
        if shape_type == 'circular':
            y, x = np.ogrid[-self.half_w:self.half_w + 1, -self.half_w:self.half_w + 1]
            mask = x ** 2 + y ** 2 <= self.half_w ** 2
            self.kernel_2d *= mask

        print(f"开始合成... 目标尺寸: {out_shape}, Window: {self.window_size}, Metric: {metric}, Shape: {shape_type}")

        while np.sum(filled_map) < total_pixels:
            progress = np.sum(filled_map)
            print(f"\r进度: {progress}/{total_pixels} ({(progress / total_pixels) * 100:.1f}%)", end='')

            neighbors = self.get_neighbors(filled_map)

            for (py, px) in neighbors:
                y1 = max(0, py - self.half_w)
                y2 = min(out_h, py + self.half_w + 1)
                x1 = max(0, px - self.half_w)
                x2 = min(out_w, px + self.half_w + 1)

                wy1 = self.half_w - (py - y1)
                wy2 = self.half_w + (y2 - py)
                wx1 = self.half_w - (px - x1)
                wx2 = self.half_w + (x2 - px)

                target_window = np.zeros((self.window_size, self.window_size, self.c))
                valid_mask = np.zeros((self.window_size, self.window_size))

                img_patch = synth_img[y1:y2, x1:x2]
                mask_patch = filled_map[y1:y2, x1:x2]

                target_window[wy1:wy2, wx1:wx2] = img_patch
                valid_mask[wy1:wy2, wx1:wx2] = mask_patch

                dists = self.compute_distance(target_window, valid_mask, metric)

                min_val = np.min(dists)
                threshold = min_val * (1 + 0.1)
                candidates = np.argwhere(dists <= threshold)

                chosen = candidates[random.randint(0, len(candidates) - 1)]
                sy, sx = chosen[0], chosen[1]

                center_y = sy + self.half_w
                center_x = sx + self.half_w

                if center_y < self.h and center_x < self.w:
                    synth_img[py, px] = self.sample[center_y, center_x]
                    filled_map[py, px] = 1

                break

                # 恢复 kernel 以防后续调用受影响
        self.kernel_2d = original_kernel
        print(f"\n完成! 用时: {time.time() - start_time:.2f}s")
        return (synth_img * 255).astype(np.uint8)


# ================= 核心执行部分 =================
if __name__ == '__main__':
    # 确保 fig.jpg 在同一目录下
    input_image = "fig.jpg"

    # 统一设定为 100x100
    target_size = (100, 100)

    # 初始化模型，使用最佳窗口大小 11
    # 注意：这里不需要再循环 5, 11, 23 了，只跑 11 即可
    model = TextureSynthesizer(input_image, window_size=11)

    print("=== 任务 1/2: 生成 L1 Metric 对比图 ===")
    # 修正点：将 (60, 60) 改为 target_size (100, 100)
    result_l1 = model.synthesize(target_size, metric='l1', shape_type='square')
    cv2.imwrite('result_metric_L1.jpg', result_l1)
    print("已保存: result_metric_L1.jpg")

    print("\n=== 任务 2/2: 生成 Circular Shape 对比图 ===")
    # 修正点：将 (60, 60) 改为 target_size (100, 100)
    result_circle = model.synthesize(target_size, metric='ssd', shape_type='circular')
    cv2.imwrite('result_shape_circular.jpg', result_circle)
    print("已保存: result_shape_circular.jpg")

    print("\n所有图片生成完毕！")