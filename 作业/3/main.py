import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def stereo_matching_save(left_img_path, right_img_path):
    # --- 1. 准备保存路径 ---
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查输入文件
    if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
        print("【错误】输入图片路径不正确")
        return

    # --- 2. 核心算法流程 ---
    imgL = cv2.imread(left_img_path, 0)
    imgR = cv2.imread(right_img_path, 0)

    # SGBM 参数配置
    window_size = 3
    min_disp = 0
    num_disp = 16 * 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    print("正在计算视差，请稍候...")
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # 归一化 (用于显示和保存纯图)
    disp_map_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 给纯图上色 (伪彩色)
    disp_color = cv2.applyColorMap(disp_map_norm, cv2.COLORMAP_PLASMA)

    # --- 3. 保存与显示结果 ---

    # (A) 保存纯视差图 (彩色版)
    save_path_disp = os.path.join(output_dir, 'stereo_disparity_only.png')
    cv2.imwrite(save_path_disp, disp_color)
    print(f"【保存】纯视差图已保存至: {save_path_disp}")

    # (B) 保存对比展示图 (最适合放作业文档)
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(imgL, 'gray')
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(imgR, 'gray')
    plt.title('Right Image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(disp_map_norm, 'plasma')
    plt.title('Disparity Map')
    plt.axis('off')

    # 保存整张画布
    save_path_plot = os.path.join(output_dir, 'stereo_comparison.png')
    plt.savefig(save_path_plot, bbox_inches='tight', dpi=150)
    print(f"【保存】对比展示图已保存至: {save_path_plot}")

    plt.show()


if __name__ == '__main__':
    # 记得用你之前的 im2.png 和 im6.png
    stereo_matching_save('im2.png', 'im6.png')