import cv2
import matplotlib.pyplot as plt
import os


def run_viola_jones_detection(image_path):
    # --- 1. 准备保存路径 ---
    output_dir = 'results_hw8'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. 加载数据与模型 ---
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"【错误】找不到图片: {image_path}")
        return

    # 加载 OpenCV 自带的 Haar 级联分类器 (Viola-Jones 核心模型)
    # cv2.data.haarcascades 会自动定位到 XML 文件的安装路径
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("【错误】无法加载级联分类器 XML 文件")
        return

    # 读取图像并转灰度 (VJ算法基于灰度图特征)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("正在进行多尺度级联扫描...")

    # --- 3. 执行检测 (对应伪代码 Viola_Jones_Face_Detection) ---
    # scaleFactor=1.1: 每次扫描窗口放大 10% (构建图像金字塔)
    # minNeighbors=5:  至少要有 5 个重叠框才判定为人脸 (非极大值抑制)
    # minSize: 最小人脸尺寸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"检测完成！共发现 {len(faces)} 张人脸。")

    # --- 4. 绘制结果 ---
    # 在原图上画矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- 5. 保存与显示 ---
    # 保存结果图
    save_path = os.path.join(output_dir, 'face_detection_result.png')
    cv2.imwrite(save_path, img)
    print(f"结果已保存至: {save_path}")

    # 显示 (转为RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Viola-Jones Detection: {len(faces)} Faces Found')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 替换成你的照片文件名
    run_viola_jones_detection('test_face.jpg')