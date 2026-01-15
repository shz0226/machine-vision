import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class BoW_Image_Retrieval:
    def __init__(self, k=50):
        self.k = k  # 视觉单词数量 (聚类中心数)
        self.sift = cv2.SIFT_create()
        self.vocabulary = None
        self.image_paths = []
        self.db_histograms = []
        self.output_dir = 'results_hw7'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_features(self, img_path):
        """提取单张图像的 SIFT 特征"""
        img = cv2.imread(img_path)
        if img is None: return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return des, img

    def build_vocabulary(self, dataset_folder):
        """构建视觉词典 (对应伪代码 Build_Visual_Vocabulary)"""
        print(f"【阶段1】正在读取图片并提取 SIFT 特征...")
        descriptors_pool = []
        self.image_paths = []

        # 遍历文件夹
        valid_extensions = ['.jpg', '.png', '.jpeg']
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    path = os.path.join(root, file)
                    des, _ = self.extract_features(path)
                    if des is not None:
                        descriptors_pool.append(des)
                        self.image_paths.append(path)

        if not descriptors_pool:
            print("错误：文件夹为空或无法读取图片")
            return

        # 堆叠所有特征并聚类
        vstack_descriptors = np.vstack(descriptors_pool)
        print(f"提取到 {len(vstack_descriptors)} 个特征点，开始 K-Means 聚类 (K={self.k})...")

        # 使用 OpenCV 的 K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(
            np.float32(vstack_descriptors), self.k, None, criteria, 10, flags
        )

        self.vocabulary = centers
        print("视觉词典构建完成。")

        # 建立倒排索引/直方图 (对应伪代码 Build_Inverted_Index)
        print("【阶段2】正在计算每张图的视觉单词直方图...")
        self.db_histograms = []
        for path in self.image_paths:
            des, _ = self.extract_features(path)
            hist = self.compute_bow_histogram(des)
            self.db_histograms.append(hist)

    def compute_bow_histogram(self, descriptors):
        """将特征描述子映射为直方图"""
        if descriptors is None:
            return np.zeros((1, self.k), dtype=np.float32)

        # 计算每个描述子到最近聚类中心的距离
        distances = cdist(descriptors, self.vocabulary, 'euclidean')
        nearest_word_indices = np.argmin(distances, axis=1)

        # 统计词频
        hist, _ = np.histogram(nearest_word_indices, bins=range(self.k + 1))

        # 归一化 (TF)
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist.reshape(1, -1)

    def query(self, query_img_path, top_n=3):
        """在线检索 (对应伪代码 Online_Object_Query)"""
        print(f"【阶段3】正在检索: {query_img_path}")
        q_des, q_img = self.extract_features(query_img_path)
        if q_des is None:
            print("查询图读取失败")
            return

        q_hist = self.compute_bow_histogram(q_des)

        scores = []
        for i, db_hist in enumerate(self.db_histograms):
            # 使用直方图相关性 (Correlation) 作为相似度，值越大越好
            score = cv2.compareHist(q_hist, db_hist, cv2.HISTCMP_CORREL)
            scores.append((score, self.image_paths[i]))

        # 排序
        scores.sort(key=lambda x: x[0], reverse=True)

        # --- 可视化与保存 ---
        self.visualize_results(q_img, scores[:top_n])

    def visualize_results(self, query_img, results):
        plt.figure(figsize=(15, 5))

        # 显示查询图
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title("Query Image")
        plt.axis('off')

        # 显示匹配结果
        for i, (score, path) in enumerate(results):
            res_img = cv2.imread(path)
            plt.subplot(1, len(results) + 1, i + 2)
            plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
            # 只显示文件名，不显示长路径
            fname = os.path.basename(path)
            plt.title(f"Rank {i + 1}\nScore: {score:.2f}\n{fname}", fontsize=9)
            plt.axis('off')

        save_path = os.path.join(self.output_dir, 'retrieval_result.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"【保存】检索结果已保存至: {save_path}")
        plt.show()


if __name__ == '__main__':
    # 1. 初始化系统
    # 建议准备 3 类物体（如 飞机、吉他、足球），每类 5 张图放在 dataset 文件夹里
    bow = BoW_Image_Retrieval(k=50)  # K值也就是视觉单词的数量

    # 2. 检查数据集是否存在
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"请先在当前目录下新建一个 '{dataset_dir}' 文件夹，并放入十几张测试图片！")
    else:
        # 3. 构建词典
        bow.build_vocabulary(dataset_dir)

        # 4. 执行查询 (请替换为你的一张测试图路径，最好不在 dataset 里)
        # 假设你有一张名为 'test_query.jpg' 的图片
        query_image = 'test_query.jpg'

        if os.path.exists(query_image):
            bow.query(query_image, top_n=3)
        else:
            print(f"找不到查询图: {query_image}，请指定一张存在的图片路径。")