import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10 

    # 数据路径
    DATA_PATH = './data'

    # 模型保存路径
    MODEL_PATH = 'best_model.pth'

    # 图像预处理均值和方差 (MNIST的标准数值)
    NORM_MEAN = (0.1307,)
    NORM_STD = (0.3081,)