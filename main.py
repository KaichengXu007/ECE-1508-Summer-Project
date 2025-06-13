# main.py

import torch
import os
from src.train import train
from src.utils import FASHION_MNIST_CLASSES # 导入类别名称，用于train.py内部访问

def main():
    # 超参数设置
    epochs = 50           # 训练的总epoch数
    batch_size = 128      # 批次大小
    latent_dim = 100      # 噪声向量维度
    lr = 0.0002           # 学习率
    beta1 = 0.5           # Adam优化器的beta1参数 (通常GANs使用0.5)
    save_interval = 10    # 每隔多少个epoch保存一次模型和生成图片

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确保保存目录存在
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # 运行训练
    train(epochs, batch_size, latent_dim, lr, beta1, device, save_interval)

if __name__ == "__main__":
    main()