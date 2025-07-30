# main.py

import torch
import os
from src.train import train


def main():
    # 超参数设置
    epochs = 200          # 训练的总epoch数
    batch_size = 16       # 批次大小
    latent_dim = 150      # 噪声向量维度
    lr_D = 0.0001           # 学习率
    lr_G = 0.0004


    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确保保存目录存在
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # parquet = './data/roberta-base_train_caps.parquet'
    parquet = './data/CLIP_train_caps.parquet'
    img_dir = './data/train_25k'
    # embed_dim = 768
    embed_dim = 512

    # 运行训练
    train(
        parquet, img_dir,
        epochs, batch_size, lr_D, lr_G, latent_dim,
        embed_dim
    )

if __name__ == "__main__":
    main()