import torch
import torch.nn as nn

from src.discriminator import SelfAttention

class Generator(nn.Module):
    """
    生成器网络，其结构与提供的 Discriminator 对称。
    它接收一个噪声向量和一个文本嵌入，生成一个 64x64 的彩色图像。
    """
    def __init__(self, noise_dim, embedding_dim, image_channels=1, base_channels=64):
        super(Generator, self).__init__()
        # fashion minist
        # project (noise + text) → 7×7
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + embedding_dim,
                               base_channels * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True)
        )
        # upsample 7→14
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True)
        )
        # upsample 14→28
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, image_channels,
                               4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, embeddings):
        # fashion minist
        x = torch.cat([noise, embeddings], dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1)  # → (B, noise+embed, 1,1)
        x = self.initial(x)  # → (B, 4*base, 7,7)
        x = self.up1(x)  # → (B, 2*base,14,14)
        x = self.up2(x)  # → (B,   1  ,28,28)
        
        return x

# 验证 Generator 的独立运行
if __name__ == '__main__':
    print("--- 开始独立验证 Generator ---")
    
    # 假设的参数 (应与最终训练脚本保持一致)
    BATCH_SIZE = 4
    NOISE_DIM = 100
    # 从embedding README中得知，roberta 是 768，CLIP 是 512
    # 以 CLIP 为例
    EMBEDDING_DIM = 512 
    IMG_CHANNELS = 3
    BASE_CHANNELS = 64
    
    # 1. 创建模拟输入
    noise = torch.randn(BATCH_SIZE, NOISE_DIM)
    embeddings = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    print(f"创建模拟输入: 噪声形状 {noise.shape}, 嵌入形状 {embeddings.shape}")

    # 2. 实例化生成器
    gen = Generator(noise_dim=NOISE_DIM, embedding_dim=EMBEDDING_DIM)
    print("成功实例化 Generator 模型。")
    
    # 3. 执行前向传播
    try:
        fake_images = gen(noise, embeddings)
        print("前向传播成功！")

        # 4. 验证输出形状
        output_shape = fake_images.shape
        expected_shape = (BATCH_SIZE, IMG_CHANNELS, 64, 64)
        print(f"生成器输出形状: {output_shape}")
        print(f"期望输出形状:    {expected_shape}")

        assert output_shape == expected_shape
        print("\n[成功] 输出形状与期望一致")

    except Exception as e:
        print(f"\n[失败] 发生错误: {e}")