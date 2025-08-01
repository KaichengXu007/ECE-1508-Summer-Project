# src/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, image_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.image_channels = image_channels
        self.base_channels = base_channels

        # 图像处理路径，与之前保持一致
        self.image_conv_blocks = nn.Sequential(
            # 256→128
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # 128→64
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # 64→32
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # 32→16
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 在 16x16 特征图上应用自注意力
            SelfAttention(base_channels * 2),
            # 16→8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8→4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- 以下是新的投影判别器实现 ---

        # 用于处理图像特征的最后一个卷积层，将 4x4 的特征图变为 1x1
        self.final_image_conv = nn.Conv2d(base_channels * 8, base_channels * 8, 4, 1, 0)

        # 用于将文本嵌入投影到与图像特征相同的维度
        self.embedding_projection = nn.Linear(embedding_dim, base_channels * 8)

        # 用于从图像特征中计算无条件的“真实性”分数
        self.final_classifier = nn.Linear(base_channels * 8, 1)

    def forward(self, images, embeddings):
        # 1. 处理图像特征
        image_features = self.image_conv_blocks(images) #
        image_features = self.final_image_conv(image_features) # -> [B, C, 1, 1]

        # 2. 移除空间维度，得到图像特征向量
        image_features = image_features.squeeze(-1).squeeze(-1) # -> [B, C]

        # --- 核心投影逻辑 ---
        # 3. 计算无条件的“真实性”分数 (只看图)
        unconditional_score = self.final_classifier(image_features)

        # 4. 计算条件匹配分数 (图与文本的点积)
        projected_embedding = self.embedding_projection(embeddings) # -> [B, C]
        
        # 使用 einsum 高效计算批量点积，判断图像特征和文本嵌入的相似度
        conditional_score = torch.einsum('bc,bc->b', image_features, projected_embedding)
        conditional_score = conditional_score.unsqueeze(1) # -> [B, 1]

        # 5. 最终输出是无条件分数和条件分数的总和
        # 这个输出值（logit）将直接送入 BCEWithLogitsLoss
        output = unconditional_score + conditional_score
        return output