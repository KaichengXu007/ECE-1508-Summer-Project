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
        # Reshape for matrix multiplication
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Weighted value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, image_channels=1, base_channels=64):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.image_channels = image_channels
        self.base_channels = base_channels

        # fashion minist
        # 28→14
        self.conv1 = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 14→7
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 7×7 → 1×1
        self.final_img = nn.Conv2d(base_channels * 2, base_channels * 2, 7, 1, 0)
        # Unconditional real/fake score
        self.uncond = nn.Linear(base_channels * 2, 1)
        # Project text to match image feature dim
        self.proj_text = nn.Linear(embedding_dim, base_channels * 2)

        # Image processing path
        # self.image_conv_blocks = nn.Sequential(
        #     # 256→128
        #     nn.Conv2d(image_channels, base_channels, 4, 2, 1),
        #     nn.BatchNorm2d(base_channels),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # 128→64
        #     nn.Conv2d(base_channels, base_channels, 4, 2, 1),
        #     nn.BatchNorm2d(base_channels),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # 64→32
        #     nn.Conv2d(base_channels, base_channels, 4, 2, 1),
        #     nn.BatchNorm2d(base_channels),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # 32→16
        #     nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
        #     nn.BatchNorm2d(base_channels * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # Self-Attention Layer at 16*16
        #     SelfAttention(base_channels * 2),
        #
        #     # 16→8
        #     nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
        #     nn.BatchNorm2d(base_channels * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # Fourth Conv Block (before combining with embedding)
        #     nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
        #     nn.BatchNorm2d(base_channels * 8),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # # 用于处理图像特征的最后一个卷积层，将 4x4 的特征图变为 1x1
        # self.final_image_conv = nn.Conv2d(base_channels * 8, base_channels * 8, 4, 1, 0)
        #
        # # 用于将文本嵌入投影到与图像特征相同的维度
        # self.embedding_projection = nn.Linear(embedding_dim, base_channels * 8)
        #
        # # 用于从图像特征中计算无条件的“真实性”分数
        # self.final_classifier = nn.Linear(base_channels * 8, 1)
        # Embedding processing path
        # Project embedding to a spatial size and number of channels to combine with image features
        # Assuming image features before combining are (batch_size, base_channels * 8, 4, 4)
        # self.embedding_projection = nn.Sequential(
        #     nn.Linear(embedding_dim, base_channels * 8 * 4 * 4), # Project to match spatial feature size
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        #
        #
        # # Final layer(s) after combining image and embedding features
        # # Assuming concatenation of flattened image features and projected embedding
        # # Flattened image features: (batch_size, base_channels * 8 * 8 * 8) based on 8x8 spatial output
        # # Projected embedding: (batch_size, base_channels * 8 * 4 * 4) based on 4x4 spatial target
        # # Combined flattened features size: (base_channels * 8 * 4 * 4) + (base_channels * 8 * 4 * 4) = 8192 + 8192 = 16384
        #
        # with torch.no_grad():
        #     img_dummy = torch.zeros(1, image_channels, 256, 256)
        #     emb_dummy = torch.zeros(1, embedding_dim)
        #
        #     img_n = self.image_conv_blocks(img_dummy).view(1, -1).size(1)
        #     emb_n = self.embedding_projection(emb_dummy).view(1, -1).size(1)
        #     final_in = img_n + emb_n
        #
        # self.final_layer = nn.Sequential(
        #     nn.Linear(final_in, 1), # Corrected input size to match concatenated features
        #     nn.Sigmoid() # Output probability
        # )

    def forward(self, images, embeddings):
        # # Process images
        # image_features = self.image_conv_blocks(images)
        # image_features_flattened = image_features.view(images.size(0), -1) # Flatten image features
        # # print(f"Shape of image_features_flattened: {image_features_flattened.shape}") # Add this print
        #
        # # Process embeddings
        # embedding_features = self.embedding_projection(embeddings)
        # embedding_features_flattened = embedding_features.view(embeddings.size(0), -1) # Flatten projected embeddings
        # # print(f"Shape of embedding_features_flattened: {embedding_features_flattened.shape}") # Add this print
        #
        # # Combine features
        # combined_features = torch.cat([image_features_flattened, embedding_features_flattened], dim=1)
        # # print(f"Shape of combined_features: {combined_features.shape}") # Add this print
        #
        # # Final prediction
        # output = self.final_layer(combined_features)

        # 1. 处理图像特征
        # image_features = self.image_conv_blocks(images)  #
        # image_features = self.final_image_conv(image_features)  # -> [B, C, 1, 1]
        #
        # # 2. 移除空间维度，得到图像特征向量
        # image_features = image_features.squeeze(-1).squeeze(-1)  # -> [B, C]
        #
        # # --- 核心投影逻辑 ---
        # # 3. 计算无条件的“真实性”分数 (只看图)
        # unconditional_score = self.final_classifier(image_features)
        #
        # # 4. 计算条件匹配分数 (图与文本的点积)
        # projected_embedding = self.embedding_projection(embeddings)  # -> [B, C]
        #
        # # 使用 einsum 高效计算批量点积，判断图像特征和文本嵌入的相似度
        # conditional_score = torch.einsum('bc,bc->b', image_features, projected_embedding)
        # conditional_score = conditional_score.unsqueeze(1)  # -> [B, 1]
        #
        # # 5. 最终输出是无条件分数和条件分数的总和
        # # 这个输出值（logit）将直接送入 BCEWithLogitsLoss
        # output = unconditional_score + conditional_score

        x = self.conv1(images)  # → (B, base,14,14)
        x = self.conv2(x)  # → (B,2*base,7,7)
        x = self.final_img(x).view(images.size(0), -1)  # → (B,2*base)
        real_score = self.uncond(x)  # → (B,1)
        txt_score = (self.proj_text(embeddings) * x).sum(-1, keepdim=True)

        return real_score + txt_score
