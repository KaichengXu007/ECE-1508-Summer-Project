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
    def __init__(self, embedding_dim, image_channels=1, base_channels=32):
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

    def forward(self, images, embeddings):

        x = self.conv1(images)  # → (B, base,14,14)
        x = self.conv2(x)  # → (B,2*base,7,7)
        x = self.final_img(x).view(images.size(0), -1)  # → (B,2*base)
        real_score = self.uncond(x)  # → (B,1)
        txt_score = (self.proj_text(embeddings) * x).sum(-1, keepdim=True)

        return real_score + txt_score
