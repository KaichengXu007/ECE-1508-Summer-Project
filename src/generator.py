import torch
import torch.nn as nn

from src.discriminator import SelfAttention

class Generator(nn.Module):

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
