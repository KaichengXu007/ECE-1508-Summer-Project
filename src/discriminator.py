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
    def __init__(self, embedding_dim, image_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.image_channels = image_channels
        self.base_channels = base_channels

        # Image processing path
        self.image_conv_blocks = nn.Sequential(
            # First Conv Block
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Second Conv Block
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Self-Attention Layer
            SelfAttention(base_channels * 2),

            # Third Conv Block
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth Conv Block (before combining with embedding)
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Embedding processing path
        # Project embedding to a spatial size and number of channels to combine with image features
        # Assuming image features before combining are (batch_size, base_channels * 8, 4, 4)
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, base_channels * 8 * 4 * 4), # Project to match spatial feature size
            nn.LeakyReLU(0.2, inplace=True)
        )


        # Final layer(s) after combining image and embedding features
        # Assuming concatenation of flattened image features and projected embedding
        # Flattened image features: (batch_size, base_channels * 8 * 8 * 8) based on 8x8 spatial output
        # Projected embedding: (batch_size, base_channels * 8 * 4 * 4) based on 4x4 spatial target
        # Combined flattened features size: (base_channels * 8 * 8 * 8) + (base_channels * 8 * 4 * 4) = 32768 + 8192 = 40960

        self.final_layer = nn.Sequential(
            nn.Linear(40960, 1), # Corrected input size to match concatenated features
            nn.Sigmoid() # Output probability
        )

    def forward(self, images, embeddings):
        # Process images
        image_features = self.image_conv_blocks(images)
        image_features_flattened = image_features.view(images.size(0), -1) # Flatten image features
        # print(f"Shape of image_features_flattened: {image_features_flattened.shape}") # Add this print

        # Process embeddings
        embedding_features = self.embedding_projection(embeddings)
        embedding_features_flattened = embedding_features.view(embeddings.size(0), -1) # Flatten projected embeddings
        # print(f"Shape of embedding_features_flattened: {embedding_features_flattened.shape}") # Add this print

        # Combine features
        combined_features = torch.cat([image_features_flattened, embedding_features_flattened], dim=1)
        # print(f"Shape of combined_features: {combined_features.shape}") # Add this print

        # Final prediction
        output = self.final_layer(combined_features)

        return output
