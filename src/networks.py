# src/networks.py

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels=1, img_size=28, embed_size=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size

        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, embed_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.init_size = self.img_size // 4 # For 28x28, this is 7

        self.main = nn.Sequential(
            nn.Linear(latent_dim + embed_size, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.ReLU(True),
            nn.Unflatten(1, (256, self.init_size, self.init_size)), # Reshape to (batch_size, 256, 7, 7)

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, img_channels, kernel_size=3, padding=1), # Ensure output is 28x28
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels)
        # Squeeze noise if it's 4D (N, Z, 1, 1) and your linear layer expects 2D
        # Or, adjust linear layer if it expects 4D, but that's less common for initial linear layer.
        # Given your noise is (batch_size, latent_dim, 1, 1), you need to squeeze.
        gen_input = torch.cat((noise.squeeze(), label_embed), 1)

        img = self.main(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels=1, img_size=28): # Removed embed_size from here if not used
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size

        # Discriminator's main convolutional block (now correctly named and registered)
        self.conv_block = nn.Sequential(
            # 将 spectral_norm 应用于卷积层
            spectral_norm(nn.Conv2d(self.img_channels + self.num_classes, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            # 也应用于全连接层
            spectral_norm(nn.Linear(256 * 3 * 3, 1)),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Create a label map by expanding the one-hot encoded labels
        # labels: (batch_size, num_classes)
        labels_reshaped = labels.view(labels.size(0), self.num_classes, 1, 1) # (N, 10, 1, 1)
        labels_map = labels_reshaped.expand(-1, -1, self.img_size, self.img_size) # (N, 10, 28, 28)
        
        # Concatenate the image and the label map along the channel dimension
        concat_input = torch.cat([img, labels_map], 1) # (N, 1 + 10, 28, 28)

        # Pass the concatenated input through the convolutional block
        output = self.conv_block(concat_input)
        return output