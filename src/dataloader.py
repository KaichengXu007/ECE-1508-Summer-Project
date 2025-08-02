# src/dataloader.py
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

transform = Compose([
    Resize((256,256)),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)
])

fm_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

class FashionTextImageDataset(Dataset):
    """
    A Dataset that pairs Fashion-MNIST images with precomputed text embeddings
    and captions loaded from a parquet file.
    """
    def __init__(self,
                 root: str,
                 parquet_path: str,
                 train: bool = True,
                 transform=None):
        # Load captions & embeddings
        self.df = pd.read_parquet(parquet_path)
        # Underlying Fashion-MNIST dataset (provides images)
        self.ds = FashionMNIST(
            root=root,
            train=train,
            download=False,
            transform=None     # apply transform in __getitem__
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1) Get the parquet row first to find the correct Fashion-MNIST index
        row = self.df.iloc[idx]
        fashion_mnist_idx = row['idx']  # The actual Fashion-MNIST index
        
        # 2) Load the correct image from FashionMNIST using the true index
        image, _ = self.ds[fashion_mnist_idx]  # image is a PIL Image in mode "L"
        if self.transform:
            image = self.transform(image)   # → (1, 28, 28) tensor

        # 3) Load corresponding embedding & caption
        embedding = torch.tensor(row['embedding'], dtype=torch.float32)
        caption   = row['caption']

        return image, embedding, caption

class TextImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # build full path and open as PIL
        image_path = os.path.join(self.image_dir, row['file_name'])
        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)
        embedding = torch.tensor(row['embedding'], dtype=torch.float)
        caption = row['caption']
        return image, embedding, caption

def get_dataloader(parquet_path, image_dir, batch_size=8):
    df = pd.read_parquet(parquet_path)
    dataset = TextImageDataset(df, image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_fashion_dataloader(parquet_path: str,
                           root: str = '../data/FashionMNIST',
                           batch_size: int = 64,
                           train: bool = True,
                           shuffle: bool = True,
                           num_workers: int = 4):
    """
    Returns a DataLoader for the FashionTextImageDataset.
    """
    dataset = FashionTextImageDataset(
        root=root,
        parquet_path=parquet_path,
        train=train,
        transform=fm_transform
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)

# view the dataset (images, captions, embeddings)
if __name__ == '__main__':
    loader = get_fashion_dataloader(
        parquet_path='../data/fashion_CLIP_train_caps.parquet',
        root='../data/',
        batch_size=4
    )

    all_captions = []
    all_images = []
    MAX_ITEMS = 500

    for imgs, _, captions in loader:
        all_images.extend(imgs)  # imgs is a Tensor [B,1,28,28]
        all_captions.extend(captions)
        if len(all_captions) >= MAX_ITEMS:
            break

    # Undo Normalize(mean=0.5,std=0.5), then plot
    plt.figure(figsize=(10, 10))
    for i in range(min(16, len(all_images))):
        img = all_images[i]  # Tensor [1,28,28]
        img = img * 0.5 + 0.5  # back to [0,1]
        np_img = img.squeeze().numpy()
        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(np_img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(all_captions[i], fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
