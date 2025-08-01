# src/dataloader.py

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
        # 1) Load image from FashionMNIST
        image, _ = self.ds[idx]             # image is a PIL Image in mode "L"
        if self.transform:
            image = self.transform(image)   # → (1, 28, 28) tensor

        # 2) Load corresponding embedding & caption
        row = self.df.iloc[idx]
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
                           num_workers: int = 12):
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
        root='../data/FashionMNIST',
        batch_size=4
    )
    images, embeddings, captions = next(iter(loader))  # images.shape = [4,3,256,256]

    # 2) Pick the first image and undo the Normalize(mean=0.5, std=0.5)
    img = images[0]  # tensor shape (1, 28, 28), values in [-1, +1]
    img = img * 0.5 + 0.5  # now in [0, 1]

    # Display with Matplotlib
    np_img = img.permute(1, 2, 0).cpu().numpy()  # HWC
    plt.imshow(np_img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.axis("off")
    plt.show()

    print("First caption:", captions[0])
    print("First embedding (first 10 dims):", embeddings[0][:10])
