# src/dataloader.py

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import os
import matplotlib.pyplot as plt

transform = Compose([
    Resize((256,256)),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)
])

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
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        embedding = torch.tensor(row['embedding'], dtype=torch.float)
        caption = row['caption']
        return image, embedding, caption

def get_dataloader(parquet_path, image_dir, batch_size=8):
    df = pd.read_parquet(parquet_path)
    dataset = TextImageDataset(df, image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# view the dataset (images, captions, embeddings)
if __name__ == '__main__':
    loader = get_dataloader("../data/roberta-base_train_caps.parquet", "../data/train_25k", batch_size=4)
    images, embeddings, captions = next(iter(loader))  # images.shape = [4,3,256,256]

    # 2) Pick the first image and undo the Normalize(mean=0.5, std=0.5)
    img = images[0]  # tensor in [-1, +1]
    img = img * 0.5 + 0.5  # now in [0,1]

    # Display with Matplotlib
    np_img = img.permute(1, 2, 0).cpu().numpy()  # HWC
    plt.imshow(np_img)
    plt.axis("off")
    plt.show()

    print("First caption:", captions[0])
    print("First embedding (first 10 dims):", embeddings[0][:10])
