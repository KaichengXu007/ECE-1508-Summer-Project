import os
import torch
import pandas as pd
from torchvision.transforms import ToPILImage
from dataloader import TextImageDataset
from torch.utils.data import DataLoader
from generator import Generator
from utils import compute_clip_similarity
from src.dataloader import transform


def evaluate(parquet, img_dir, num_samples):
    # 1. Load and subset the DataFrame
    df = pd.read_parquet(parquet)
    df = df.iloc[:num_samples].reset_index(drop=True)

    # 2. Create dataset and loader
    ds = TextImageDataset(df, img_dir, transform=transform)
    loader = DataLoader(ds, batch_size=1 , shuffle=False)

    # 3. Initialize generator
    G = Generator(latent_dim, embed_dim).to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()

    to_pil = ToPILImage()
    scores = []

    for i, (_, embed, caption) in enumerate(loader):
        eb = embed.to(device)
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            fake = G(z, eb).cpu()
        img = ((fake.squeeze(0)+1)*0.5).clamp(0,1)
        pil = to_pil(img)
        # pil.save(os.path.join(output_dir, f"gen_{i}.png"))
        scores.append(compute_clip_similarity(pil, caption[0]))

    print(f"Mean CLIP similarity: {torch.tensor(scores).mean().item():.4f}")

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Evaluation parameters
    latent_dim = 150
    embed_dim = 512
    num_samples = 500
    model_path = os.path.join(ROOT_DIR, 'models', 'generator_final.pth')
    output_dir = os.path.join(ROOT_DIR, 'eval_results')

    os.makedirs(output_dir, exist_ok=True)
    # parquet = '../data/roberta-base_train_caps.parquet'
    parquet = '../data/CLIP_train_caps.parquet'
    img_dir = '../data/train_25k'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(parquet, img_dir, num_samples)