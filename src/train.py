import os

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from src.dataloader import get_dataloader, get_fashion_dataloader
from src.generator import Generator
from src.discriminator import Discriminator
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.cuda.amp import autocast
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
import time
import json

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for results & models
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

# load clip model
clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
    model_name='ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def save_real_fake_grid(real_imgs, fake_imgs, captions, path, n=8):
    """
    real_imgs, fake_imgs : tensors in (-1,1)  shape (B,3,H,W)
    captions            : list[str]
    path                : png path
    """
    real = torch.clamp((real_imgs[:n].cpu() + 1) * 0.5, 0, 1)  # → (0,1) with clamping
    fake = torch.clamp((fake_imgs[:n].cpu() + 1) * 0.5, 0, 1)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for row, imgs, row_name in zip(range(2), [real, fake], ["real", "fake"]):
        for col in range(n):
            ax = axes[row, col]
            img = TF.to_pil_image(imgs[col])
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(captions[col], fontsize=6, pad=2)
            if col == 0:
                ax.set_ylabel(row_name, rotation=0, labelpad=25, fontsize=8, va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def clip_score(images, captions):
    """
    images: (B,3,H,W) in (-1,1)
    captions: list[str]
    return: mean cosine similarity (scalar)
    """
    with torch.no_grad(), autocast():
        imgs_norm = (images + 1) * 0.5  # (0,1)

        # convert each (1,H,W) tensor → PIL “L” → CLIP preprocess (RGB,224×224)
        to_pil = ToPILImage()
        img_inputs = torch.stack([
            clip_preprocess(to_pil(img.cpu())).to(device)
            for img in imgs_norm
        ], dim=0)  # → (B, 3, 224, 224)

        img_feats = clip_model.encode_image(img_inputs)

        text_tokens = tokenizer(captions).to(device)
        text_feats = clip_model.encode_text(text_tokens)

        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return (img_feats * text_feats).sum(dim=-1).mean()  # scalar tensor

def weights_init(m):
    """Initialize weights using DCGAN initialization scheme"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Training function
def train(parquet, img_dir, epochs, batch_size, lr_D, lr_G, latent_dim, embed_dim):
    loader = get_fashion_dataloader(parquet, img_dir, batch_size)
    G = Generator(latent_dim, embed_dim).to(device)
    D = Discriminator(embed_dim).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    opt_G = Adam(G.parameters(), lr=lr_G, betas=(0.5,0.999))
    opt_D = Adam(D.parameters(), lr=lr_D, betas=(0.5,0.999))
    criterion = nn.BCEWithLogitsLoss()

    # 创建固定的噪声和嵌入，用于可视化训练过程中的效果演变
    fixed_noise = torch.randn(16, latent_dim, device=device)
    temp_batch = next(iter(loader))
    fixed_embeds = temp_batch[1][:16].to(device)

    # fid metric
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    print('Start training...')

    start_time = time.time()
    epoch_loss_D_list = []
    epoch_loss_G_list = []
    clip_list = []
    fid_list = []
    time_list =[]

    for e in range(1, epochs + 1):
        start_time = time.time()
        G.train()
        D.train()
        epoch_loss_G, epoch_loss_D, n_batches = 0., 0., 0

        clip_scores_epoch = []
        fid_scores_epoch = []
        fid_metric.reset()

        for real_imgs, embeds, captions in tqdm(loader):
            real_imgs, embeds = real_imgs.to(device), embeds.to(device)
            bs = real_imgs.size(0)
            real_label = torch.ones(bs, 1, device=device)
            fake_label = torch.zeros(bs, 1, device=device)

            # Discriminator update
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z, embeds)
            d_real = D(real_imgs, embeds)
            d_fake = D(fake_imgs.detach(), embeds)
            loss_D_real = criterion(d_real, real_label)
            loss_D_fake = criterion(d_fake, fake_label)
            loss_D = (loss_D_real + loss_D_fake) * 0.5  # smooth

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Generator update
            pred = D(fake_imgs, embeds)
            loss_G = criterion(pred, real_label)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

            # FID 累计
            real_batch = (real_imgs + 1) * 0.5  # scale to [0,1]
            if real_batch.size(1) == 1:
                real_batch = real_batch.repeat(1, 3, 1, 1)  # (B,1,H,W)→(B,3,H,W)
            fid_metric.update(real_batch, real=True)

            fake_batch = (fake_imgs + 1) * 0.5  # scale to [0,1]
            if fake_batch.size(1) == 1:
                fake_batch = fake_batch.repeat(1, 3, 1, 1)  # (B,1,H,W)→(B,3,H,W)
            fid_metric.update(fake_batch, real=False)

            # CLIP
            c_score = clip_score(fake_imgs, captions).item()
            clip_scores_epoch.append(c_score)

            n_batches += 1

        epoch_loss_D /= n_batches
        epoch_loss_G /= n_batches
        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)

        tqdm.write(f"Epoch {e}/{epochs} | D: {epoch_loss_D:.4f} | G: {epoch_loss_G:.4f} | Time: {mins}m{secs}s")

        epoch_loss_D_list.append(epoch_loss_D)
        epoch_loss_G_list.append(epoch_loss_G)
        time_list.append(elapsed_time)

        # Save checkpoints every 10 epochs
        if e % 10 == 0 or e == 1:
            fid_value = fid_metric.compute().item()
            clip_mean = sum(clip_scores_epoch) / len(clip_scores_epoch)

            tqdm.write(f"---------FID: {fid_value:.2f}  CLIP: {clip_mean:.3f} ---------")
            fid_list.append(fid_value)
            clip_list.append(clip_mean)

            # Save model weights
            torch.save(G.state_dict(), f"{models_dir}/generator_{e}.pth")
            torch.save(D.state_dict(), f"{models_dir}/discriminator_{e}.pth")

            # Save sample generated images
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_embeds)
                save_image((samples + 1) * 0.5, f"{results_dir}/sample_epoch{e}.png", nrow=4)

                real_batch, embed_batch, cap_batch = next(iter(loader))
                real_batch = real_batch.to(device)
                embed_batch = embed_batch.to(device)

                n = 8
                z = torch.randn(n, latent_dim, device=device)
                fake_batch = G(z, embed_batch[:n])

                grid_path = f"results/real_fake_epoch{e}.png"
                save_real_fake_grid(real_batch, fake_batch, cap_batch, grid_path, n=n)

            G.train()

    # Save final models after training
    torch.save(G.state_dict(), f"{models_dir}/generator_final.pth")
    torch.save(D.state_dict(), f"{models_dir}/discriminator_final.pth")

    data_dict = {
        'epoch_loss_D': epoch_loss_D_list,
        'epoch_loss_G': epoch_loss_G_list,
        'clip': clip_list,
        'fid': fid_list,
        'time': time_list,
    }

    # save training process
    with open('results/result.json', 'w') as f:
        json.dump(data_dict, f)
    print('result saved')

    print("Training finished!")


