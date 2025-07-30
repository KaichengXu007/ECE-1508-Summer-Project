import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from src.dataloader import get_dataloader
from src.generator import Generator
from src.discriminator import Discriminator
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.cuda.amp import autocast
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF


def save_real_fake_grid(real_imgs, fake_imgs, captions, path, n=8):
    """
    real_imgs, fake_imgs : tensors in (-1,1)  shape (B,3,H,W)
    captions            : list[str]
    path                : png path
    """
    real = (real_imgs[:n].cpu() + 1) * 0.5  # → (0,1)
    fake = (fake_imgs[:n].cpu() + 1) * 0.5

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


def clip_score(images, captions):
    """
    images: (B,3,H,W) in (-1,1)
    captions: list[str]
    return: mean cosine similarity (scalar)
    """
    with torch.no_grad(), autocast():
        imgs_norm = (images + 1) * 0.5  # (0,1)
        img_inputs = clip_preprocess(imgs_norm).to(device)  # 224×224
        img_feats = clip_model.encode_image(img_inputs)

        text_tokens = tokenizer(captions).to(device)
        text_feats = clip_model.encode_text(text_tokens)

        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return (img_feats * text_feats).sum(dim=-1).mean()  # scalar tensor


# Training function
def train(parquet, img_dir, epochs, batch_size, lr, latent_dim, embed_dim):
    # dataloader
    loader = get_dataloader(parquet, img_dir, batch_size)
    # networks
    G = Generator(latent_dim, embed_dim).to(device)
    D = Discriminator(embed_dim).to(device)
    # opt & loss
    opt_G = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # fid metric
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
        model_name='ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    print('Start training...')
    for e in range(1, epochs + 1):

        G.train()
        D.train()
        epoch_loss_G, epoch_loss_D, n_batches = 0., 0., 0

        fid_metric.reset()
        clip_scores_epoch = []

        for real_imgs, embeds, captions in loader:
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

            opt_D.zero_grad()  # ★ 改这里
            loss_D.backward()
            opt_D.step()

            # Generator update
            pred = D(fake_imgs, embeds)
            loss_G = criterion(pred, real_label)

            opt_G.zero_grad()  # ★ 改这里
            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

            # FID 累计
            fid_metric.update((real_imgs + 1) * 0.5, real=True)  # (0,1)
            fid_metric.update((fake_imgs + 1) * 0.5, real=False)

            # CLIP
            clip_scores_epoch.append(clip_score(fake_imgs, captions).item())

            n_batches += 1

        # print(f"Epoch {e}/{epochs} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

        epoch_loss_D /= n_batches
        epoch_loss_G /= n_batches

        # Save checkpoints every 10 epochs
        if e % 10 == 0 or e == 1:
            G.eval()
            fid_value = fid_metric.compute().item()
            clip_mean = sum(clip_scores_epoch) / len(clip_scores_epoch)

            print(f"[{e}/{epochs}]  "
                  f"D: {epoch_loss_D:.4f}  G: {epoch_loss_G:.4f}  "
                  f"FID: {fid_value:.2f}  CLIP: {clip_mean:.3f}")

            # Save model weights
            torch.save(G.state_dict(), f"{models_dir}/generator_{e}.pth")
            torch.save(D.state_dict(), f"{models_dir}/discriminator_{e}.pth")

            # Save sample generated images
            G.eval()
            with torch.no_grad():
                # # take first batch of embeddings for sampling
                # sample_embeds = next(iter(loader))[1][:16].to(device)
                # sample_z = torch.randn(sample_embeds.size(0), latent_dim, device=device)
                # samples = G(sample_z, sample_embeds)
                # save_image((samples + 1) * 0.5, f"{results_dir}/sample_epoch{e}.png", nrow=4)
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

    print("Training finished!")


if __name__ == '__main__':
    train(parquet="../data/roberta-base_train_caps.parquet", img_dir="../data/train_25k", epochs=50, batch_size=4,
          lr=0.0001, latent_dim=64, embed_dim=768)
