import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from src.dataloader import get_dataloader, get_fashion_dataloader
from src.generator import Generator
from src.discriminator import Discriminator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for results & models
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

# Training function
def train(parquet, img_dir, epochs, batch_size, lr_D, lr_G, latent_dim, embed_dim):
    loader = get_fashion_dataloader(parquet, img_dir, batch_size)
    G = Generator(latent_dim, embed_dim).to(device)
    D = Discriminator(embed_dim).to(device)
    opt_G = Adam(G.parameters(), lr=lr_G, betas=(0.5,0.999))
    opt_D = Adam(D.parameters(), lr=lr_D, betas=(0.5,0.999))
    criterion = nn.BCEWithLogitsLoss()

    # 创建固定的噪声和嵌入，用于可视化训练过程中的效果演变
    fixed_noise = torch.randn(16, latent_dim, device=device)
    fixed_embeds_batch = next(iter(loader))
    fixed_embeds = fixed_embeds_batch[1][:16].to(device)

    for e in range(1, epochs+1):
        start = time.perf_counter()
        for imgs, embeds, _ in loader:
            imgs, embeds = imgs.to(device), embeds.to(device)
            bs = imgs.size(0)
            real = torch.ones(bs,1,device=device)
            fake = torch.zeros(bs,1,device=device)

            # Discriminator update
            z = torch.randn(bs, latent_dim, device=device)
            f_imgs = G(z, embeds)
            d_real = D(imgs, embeds)
            d_fake = D(f_imgs.detach(), embeds)
            loss_D = criterion(d_real, real) + criterion(d_fake, fake)
            D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator update
            pred = D(f_imgs, embeds)
            loss_G = criterion(pred, real)
            G.zero_grad(); loss_G.backward(); opt_G.step()
        end = time.perf_counter()
        epoch_time = end - start

        mins, secs = divmod(epoch_time, 60)

        print(f"Epoch {e}/{epochs} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f} | Time: {int(mins)}m{int(secs)}s")

        # Save checkpoints every 10 epochs
        if e % 10 == 0:
            # Save model weights
            torch.save(G.state_dict(), f"{models_dir}/generator_{e}.pth")
            torch.save(D.state_dict(), f"{models_dir}/discriminator_{e}.pth")
            # Save sample generated images
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_embeds)
                save_image((samples + 1) * 0.5, f"{results_dir}/sample_epoch{e}.png", nrow=4)
            # with torch.no_grad():
            #     # take first batch of embeddings for sampling
            #     sample_embeds = next(iter(loader))[1][:16].to(device)
            #     sample_z = torch.randn(sample_embeds.size(0), latent_dim, device=device)
            #     samples = G(sample_z, sample_embeds)
            #     save_image((samples + 1) * 0.5, f"{results_dir}/sample_epoch{e}.png", nrow=4)
            G.train()

    # Save final models after training
    torch.save(G.state_dict(), f"{models_dir}/generator_final.pth")
    torch.save(D.state_dict(), f"{models_dir}/discriminator_final.pth")


    print("Training finished!")