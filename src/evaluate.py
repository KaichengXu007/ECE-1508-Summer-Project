import os
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.cuda.amp import autocast

import open_clip
from src.dataloader import get_fashion_dataloader
from src.generator import Generator


def save_real_fake_grid(real_imgs, fake_imgs, captions, path, n=8, start_idx=0):
    """
    real_imgs, fake_imgs : tensors in (-1,1)  shape (B,3,H,W)
    captions            : list[str]
    path                : png path
    """
    # Select the slice of images starting from start_idx
    real = torch.clamp((real_imgs[start_idx:start_idx+n].cpu() + 1) * 0.5, 0, 1)  # → (0,1)
    fake = torch.clamp((fake_imgs[start_idx:start_idx+n].cpu() + 1) * 0.5, 0, 1)
    selected_captions = captions[start_idx:start_idx+n]

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for row, imgs, row_name in zip(range(2), [real, fake], ["real", "fake"]):
        for col in range(n):
            if col < len(imgs):  # Make sure we have enough images
                ax = axes[row, col]
                img = TF.to_pil_image(imgs[col])
                ax.imshow(img)
                ax.axis("off")
                if row == 0:
                    ax.set_title(selected_captions[col], fontsize=6, pad=2)
                if col == 0:
                    ax.set_ylabel(row_name, rotation=0, labelpad=25,
                                  fontsize=8, va="center")
            else:
                # Hide unused subplots if we don't have enough images
                axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_clip(device):
    # ckpt_path = 'data/clip_v32/open_clip_pytorch_model.bin'
    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     model_name="ViT-B-32",
    #     pretrained=str(ckpt_path)
    # )
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name='ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return model, preprocess, tokenizer


def clip_score(images, captions, clip_model, clip_preprocess,
               tokenizer, device):
    """
    images   : (B,3,H,W) in (-1,1)
    captions : list[str]
    return   : mean cosine similarity (scalar)
    """

    with torch.no_grad(), autocast():
        imgs_norm = (images + 1) * 0.5  # scale to (0,1)

        to_pil = ToPILImage()
        img_inputs = torch.stack([
            clip_preprocess(to_pil(img.cpu())).to(device)
            for img in imgs_norm
        ], dim=0)  # (B,3,224,224)

        img_feats = clip_model.encode_image(img_inputs)
        text_tokens = tokenizer(captions).to(device)
        text_feats = clip_model.encode_text(text_tokens)

        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return (img_feats * text_feats).sum(dim=-1).mean()  # scalar


def evaluate(parquet, img_dir, batch_size,
             latent_dim, embed_dim, ckpt_path,
             results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataloader
    loader = get_fashion_dataloader(parquet, img_dir,
                                    batch_size=batch_size, train=False, shuffle=False)
    print("Test dataset loaded")

    G = Generator(latent_dim, embed_dim).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    G.eval()
    print("Generator weights loaded")

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    # CLIP
    clip_model, clip_preprocess, tokenizer = build_clip(device)

    clip_scores = []

    first_batch_done = False
    grid_path = os.path.join(results_dir, "eval_real_fake.png")

    with torch.no_grad():
        for real_imgs, embeds, captions in tqdm(loader):
            real_imgs = real_imgs.to(device)
            embeds = embeds.to(device)
            bs = real_imgs.size(0)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z, embeds)

            # --- FID ---
            real_batch = (real_imgs + 1) * 0.5
            if real_batch.size(1) == 1:
                real_batch = real_batch.repeat(1, 3, 1, 1)
            fid_metric.update(real_batch, real=True)

            fake_batch = (fake_imgs + 1) * 0.5
            if fake_batch.size(1) == 1:
                fake_batch = fake_batch.repeat(1, 3, 1, 1)
            fid_metric.update(fake_batch, real=False)

            # --- CLIP ---
            c_score = clip_score(fake_imgs, captions,
                                 clip_model, clip_preprocess,
                                 tokenizer, device).item()
            clip_scores.append(c_score)

            # generate sample image
            if not first_batch_done:
                save_real_fake_grid(real_imgs, fake_imgs,
                                    captions, grid_path, n=8, start_idx=8)
                first_batch_done = True

    # 统计结果
    fid_value = fid_metric.compute().item()
    clip_mean = sum(clip_scores) / len(clip_scores)

    print(f"\n========= Evaluation Finished =========")
    print(f"FID  : {fid_value:.2f}")
    print(f"CLIP : {clip_mean:.3f}")
    print(f"Grid saved to: {grid_path}")

    # 保存 json
    with open(os.path.join(results_dir, "eval_result.json"), "w") as f:
        json.dump({"fid": fid_value,
                   "clip": clip_mean}, f, indent=2)
    print("Evaluation result saved")


if __name__ == "__main__":
    # params
    parquet = "../data/fashion_CLIP_test_caps.parquet"
    img_dir = "../data"
    batch_size = 64
    latent_dim = 150
    embed_dim = 512
    models_dir = "../models"
    results_dir = "../results"
    ckpt_path = os.path.join(models_dir, "generator_190.pth")
    # ---------------------------------------

    evaluate(parquet, img_dir, batch_size,
             latent_dim, embed_dim, ckpt_path,
             results_dir)
