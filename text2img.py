import torch
from torchvision.utils import save_image
from transformers import RobertaTokenizer, RobertaModel

from src.generator import Generator
from src.embeddings import roberta_embed
import os
import open_clip
from open_clip import tokenize

def prompt2img():

    os.makedirs('./text2img_results', exist_ok=True)

    # 1) device and hyper-params
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 150
    embed_dim  = 512   # for roberta-base
    model_path = "./models/generator_280.pth"

    # 2) load Generator
    G = Generator(latent_dim, embed_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    # 3) load and freeze text encoder
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # text_model = RobertaModel.from_pretrained("roberta-base")\
    #                      .to(device).eval()
    # for p in text_model.parameters(): p.requires_grad_(False)

    # 3) load and freeze CLIP text encoder
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_tok = open_clip.get_tokenizer("ViT-B-32")

    # 4) ask the user for their prompt
    prompt = input("Enter your image prompt: ")

    # 5) get embedding
    # embed = roberta_embed(
    #     prompt,
    #     tokenizer,
    #     text_model,
    #     device,
    #     pooling="mean"
    # ).to(device)
    tokens = tokenize([prompt]).to(device)
    with torch.no_grad():
        embed = clip_model.encode_text(tokens)  # → (1, 512)
    embed = embed.to(device)

    # 6) sample noise + generate
    z = torch.randn(1, latent_dim, device=device)

    with torch.no_grad():
        fake = G(z, embed)  # both are 4-D now

    out = (fake + 1) * 0.5  # [-1,1] → [0,1]

    # 7) save result
    save_image(out, './text2img_results/text2img.png', nrow=1)
    print("Image saved")

if __name__ == "__main__":
    prompt2img()
