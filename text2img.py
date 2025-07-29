import torch
from torchvision.utils import save_image
from transformers import RobertaTokenizer, RobertaModel

from src.generator import Generator
from src.embeddings import roberta_embed
import os

def prompt2img():

    os.makedirs('./text2img_results', exist_ok=True)

    # 1) device and hyper-params
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    embed_dim  = 768   # for roberta-base
    model_path = "./models/generator_final.pth"

    # 2) load Generator
    G = Generator(latent_dim, embed_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    # 3) load and freeze text encoder
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    text_model = RobertaModel.from_pretrained("roberta-base")\
                         .to(device).eval()
    for p in text_model.parameters(): p.requires_grad_(False)

    # 4) ask the user for their prompt instead of hard-coding it
    prompt = input("Enter your image prompt: ")

    # 5) get embedding
    embed = roberta_embed(
        prompt,
        tokenizer,
        text_model,
        device,
        pooling="mean"
    ).to(device)

    # 6) sample noise + generate
    z = torch.randn(1, latent_dim, device=device)
    embed = roberta_embed(prompt, tokenizer, text_model, device, pooling="mean").to(device)

    with torch.no_grad():
        fake = G(z, embed)  # both are 4-D now

    out = (fake + 1) * 0.5  # [-1,1] → [0,1]

    # 7) save result
    save_image(out, './text2img_results/text2img.png', nrow=1)
    print("Image saved")

if __name__ == "__main__":
    prompt2img()
