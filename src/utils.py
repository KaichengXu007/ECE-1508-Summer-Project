import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_similarity(image: Image.Image, caption: str) -> float:
    img_in = clip_preprocess(image).unsqueeze(0).to(device)
    txt_in = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        img_f = clip_model.encode_image(img_in)
        txt_f = clip_model.encode_text(txt_in)
    return torch.cosine_similarity(img_f, txt_f).item()