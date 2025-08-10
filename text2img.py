import torch
from torchvision.utils import save_image
from transformers import RobertaTokenizer, RobertaModel

from src.generator import Generator
from src.embeddings import roberta_embed
import os
import open_clip
from open_clip import tokenize

def load_models(model_path="./models/generator_190.pth", device=None):
    """
    Load the generator and CLIP models
    
    Returns:
        generator: Loaded generator model
        clip_model: Loaded CLIP text encoder
        device: Device being used
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    latent_dim = 150
    embed_dim = 512
    
    # Load Generator
    generator = Generator(latent_dim, embed_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Load CLIP text encoder
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    return generator, clip_model, device

def generate_from_prompts(prompts, generator, clip_model, device, latent_dim=150):
    """
    Generate images from text prompts
    
    Args:
        prompts: List of text prompts
        generator: Loaded generator model
        clip_model: Loaded CLIP model
        device: Device to use
        latent_dim: Latent dimension for noise
        
    Returns:
        generated_images: Tensor of generated images [0,1]
    """
    # Get embeddings
    tokens = tokenize(prompts).to(device)
    with torch.no_grad():
        embed = clip_model.encode_text(tokens)
    embed = embed.to(device)
    
    # Generate images
    batch_size = embed.size(0)
    z = torch.randn(batch_size, latent_dim, device=device)
    
    with torch.no_grad():
        fake = generator(z, embed)
    
    # Convert from [-1,1] to [0,1]
    out = (fake + 1) * 0.5
    
    return out

def prompt2img():
    """
    Original function for backward compatibility
    """
    os.makedirs('./text2img_results', exist_ok=True)
    
    # Load models
    generator, clip_model, device = load_models()
    
    # Generate image
    prompts = ["A picture of a shirt"]
    out = generate_from_prompts(prompts, generator, clip_model, device)
    
    # Save result
    save_image(
        out,
        'text2img_results/shirt_new_prompt.png',
        nrow=len(prompts),
        normalize=False
    )
    print("Image saved")

if __name__ == "__main__":
    prompt2img()
