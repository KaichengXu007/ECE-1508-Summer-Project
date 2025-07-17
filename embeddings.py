import json
import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import pyarrow  # used for parquet
import open_clip


@torch.no_grad()
def roberta_embed(texts, tokenizer, model, device, pooling: str = "mean", max_length: int = 64):
    """
    List[str] to Tensor[n, hidden]
    pooling: "cls" or "mean"
    """
    if isinstance(texts, str):
        texts = [texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    out = model(**enc)  # last_hidden_state [B, L, H]
    if pooling == "cls":
        vec = out.last_hidden_state[:, 0]
    elif pooling == "mean":
        mask = enc.attention_mask.unsqueeze(-1)
        vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    else:
        raise ValueError("pooling should be 'cls' or 'mean'")

    return vec.cpu()


@torch.no_grad()
def clip_sentence_embed(captions, model, tokenizer, device, batch_size=256):
    """
    captions : List[str] → Tensor [N, 512]  (CPU)
    """
    all_vecs = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        tokens = tokenizer(batch).to(device)  # [B, L]
        vec = model.encode_text(tokens).float()  # [B, 512]
        all_vecs.append(vec.cpu())
    return torch.cat(all_vecs, 0)


if __name__ == "__main__":
    # init settings
    SOURCE_JSON = 'data/annotations/captions_train2017.json'
    SOURCE_IMG_DIR = 'data/train2017'
    DEST_IMG_DIR = 'data/train_25k'

    SAMPLE_SIZE = 25000
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    with open(SOURCE_JSON, 'r') as f:
        data = json.load(f)

    # sample image
    images = data['images']
    annotations = data['annotations']

    sampled_images = random.sample(images, SAMPLE_SIZE)
    sampled_ids = {img['id'] for img in sampled_images}

    # image_id → captions
    id2captions = defaultdict(list)
    for ann in annotations:
        if ann['image_id'] in sampled_ids:
            id2captions[ann['image_id']].append(ann['caption'])

    captions = []
    file_names = []
    img_ids = []
    batch_size = 512  # 1024
    embeds = []

    for img in sampled_images:
        img_id = img['id']
        # pick first cap
        cap = id2captions[img_id][0]
        captions.append(cap)
        file_names.append(img['file_name'])
        img_ids.append(img_id)

    # load model

    MODEL_NAME = "roberta-base"
    # MODEL_NAME = "CLIP"

    if MODEL_NAME == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        model = RobertaModel.from_pretrained(MODEL_NAME)
        model.eval()  # freeze dropout / layernorm
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        for i in tqdm(range(0, len(captions), batch_size)):
            batch_caps = captions[i:i + batch_size]
            vec = roberta_embed(batch_caps, tokenizer, model, device, pooling="mean")
            embeds.append(vec)

        embeds = torch.cat(embeds).cpu().numpy()  # [N, 768] → NumPy
    else:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )
        clip_model = clip_model.to(device).eval().requires_grad_(False)

        clip_tok = open_clip.get_tokenizer("ViT-B-32")

        embeds = clip_sentence_embed(captions, clip_model, clip_tok,
                                     device=device, batch_size=batch_size)
        embeds = embeds.numpy()  # (N, 512)

    # store result

    print(len(captions), len(file_names))
    df = pd.DataFrame({
        "file_name": file_names,
        "caption": captions,
        "embedding": [vec.tolist() for vec in embeds]
    })
    df.to_parquet(f'data/{MODEL_NAME}_train_caps.parquet',
                  engine="pyarrow",
                  compression="zstd",
                  index=False)


    '''create sampled dataset, comment if don need it'''
    # os.makedirs(DEST_IMG_DIR, exist_ok=True)
    # for file_name in tqdm(df["file_name"], desc="Copying images"):
    #     src_path = os.path.join(SOURCE_IMG_DIR, file_name)
    #     dst_path = os.path.join(DEST_IMG_DIR, file_name)
    #     shutil.copyfile(src_path, dst_path)

    # read
    # df = pd.read_parquet("caps.parquet")
