# ECE-1508-Summer-Project: Text-to-Image Generation with Conditional GAN and CLIP Embeddings

This project implements a text-to-image generation pipeline using a Conditional GAN (cGAN) trained on the Fashion-MNIST dataset with synthetic captions. A frozen CLIP text encoder provides semantically rich text embeddings, enabling the generator to produce images conditioned on natural language descriptions.

### Features
- **Custom Dataset Loader** for Fashion-MNIST with precomputed CLIP embeddings and captions stored in Parquet format.
- **Generator and Projection Discriminator** architectures tailored for 28×28 grayscale images.
- **Synthetic Caption Generation** using flexible templates to improve generalization.
- **Frozen CLIP (ViT-B/32) Text Encoder** for extracting 512-dimensional sentence embeddings with strong visual grounding.
- **Training Pipeline** with periodic evaluation of Fréchet Inception Distance (FID) and CLIP score.
- **Evaluation Script** to generate real vs. fake comparison grids and compute both quantitative and qualitative metrics.
- **Utility Functions** for computing CLIP similarity between images and text prompts.
- **Plotting Tools** to visualize training curves such as loss, FID, and CLIP score over epochs.


### Embedding
1. dataset: 
   1. subsampled [ms-coco2017](https://cocodataset.org/#download) 118k to 25k
      1. train/val/test 2017 
      2. 2017 Train/Val annotations [241MB]
   2. fashion-mnist
      1. **CHANGE ```file_name``` into ```idx``` in first column !!!**
      2. caption for each idx:```"A grayscale image of a {label2text[label]}"```
2. pretrained encoder: 
   1. roberta(768)
   2. Clip ViT-B-32(512) *recommended
   3. all frozen, sentence-level
3. dataframe structure:
   ```python
   df = pd.DataFrame({
        "file_name": file_names,  # "idx" in fashion-mnist dataset
        "caption": captions,
        "embedding": [vec.tolist() for vec in embeds]
    })
   ```
   
## Project Structure

```bash
data/
    ├── annotations/
        └── captions_train2017.json
    ├── fashion_CLIP_train_caps.parquet
    └── roberta-base_train_caps.parquet
src/
    ├── dataloader.py
    ├── discriminator.py
    ├── embeddings.py
    ├── evaluate.py
    ├── generator.py
    ├── plot.py
    ├── train.py
    └── utils.py
text2img_results/
    ├── ankle_boot.png
    ├── bag.png
    ├── coat.png
    ├── dress.png
    ├── pullover.png
    ├── sandal.png
    ├── shirt.png
    ├── sneaker.png
    ├── T-shirt or top.png
    ├── T-shirt.png
    ├── text2img.png
    ├── top.png
    └── trouser.png
main.py
README.md
requirements.txt
text2img.py
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KaichengXu007/ECE-1508-Summer-Project.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

### Usage

1. Data Preparation
   ```bash
   python src/embeddings.py
2. Train the Model
   ```bash
   python main.py
3. Evaluate the Model
   ```bash
   python src/evaluate.py
4. Plot Training Curves
   ```bash
   python src/plot.py
5. Generate Images
   ```bash
   python text2img.py