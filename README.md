# ECE-1508-Summer-Project

### embedding
1. dataset: subsampled [ms-coco2017](https://cocodataset.org/#download) 118k to 25k
   1. train/val/test 2017 
   2. 2017 Train/Val annotations [241MB]
2. pretrained encoder: 
   1. roberta(768)
   2. Clip ViT-B-32(512) 
   3. all frozen, sentence-level
3. dataframe structure:
   ```python
   df = pd.DataFrame({
        "file_name": file_names,
        "caption": captions,
        "embedding": [vec.tolist() for vec in embeds]
    })
   ```
   
## Project Structure

```bash
ECE-1508-Summer-Project/
│
├── data/
│   │
│   ├── annotations/
│   │   └── captions_train2017.json
│   │
│   ├── train2017/
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   │   └── ... (其他 MS-COCO 原始图片)
│   │
│   └── roberta-base_train_caps.parquet  (由 embeddings.py 生成的数据文件)
│
├── src/
│   │
│   ├── __init__.py
│   ├── dataset.py          # (推荐) 将数据加载类从 train.py 中移到这里
│   ├── discriminator.py
│   ├── generator.py
│   ├── embeddings.py
│   └── train.py
│
├── output/
│   │
│   ├── images/
│   │   ├── fake_samples_epoch_0.png
│   │   ├── fake_samples_epoch_1.png
│   │   └── ...
│   │
│   └── checkpoints/
│       ├── netG_epoch_0.pth
│       ├── netD_epoch_0.pth
│       └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```
