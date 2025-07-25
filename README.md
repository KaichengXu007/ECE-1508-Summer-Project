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
