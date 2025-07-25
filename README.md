## 项目结构 (Project Structure)

本项目的代码和数据遵循一个清晰、模块化的目录结构，以便于维护和扩展。

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

### 目录说明

* **`data/`**: 存放所有原始数据和预处理后的数据文件。
* **`src/`**: 存放所有 Python 源代码。
* **`output/`**: 存放训练过程中生成的图片和模型文件。
* **`README.md`**: 项目说明文档。
* **`requirements.txt`**: 项目依赖的 Python 库列表。
* **`.gitignore`**: 指定 Git 版本控制应忽略的文件和目录。
