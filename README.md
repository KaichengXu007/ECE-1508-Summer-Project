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
