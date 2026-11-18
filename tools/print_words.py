import os, json
import pandas as pd
import torch

BASE = r"E:\Programs\prilojuxa"
df = pd.read_csv(os.path.join(BASE, "annotations.csv"), sep="\t", encoding="utf-8-sig")
ck = torch.load(r"E:\Programs\project\runs_r3d\r3d18_slovo_100words_best.pth", map_location="cpu", weights_only=True)
words = [k for k, v in ck["label_to_idx"].items()]
print("WORD COUNT:", len(words))
for w in sorted(words, key=lambda x: ck["label_to_idx"][x]):
    print(f"{ck['label_to_idx'][w]:3d} | {w}")
