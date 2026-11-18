import os, json
import pandas as pd
import torch
from collections import defaultdict

BASE = r"E:\Programs\prilojuxa"
CKPT = r"E:\Programs\project\runs_r3d\r3d18_slovo_100words_best.pth"
df = pd.read_csv(os.path.join(BASE, "annotations.csv"), sep="\t", encoding="utf-8-sig")
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
words = set(ck["label_to_idx"].keys())
dfw = df[df["text"].isin(words)].copy()

by_word = defaultdict(list)
for _, r in dfw.iterrows():
    split = "test" if (not bool(r["train"])) else "train"
    d = os.path.join(BASE, split)
    fname = str(r["attachment_id"]) + ".mp4"
    exists = os.path.exists(os.path.join(d, fname))
    by_word[str(r["text"])].append((fname, split, exists, float(r["length"]) if pd.notna(r["length"]) else 0.0, float(r["height"]) if pd.notna(r["height"]) else 0.0, float(r["width"]) if pd.notna(r["width"]) else 0.0))

print("words in model:", len(by_word))
tot = 0
for w in sorted(by_word):
    items = by_word[w]
    tot += len(items)
    ex = [x for x in items if x[2]]
    if len(ex) != len(items):
        print("MISSING FILES for", w, [x[0] for x in items if not x[2]])
print("total video rows for model words:", tot)
tr = sum(1 for w in by_word for x in by_word[w] if x[1] == "train" and x[2])
te = sum(1 for w in by_word for x in by_word[w] if x[1] == "test" and x[2])
print("train rows:", tr, "test rows:", te)

out_lines = []
for w in sorted(by_word):
    for fname, split, exists, length, h, wd in sorted(by_word[w], key=lambda x: x[0]):
        out_lines.append(f"{w} | {split} | {fname} | frames={length:.0f} | h={h:.0f} w={wd:.0f} | exists={exists}")
with open(r"E:\Programs/prilojuxa\video_model_words_list.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print("wrote video_model_words_list.txt, lines:", len(out_lines))
