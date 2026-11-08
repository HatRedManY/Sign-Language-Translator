import os, glob, json, csv
from collections import defaultdict

BASE = r"E:\Programs\prilojuxa"
ANN = os.path.join(BASE, "annotations.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
OUT_CSV = os.path.join(BASE, "video_list_with_gestures.csv")

import pandas as pd

df = pd.read_csv(ANN, sep="\t", encoding="utf-8-sig")
print("columns:", df.columns.tolist())
print("rows:", len(df))
print("unique texts:", df["text"].nunique())
print("no_event count:", int((df["text"] == "no_event").sum()))

exts = (".mp4", ".avi", ".mov", ".mkv")
tf = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(exts))
lf = sorted(f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(exts))
print("files in test/: ", len(tf))
print("files in train/: ", len(lf))

rows = []
for _, r in df.iterrows():
    att = str(r["attachment_id"])
    text = str(r["text"])
    split = bool(r["train"])
    length = float(r["length"]) if pd.notna(r["length"]) else float("nan")
    h = float(r["height"]) if pd.notna(r["height"]) else float("nan")
    w = float(r["width"]) if pd.notna(r["width"]) else float("nan")
    d = TEST_DIR if (not split) else TRAIN_DIR
    fname = att + ".mp4"
    exists = os.path.exists(os.path.join(d, fname))
    rows.append({
        "split": "test" if (not split) else "train",
        "attachment_id": att,
        "text": text,
        "file": fname,
        "exists": exists,
        "length_frames": length,
        "height": h,
        "width": w,
        "user_id": str(r["user_id"]),
    })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["split", "attachment_id", "text", "file", "exists", "length_frames", "height", "width", "user_id"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("wrote", OUT_CSV, "rows:", len(rows))

by_text = defaultdict(list)
for r in rows:
    by_text[r["text"]].append(r)
word_texts = sorted([k for k, v in by_text.items() if k != "no_event"])
print("distinct words (excl no_event):", len(word_texts))
print("videos of those words in test+train:", sum(len(by_text[w]) for w in word_texts))
print("total rows where exists:", sum(1 for r in rows if r["exists"]))

import json as _json
_json.dump(rows, open(r"E:\Programs/prilojuxa/video_list_with_gestures.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

lines = []
for w in word_texts:
    for r in sorted(by_text[w], key=lambda x: x["attachment_id"]):
        lines.append((w, r))
with open(r"E:\Programs/prilojuxa/video_words_list.txt", "w", encoding="utf-8") as f:
    for w, r in lines:
        f.write(f"{w} | {r['split']} | {r['file']} | frames={r['length_frames']} | h={r['height']} w={r['width']} | exists={r['exists']}\n")
print("wrote video_words_list.txt lines:", len(lines))
print("\n--- ПРИМЕР (первые 60 строк списка) ---")
for w, r in lines[:60]:
    print(f"{w} | {r['split']} | {r['file']} | frames={r['length_frames']} | h={r['height']} w={r['width']} | exists={r['exists']}")
