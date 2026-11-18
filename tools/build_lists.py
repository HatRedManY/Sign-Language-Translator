"""Строит списки видео по словам модели.

Вход:
  tools/annotations.csv                                - разметка датасета
  <BASE>/runs_r3d/r3d18_slovo_100words_best.pth        - чекпойнт (берём label_to_idx)
  <BASE>/train/*.mp4, <BASE>/test/*.mp4                - сами видео

Выход (docs/):
  video_model_words_list.txt  - сгруппировано по слову (как раньше)
  video_to_word.txt           - "видео -> слово" в столбик
  words_to_videos.txt         - "слово -> все его видео"
"""

import csv
import os
from collections import defaultdict

import torch

BASE = r"E:\Programs\project"
HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "annotations.csv")
CKPT = os.path.join(BASE, "runs_r3d", "r3d18_slovo_100words_best.pth")
OUT = os.path.abspath(os.path.join(HERE, "..", "docs"))


def is_train(value: str) -> bool:
    v = (value or "").strip().lower()
    return v not in ("", "0", "false", "no")


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    words = list(ck["label_to_idx"].keys())
    word_set = set(words)
    print(f"слов в модели: {len(words)}")

    by_word = defaultdict(list)
    rows_total = 0
    missing = []

    with open(ANN, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get("text") or "").strip()
            if text not in word_set:
                continue
            rows_total += 1
            split = "train" if is_train(row.get("train")) else "test"
            fname = (row.get("attachment_id") or "").strip() + ".mp4"
            path = os.path.join(BASE, split, fname)
            exists = os.path.isfile(path)
            if not exists:
                missing.append((split, fname, text))
            by_word[text].append(
                {
                    "video": fname,
                    "split": split,
                    "exists": exists,
                    "frames": float(row.get("length") or 0),
                    "height": float(row.get("height") or 0),
                    "width": float(row.get("width") or 0),
                }
            )

    print(f"строк по словам модели: {rows_total}")
    print(f"слов с видео: {len(by_word)}")
    print(f"отсутствующих файлов: {len(missing)}")
    for item in missing[:20]:
        print("  НЕТ ФАЙЛА:", item)

    os.makedirs(OUT, exist_ok=True)

    grouped = []
    for word in sorted(by_word):
        for item in sorted(by_word[word], key=lambda x: x["video"]):
            grouped.append(
                f"{word} | {item['split']} | {item['video']} | "
                f"frames={item['frames']:.0f} | "
                f"h={item['height']:.0f} w={item['width']:.0f} | "
                f"exists={item['exists']}"
            )
    with open(os.path.join(OUT, "video_model_words_list.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(grouped) + "\n")

    to_word = []
    for word in sorted(by_word):
        for item in sorted(by_word[word], key=lambda x: x["video"]):
            to_word.append(f"{item['video']}\t{word}\t{item['split']}")
    with open(os.path.join(OUT, "video_to_word.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("video\tword\tsplit\n")
        f.write("\n".join(to_word) + "\n")

    to_videos = []
    for word in sorted(by_word):
        names = ", ".join(x["video"] for x in sorted(by_word[word], key=lambda x: x["video"]))
        to_videos.append(f"{word} ({len(by_word[word])}): {names}")
    with open(os.path.join(OUT, "words_to_videos.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(to_videos) + "\n")

    print("готово:")
    print("  docs/video_model_words_list.txt", len(grouped), "строк")
    print("  docs/video_to_word.txt", len(to_word), "строк")
    print("  docs/words_to_videos.txt", len(to_videos), "слов")


if __name__ == "__main__":
    main()
