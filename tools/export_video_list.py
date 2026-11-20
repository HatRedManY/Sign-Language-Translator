# -*- coding: utf-8 -*-
r"""Строит списки видео для 100 слов, которые знает модель.

Источник разметки: tools/annotations.csv.
Видео: E:\Programs\project\train и E:\Programs\project\test.
Слова: app/src/main/assets/classes.txt (100 слов, порядок совпадает с checkpoint).

В docs/ создаются:
  video_model_words_list.txt  - сгруппировано по слову, формат как в отчёте;
  video_to_word.txt           - видео -> слово, отсортировано по имени видео;
  video_to_word.csv           - то же для Excel, с путём к файлу;
  video_to_word_by_video.txt  - читаемый список "видео — слово (split)";
  words_to_videos.txt         - слово -> все его видео.
"""

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ANN = HERE / "annotations.csv"
WORDS_FILE = REPO / "app" / "src" / "main" / "assets" / "classes.txt"
BASE = Path(r"E:\Programs\project")
OUT = REPO / "docs"


def is_train(value: str) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no")


def main() -> None:
    words = [
        line.strip()
        for line in WORDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    word_set = set(words)

    by_word = defaultdict(list)
    with ANN.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get("text") or "").strip()
            if text not in word_set:
                continue
            split = "train" if is_train(row.get("train")) else "test"
            video = f"{(row.get('attachment_id') or '').strip()}.mp4"
            path = BASE / split / video
            by_word[text].append(
                {
                    "video": video,
                    "split": split,
                    "path": str(path),
                    "exists": path.is_file(),
                    "frames": float(row.get("length") or 0),
                    "height": float(row.get("height") or 0),
                    "width": float(row.get("width") or 0),
                }
            )

    missing = [
        (word, item["split"], item["video"])
        for word, items in by_word.items()
        for item in items
        if not item["exists"]
    ]
    bad_counts = {
        word: len(items) for word, items in by_word.items() if len(items) != 20
    }

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Сгруппировано по слову — точный формат из отчёта.
    grouped = []
    for word in sorted(by_word):
        for item in sorted(by_word[word], key=lambda x: x["video"]):
            grouped.append(
                f"{word} | {item['split']} | {item['video']} | "
                f"frames={item['frames']:.0f} | "
                f"h={item['height']:.0f} w={item['width']:.0f} | "
                f"exists={item['exists']}"
            )
    (OUT / "video_model_words_list.txt").write_text(
        "\n".join(grouped) + "\n", encoding="utf-8", newline="\n"
    )

    # 2. Видео -> слово, как в прежнем файле: по слову, затем по видео.
    by_video = sorted(
        ((word, item) for word, items in by_word.items() for item in items),
        key=lambda x: (x[1]["video"], x[0]),
    )
    with (OUT / "video_to_word.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write("video\tword\tsplit\n")
        for word in sorted(by_word):
            for item in sorted(by_word[word], key=lambda x: x["video"]):
                f.write(f"{item['video']}\t{word}\t{item['split']}\n")

    # 3. CSV для Excel: видео, слово, split, кадры, размеры, путь.
    with (OUT / "video_to_word.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video", "word", "split", "frames", "height", "width", "exists", "path"]
        )
        for word, item in by_video:
            writer.writerow(
                [
                    item["video"],
                    word,
                    item["split"],
                    int(round(item["frames"])),
                    int(round(item["height"])),
                    int(round(item["width"])),
                    item["exists"],
                    item["path"],
                ]
            )

    # 4. Читаемый список в столбик.
    with (OUT / "video_to_word_by_video.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for word, item in by_video:
            f.write(f"{item['video']} — {word} ({item['split']})\n")

    # 5. Слово -> все видео.
    with (OUT / "words_to_videos.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for word in sorted(by_word):
            names = ", ".join(
                item["video"] for item in sorted(by_word[word], key=lambda x: x["video"])
            )
            f.write(f"{word} ({len(by_word[word])}): {names}\n")

    print(f"слов в модели: {len(words)}")
    print(f"слов с видео: {len(by_word)}")
    print(f"строк по словам модели: {len(by_video)}")
    print(
        "split:",
        sum(1 for _, item in by_video if item["split"] == "train"),
        "train /",
        sum(1 for _, item in by_video if item["split"] == "test"),
        "test",
    )
    print(f"отсутствующих файлов: {len(missing)}")
    if bad_counts:
        print("слов с неожиданным числом видео:", bad_counts)
    print("готово: docs/video_model_words_list.txt")
    print("готово: docs/video_to_word.txt")
    print("готово: docs/video_to_word.csv")
    print("готово: docs/video_to_word_by_video.txt")
    print("готово: docs/words_to_videos.txt")


if __name__ == "__main__":
    main()
