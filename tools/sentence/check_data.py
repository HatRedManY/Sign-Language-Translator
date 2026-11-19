# -*- coding: utf-8 -*-
"""Быстрая проверка сгенерированных данных.

    python check_data.py --data data_smoke
"""

import argparse
import collections
from pathlib import Path


def read_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            inp, out = line.split("\t")
            rows.append((inp, out))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    data = Path(args.data)

    rows = []
    for split in ("train", "val", "test"):
        part = read_rows(data / f"{split}.tsv")
        print(f"{split}: {len(part)}")
        rows.extend(part)

    vocab = set((data / "vocab.txt").read_text(encoding="utf-8").splitlines())
    print(f"vocab: {len(vocab)}")
    print(f"всего пар: {len(rows)}")

    by_input = collections.defaultdict(set)
    missing = set()
    empty = 0
    for inp, out in rows:
        by_input[inp].add(out)
        if not out.strip():
            empty += 1
        for token in out.split(" "):
            if token not in vocab:
                missing.add(token)

    conflicts = {k: v for k, v in by_input.items() if len(v) > 1}
    print(f"пустых выходов: {empty}")
    print(f"входов с разными выходами: {len(conflicts)}")
    for inp, outs in list(conflicts.items())[:5]:
        print("  CONFLICT", inp, "->", sorted(outs))
    print(f"токенов вне словаря: {len(missing)} {sorted(missing)[:20]}")

    patterns = [
        ("ест", lambda s: " ест " in f" {s} "),
        ("больше", lambda s: " больше " in f" {s} "),
        ("живёт", lambda s: " живёт " in f" {s} "),
        ("у ... есть/нет", lambda s: s.startswith("у ")),
        ("—", lambda s: " — " in f" {s} "),
        ("список", lambda s: "," in s),
        ("виляет", lambda s: "виляет" in s),
        ("плетёт", lambda s: "плетёт" in s),
        ("делает", lambda s: "делает" in s),
    ]
    print()
    for name, pred in patterns:
        picked = [(i, o) for i, o in rows if pred(o)]
        print(f"{name}: {len(picked)}")
        for i, o in picked[:4]:
            print(f"  {i.replace('|', ' + '):45s} -> {o}")

    max_in = max(len(i.split("|")) for i, _ in rows)
    max_out = max(len(o.split(" ")) for _, o in rows)
    print(f"\nмакс. вход: {max_in} слов, макс. выход: {max_out} токенов")


if __name__ == "__main__":
    main()
