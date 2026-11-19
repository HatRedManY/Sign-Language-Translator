# -*- coding: utf-8 -*-
"""Экспорт обученной модели в TorchScript Lite (.ptl) для Android.

    python export_ptl.py --checkpoint checkpoints/best.pt

На выходе:

* ``sentence/sent.ptl`` — модель с уже встроенным жадным декодированием;
  на вход подаётся LongTensor [1, N] с id слов, на выходе LongTensor [1, M]
  с id словоформ (останавливаться по <eos>);
* ``sentence/vocab.txt`` — тот же словарь, что при обучении.

Скрипт также сверяет выходы обычной и scripted-модели на test.tsv.
"""

import argparse
import pickle
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn as nn

from train_seq2seq import BOS, EOS, PAD, Seq2Seq


class Exported(nn.Module):
    """Обёртка, чтобы в Android можно было вызывать module.forward(src)."""

    def __init__(self, model: Seq2Seq):
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model.generate(src)


def check_ptl_compat(path: Path) -> None:
    """Печатает версии формата и проверяет совместимость с Mobile 2.1.0.

    PyTorch Mobile 2.1.0 поддерживает bytecode 4..9 и file format <= 10.
    Экспорт из более новых версий PyTorch обычно даёт те же 8/10, но
    проверить полезно: при несовместимости Android не загрузит sent.ptl.
    """
    with zipfile.ZipFile(path) as archive:
        root = path.stem
        bytecode = pickle.loads(archive.read(f"{root}/bytecode.pkl"))[0]
        file_version = int(
            archive.read(f"{root}/.data/version").decode("ascii").strip()
        )

    print(f"Формат .ptl: bytecode={bytecode}, file={file_version}")
    if not (4 <= bytecode <= 9):
        print(
            "ВНИМАНИЕ: bytecode вне диапазона 4..9 — "
            "PyTorch Mobile 2.1.0 может не загрузить модель."
        )
    if file_version > 10:
        print(
            "ВНИМАНИЕ: file format больше 10 — "
            "PyTorch Mobile 2.1.0 может не загрузить модель."
        )


def detokenize(tokens) -> str:
    out = ""
    for token in tokens:
        if token in {".", ",", "?", "!"}:
            out += token
        elif token == "—":
            out = out.rstrip() + " — "
        else:
            if out and not out.endswith(" "):
                out += " "
            out += token
    return out.strip().capitalize()


def encode(labels, token_to_id):
    ids = [BOS]
    for label in labels:
        ids.append(token_to_id.get(label, 3))
    ids.append(EOS)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт модели в sent.ptl")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="sentence")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Нет чекпоинта {ckpt_path}. Сначала запустите train_seq2seq.py"
        )

    data_dir = Path(args.data)
    vocab_path = data_dir / "vocab.txt"
    if not vocab_path.exists():
        raise SystemExit(f"Нет словаря {vocab_path}. Сначала запустите build_pairs.py")

    vocab = [t for t in vocab_path.read_text(encoding="utf-8").splitlines() if t]
    if vocab[:4] != ["<pad>", "<bos>", "<eos>", "<unk>"]:
        raise SystemExit("vocab.txt повреждён: первые 4 токена должны быть служебными")

    token_to_id = {token: i for i, token in enumerate(vocab)}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model = Seq2Seq(**config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    exported = Exported(model).eval()
    scripted = torch.jit.script(exported)

    # Сверка на тестовых примерах.
    test_path = data_dir / "test.tsv"
    rows = []
    if test_path.exists():
        with test_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    inp, out = line.split("\t")
                    rows.append((inp.split("|"), out.split(" ")))
    rows = rows[: args.limit]

    mismatches = 0
    examples = []
    with torch.no_grad():
        for labels, target in rows:
            src = torch.tensor([encode(labels, token_to_id)], dtype=torch.int64)
            a = model.generate(src)
            b = scripted(src)
            if not torch.equal(a, b):
                mismatches += 1
            if len(examples) < 5:
                ids = [int(x) for x in b[0].tolist()]
                words = []
                for i in ids:
                    if i == EOS:
                        break
                    if i in (PAD, BOS):
                        continue
                    words.append(vocab[i])
                examples.append((" + ".join(labels), detokenize(words), " ".join(target)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptl_path = out_dir / "sent.ptl"
    scripted._save_for_lite_interpreter(str(ptl_path))
    check_ptl_compat(ptl_path)
    shutil.copyfile(vocab_path, out_dir / "vocab.txt")

    print(f"Проверено примеров: {len(rows)}, расхождений eager/scripted: {mismatches}")
    for inp, pred, target in examples:
        print(f"  {inp:40s} -> {pred}")
        print(f"  {'':40s}    эталон: {target}")
    print()
    print(f"Модель: {ptl_path.resolve()}  ({ptl_path.stat().st_size/1e6:.1f} МБ)")
    print(f"Словарь: {(out_dir/'vocab.txt').resolve()}")
    print("Теперь скопируйте оба файла в app/src/main/assets/sentence/.")


if __name__ == "__main__":
    main()
