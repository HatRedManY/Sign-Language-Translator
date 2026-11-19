# -*- coding: utf-8 -*-
"""Быстрый smoke-тест модели и экспорта.  Обучение не запускается.

    python smoke_test.py

Проверяет: forward, backward, generate, torch.jit.script, сохранение .ptl.
Если этот тест проходит, архитектура и экспорт совместимы с PyTorch.
"""

from pathlib import Path

import torch

from export_ptl import Exported, check_ptl_compat
from train_seq2seq import Seq2Seq


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 300
    model = Seq2Seq(
        vocab_size=vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=0.0,
        max_len=16,
    )
    src = torch.randint(4, vocab_size, (4, 6))
    tgt = torch.randint(4, vocab_size, (4, 8))

    logits = model(src, tgt[:, :-1])
    assert logits.shape == (4, 7, vocab_size), logits.shape

    loss = logits.float().pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    model.eval()
    with torch.no_grad():
        eager = model.generate(src)

    scripted = torch.jit.script(Exported(model).eval())
    with torch.no_grad():
        script = scripted(src)

    assert torch.equal(eager, script), "scripted output != eager output"
    assert script.shape[0] == 4 and script.shape[1] <= 16

    out_dir = Path("smoke")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "sent.ptl"
    scripted._save_for_lite_interpreter(str(path))
    assert path.exists() and path.stat().st_size > 0
    check_ptl_compat(path)

    print(f"SMOKE OK: logits={tuple(logits.shape)}, out={tuple(script.shape)}, "
          f"ptl={path.stat().st_size/1024:.0f} КБ")


if __name__ == "__main__":
    main()
