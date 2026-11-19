# -*- coding: utf-8 -*-
"""Обучение маленького seq2seq «слова -> предложение».

Модель — обычный Transformer encoder-decoder на 4+4 слоя, d_model=192.
Вход: id распознанных слов (меток).  Выход: id словоформ и служебных слов.
Словарь общий, поэтому никаких «посторонних» слов модель не придумает.

Обычный запуск::

    python train_seq2seq.py --data data --out checkpoints --epochs 15

Скрипт не требует transformers и работает на чистом PyTorch.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD, BOS, EOS, UNK = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Attention(nn.Module):
    """Multi-head attention на bmm/softmax.

    Специально не используем nn.MultiheadAttention и
    scaled_dot_product_attention: PyTorch Mobile 2.1.0 (его использует
    приложение) не умеет часть этих fused-операций.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model должен делиться на nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = query.size(0)
        tq = query.size(1)
        tk = key.size(1)
        heads = self.nhead
        dh = self.d_head

        q = self.q_proj(query).reshape(batch, tq, heads, dh).transpose(1, 2)
        q = q.reshape(batch * heads, tq, dh)
        k = self.k_proj(key).reshape(batch, tk, heads, dh).transpose(1, 2)
        k = k.reshape(batch * heads, tk, dh)
        v = self.v_proj(value).reshape(batch, tk, heads, dh).transpose(1, 2)
        v = v.reshape(batch * heads, tk, dh)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(0).expand(batch * heads, tq, tk)
            scores = scores.masked_fill(mask, -1e9)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).expand(batch, heads, tk)
            mask = mask.reshape(batch * heads, tk)
            mask = mask.unsqueeze(1).expand(batch * heads, tq, tk)
            scores = scores.masked_fill(mask, -1e9)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.bmm(weights, v)
        out = out.reshape(batch, heads, tq, dh).transpose(1, 2)
        out = out.reshape(batch, tq, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.self_attn(h, h, h, None, src_pad))
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout)
        self.cross_attn = Attention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad: torch.Tensor,
        causal_mask: torch.Tensor,
        src_pad: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.self_attn(h, h, h, causal_mask, tgt_pad))
        h = self.norm2(x)
        x = x + self.drop2(self.cross_attn(h, memory, memory, None, src_pad))
        h = self.norm3(x)
        x = x + self.drop3(self.ff(h))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_pad)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad: torch.Tensor,
        causal_mask: torch.Tensor,
        src_pad: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_pad, causal_mask, src_pad)
        return x


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_ff: int = 768,
        dropout: float = 0.1,
        max_len: int = 24,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.max_len = max_len
        self.pad_id = PAD
        self.bos_id = BOS
        self.eos_id = EOS

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len)
        self.encoder = Encoder(num_layers, d_model, nhead, dim_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, nhead, dim_ff, dropout)
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        causal = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.src_embed(src))
        x = self.encoder(x, src == self.pad_id)
        return self.enc_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_pad: torch.Tensor,
    ) -> torch.Tensor:
        t = tgt.size(1)
        x = self.pos(self.tgt_embed(tgt))
        x = self.decoder(
            x, memory, tgt == self.pad_id, self.causal_mask[:t, :t], src_pad
        )
        return self.lm_head(self.dec_norm(x))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src)
        return self.decode(tgt_in, memory, src == self.pad_id)

    def generate(self, src: torch.Tensor) -> torch.Tensor:
        """Жадное декодирование.  Возвращает [B, <= max_len] id."""
        memory = self.encode(src)
        tgt = torch.full(
            (src.size(0), 1), self.bos_id, dtype=torch.int64, device=src.device
        )
        for _ in range(self.max_len - 1):
            logits = self.decode(tgt, memory, src == self.pad_id)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tgt = torch.cat([tgt, nxt], dim=1)
        return tgt


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, path: Path, vocab: dict, max_len: int):
        self.rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                inp, out = line.split("\t")
                src = [BOS] + [vocab.get(x, UNK) for x in inp.split("|")] + [EOS]
                tgt = [BOS] + [vocab.get(x, UNK) for x in out.split(" ")] + [EOS]
                self.rows.append((self._fit(src, max_len), self._fit(tgt, max_len)))

    @staticmethod
    def _fit(ids, max_len):
        if len(ids) > max_len:
            return ids[: max_len - 1] + [EOS]
        return ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch):
    srcs, tgts = zip(*batch)
    max_s = max(len(x) for x in srcs)
    max_t = max(len(x) for x in tgts)
    src = torch.full((len(batch), max_s), PAD, dtype=torch.int64)
    tgt = torch.full((len(batch), max_t), PAD, dtype=torch.int64)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.int64)
        tgt[i, : len(t)] = torch.tensor(t, dtype=torch.int64)
    return src, tgt


def trim_ids(ids):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        out.append(i)
    return out


def load_vocab(path: Path):
    tokens = path.read_text(encoding="utf-8").splitlines()
    tokens = [t for t in tokens if t != ""]
    if tokens[:4] != ["<pad>", "<bos>", "<eos>", "<unk>"]:
        raise SystemExit(
            "vocab.txt должен начинаться с <pad>, <bos>, <eos>, <unk>"
        )
    return tokens


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, max_exact: int = 1000):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    exact = 0
    seen = 0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(next(model.parameters()).device)
            tgt = tgt.to(next(model.parameters()).device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            total_batches += 1

            if seen < max_exact:
                pred = model.generate(src)
                for p, t in zip(pred.tolist(), tgt.tolist()):
                    if seen >= max_exact:
                        break
                    seen += 1
                    if trim_ids(p) == trim_ids(t):
                        exact += 1

    model.train()
    avg_loss = total_loss / max(1, total_batches)
    exact_rate = exact / max(1, seen)
    return avg_loss, exact, seen, exact_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение seq2seq для предложений")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if is_cuda:
        torch.cuda.manual_seed_all(args.seed)

    vocab = load_vocab(data_dir / "vocab.txt")
    token_to_id = {token: i for i, token in enumerate(vocab)}
    vocab_size = len(vocab)

    train_ds = PairDataset(data_dir / "train.tsv", token_to_id, args.max_len)
    val_ds = PairDataset(data_dir / "val.tsv", token_to_id, args.max_len)

    pin = is_cuda
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=pin,
    )

    model = Seq2Seq(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_ff=args.ff,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "nhead": args.heads,
        "num_layers": args.layers,
        "dim_ff": args.ff,
        "dropout": args.dropout,
        "max_len": args.max_len,
    }

    log_path = out_dir / "train_log.txt"
    log_file = log_path.open("a", encoding="utf-8")

    def log(text: str = "") -> None:
        print(text)
        log_file.write(text + "\n")
        log_file.flush()

    log("=" * 72)
    log(f"device={device}  vocab={vocab_size}  params={params/1e6:.2f}M")
    log(f"train={len(train_ds)}  val={len(val_ds)}  batch={args.batch}  epochs={args.epochs}")
    log(f"config={config}")
    log("=" * 72)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = max(1, int(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_score = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    best_path = out_dir / "best.pt"
    global_step = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        epoch_start = time.time()

        for src, tgt in train_loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=is_cuda
            ):
                logits = model(src, tgt_in)
                loss = criterion(
                    logits.reshape(-1, vocab_size), tgt_out.reshape(-1)
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            batches += 1
            global_step += 1

            if global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                log(
                    f"epoch {epoch:02d}  step {global_step:06d}  "
                    f"loss {running/batches:.4f}  lr {lr:.2e}"
                )

        train_loss = running / max(1, batches)
        val_loss, exact, seen, exact_rate = evaluate(
            model, val_loader, criterion, max_exact=1000
        )

        score = exact_rate - 0.01 * val_loss
        improved = score > best_score
        if improved:
            best_score = score
            best_loss = val_loss
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": config}, best_path)
        else:
            bad_epochs += 1

        elapsed = time.time() - epoch_start
        log(
            f"epoch {epoch:02d}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  exact {exact}/{seen} ({exact_rate:.1%})  "
            f"{elapsed:.0f}s  {'*best*' if improved else ''}"
        )

        torch.save({"model": model.state_dict(), "config": config}, out_dir / "last.pt")

        if bad_epochs >= args.patience:
            log(f"Ранняя остановка: {args.patience} эпох без улучшения.")
            break

    log("=" * 72)
    log(f"Готово за {(time.time()-started)/60:.1f} мин.  best: {best_path}")
    log(f"best_score={best_score:.4f}  best_val_loss={best_loss:.4f}")
    log("Следующий шаг: python export_ptl.py --checkpoint checkpoints/best.pt")
    log_file.close()


if __name__ == "__main__":
    main()
