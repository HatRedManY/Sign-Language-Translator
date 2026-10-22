# train.py
# ============================================================
# ВАЖНО: TORCH_HOME задаётся ДО импорта torch
# ============================================================
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TORCH_CACHE_DIR = os.path.join(SCRIPT_DIR, "torch_cache")
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)

os.environ["TORCH_HOME"] = TORCH_CACHE_DIR
os.environ["HF_HOME"] = os.path.join(SCRIPT_DIR, "hf_cache")

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, SCRIPT_DIR)
from dataset import SlovoDataset


# ============================================================
# 0. Настройки
# ============================================================
ANNOTATIONS_PATH = os.path.join(SCRIPT_DIR, "annotations.csv")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
TEST_DIR = os.path.join(SCRIPT_DIR, "test")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

NUM_CLASSES = 100
NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # 0 = без воркеров, надёжно на Windows


# ============================================================
# Функция загрузки модели (вне __main__, чтобы было чисто)
# ============================================================
def build_model(num_classes):
    model = mvit_v2_s(weights=None)

    local_ckpt = os.path.join(
        TORCH_CACHE_DIR, "hub", "checkpoints",
        "mvit_v2_s-ae3be167.pth",
    )
    if os.path.exists(local_ckpt):
        print(f"Загружаю предобученные веса из:\n  {local_ckpt}")
        state = torch.load(local_ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [!] Не загружено слоёв: {len(missing)}")
        if unexpected:
            print(f"  [!] Лишних ключей: {len(unexpected)}")
        print("  Готово. Веса Kinetics-400 применены.")
    else:
        print(f"[!] Файл с весами не найден:\n  {local_ckpt}")
        print("    Модель будет обучаться с нуля.")

    in_features = model.head[1].in_features
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features, num_classes),
    )
    return model


# ============================================================
# Основной блок — ВСЁ ВНУТРИ if __name__ == "__main__"
# ============================================================
if __name__ == "__main__":

    # --- 1. Данные ---
    df = pd.read_csv(ANNOTATIONS_PATH, encoding='utf-8', sep='\t')
    df = df[df['text'] != 'no_event'].copy()

    letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
    df_words = df[~df['text'].isin(letters)].copy()

    top_classes = df_words['text'].value_counts().head(NUM_CLASSES).index.tolist()
    df_small = df_words[df_words['text'].isin(top_classes)].copy()

    print(f"Классов: {len(top_classes)}")
    print(f"Видео: {len(df_small)}")
    print(f"Первые 10 слов: {top_classes[:10]}")

    label_to_idx = {text: i for i, text in enumerate(top_classes)}
    idx_to_label = {i: text for text, i in label_to_idx.items()}

    train_df = df_small[df_small['train'] == True]
    test_df = df_small[df_small['train'] == False]

    print(f"Train: {len(train_df)} видео")
    print(f"Test: {len(test_df)} видео")

    # --- 2. Датасеты и загрузчики ---
    train_dataset = SlovoDataset(
        df=train_df, root_dir=TRAIN_DIR, label_to_idx=label_to_idx,
        num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=True,
    )
    test_dataset = SlovoDataset(
        df=test_df, root_dir=TEST_DIR, label_to_idx=label_to_idx,
        num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        # persistent_workers=True,   # убрано — вызывает deadlock на Windows
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        # persistent_workers=True,   # убрано
    )

    # --- 3. Модель ---
    model = build_model(len(top_classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nУстройство: {device}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {trainable:,}")

    # --- 4. Оптимизатор, loss, scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )   # verbose убран

    # --- 5. Обучение ---
    best_test_acc = 0.0
    best_epoch = 0
    patience = 5
    epochs_without_improvement = 0

    best_ckpt_path = os.path.join(RUNS_DIR, "mvitv2_slovo_100words_best.pth")
    last_ckpt_path = os.path.join(RUNS_DIR, "mvitv2_slovo_100words_last.pth")
    log_path = os.path.join(RUNS_DIR, "train_log.txt")

    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"RUNS_DIR = {RUNS_DIR}\n")
    log_file.write(f"Classes  = {len(top_classes)}\n")
    log_file.write(f"Device   = {device}\n\n")

    for epoch in range(NUM_EPOCHS):

        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in train_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Test ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in test_loader:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(videos)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100 * correct / total
        scheduler.step(test_acc)

        msg = (f"Epoch {epoch+1}/{NUM_EPOCHS} | "
               f"Loss: {avg_loss:.4f} | "
               f"Train Acc: {train_acc:.2f}% | "
               f"Test Acc: {test_acc:.2f}% | "
               f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        # --- Сохранение ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_frames': NUM_FRAMES,
            'img_size': IMG_SIZE,
            'test_acc': test_acc,
        }, last_ckpt_path)

        # --- Early stopping ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'num_frames': NUM_FRAMES,
                'img_size': IMG_SIZE,
                'test_acc': test_acc,
            }, best_ckpt_path)

            print(f"  → Новая лучшая модель (Test Acc: {test_acc:.2f}%)")
            log_file.write(f"  -> new best: {test_acc:.2f}%\n")
            log_file.flush()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping на эпохе {epoch+1}")
                log_file.write(f"\nEarly stopping on epoch {epoch+1}\n")
                break

    log_file.close()

    print(f"\nОбучение завершено.")
    print(f"Лучший Test Acc: {best_test_acc:.2f}% (эпоха {best_epoch})")
    print(f"Лучший чекпоинт: {best_ckpt_path}")
    print(f"Лог: {log_path}")
