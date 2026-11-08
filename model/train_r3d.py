# train_r3d.py
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_HOME"] = os.path.join(SCRIPT_DIR, "torch_cache")

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, SCRIPT_DIR)
from dataset_r3d import SlovoDatasetR3D

# ============================================================
# 0. Настройки
# ============================================================
ANNOTATIONS_PATH = os.path.join(SCRIPT_DIR, "annotations.csv")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
TEST_DIR = os.path.join(SCRIPT_DIR, "test")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs_r3d")
os.makedirs(RUNS_DIR, exist_ok=True)

NUM_CLASSES = 100
NUM_FRAMES = 16
IMG_SIZE = 112            # R3D-18 работает с 112x112
BATCH_SIZE = 4            # R3D меньше, чем MViT, можно 4 или 8
NUM_EPOCHS = 20           # 20 эпох
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

def build_model(num_classes):
    model = r3d_18(weights=None)
    
    # Грузим веса Kinetics-400, если есть
    local_ckpt = os.path.join(SCRIPT_DIR, "torch_cache", "hub", "checkpoints", "r3d_18-b3b3357e.pth")
    if os.path.exists(local_ckpt):
        print(f"Загружаю предобученные веса из:\n  {local_ckpt}")
        state = torch.load(local_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print("  Готово. Веса Kinetics-400 применены.")
    else:
        print(f"[!] Файл с весами не найден: {local_ckpt}")
        print("    Модель будет обучаться с нуля.")

    # Заменяем голову (fc) под наши классы
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features, num_classes)
    )
    return model

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

    label_to_idx = {text: i for i, text in enumerate(top_classes)}
    idx_to_label = {i: text for text, i in label_to_idx.items()}

    train_df = df_small[df_small['train'] == True]
    test_df = df_small[df_small['train'] == False]

    print(f"Train: {len(train_df)} видео, Test: {len(test_df)} видео")

    # --- 2. Датасеты ---
    train_dataset = SlovoDatasetR3D(
        df=train_df, root_dir=TRAIN_DIR, label_to_idx=label_to_idx,
        num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=True,
    )
    test_dataset = SlovoDatasetR3D(
        df=test_df, root_dir=TEST_DIR, label_to_idx=label_to_idx,
        num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 3. Модель ---
    model = build_model(len(top_classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nУстройство: {device}")
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 4. Оптимизатор ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # --- 5. Обучение ---
    best_test_acc = 0.0
    best_epoch = 0
    patience = 5
    epochs_without_improvement = 0

    best_ckpt_path = os.path.join(RUNS_DIR, "r3d18_slovo_100words_best.pth")
    last_ckpt_path = os.path.join(RUNS_DIR, "r3d18_slovo_100words_last.pth")
    log_path = os.path.join(RUNS_DIR, "train_log.txt")

    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"RUNS_DIR = {RUNS_DIR}\nClasses = {len(top_classes)}\nDevice = {device}\n\n")

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for videos, labels in train_loader:
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        correct, total = 0, 0
        with torch.no_grad():
            for videos, labels in test_loader:
                videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100 * correct / total
        scheduler.step(test_acc)

        msg = (f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
               f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
               f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label, 'num_frames': NUM_FRAMES, 'img_size': IMG_SIZE, 'test_acc': test_acc}, last_ckpt_path)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({'model_state_dict': model.state_dict(), 'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label, 'num_frames': NUM_FRAMES, 'img_size': IMG_SIZE, 'test_acc': test_acc}, best_ckpt_path)
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
    print(f"\nОбучение завершено. Лучший Test Acc: {best_test_acc:.2f}% (эпоха {best_epoch})")
    print(f"Лучший чекпоинт: {best_ckpt_path}")
