import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.model_selection import train_test_split

from dataset import SlovoDataset


# ============================================================
# 0. Настройки
# ============================================================
ANNOTATIONS_PATH = 'annotations.csv'
TRAIN_DIR = 'train'
TEST_DIR = 'test'
NUM_CLASSES = 100          # <-- НАЧНЁМ СО 100 СЛОВ
NUM_FRAMES = 16
IMG_SIZE = 112
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0


# ============================================================
# 1. Загружаем разметку
# ============================================================
df = pd.read_csv(ANNOTATIONS_PATH, encoding='utf-8', sep='\t')
df = df[df['text'] != 'no_event'].copy()

print(f"Всего строк: {len(df)}")
print(f"Уникальных классов: {df['text'].nunique()}")


# ============================================================
# 2. Берём NUM_CLASSES самых частых классов (слова, не буквы)
# ============================================================
# Убираем буквы
letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
df_words = df[~df['text'].isin(letters)].copy()

print(f"Видео со словами: {len(df_words)}")
print(f"Уникальных слов: {df_words['text'].nunique()}")

# Топ-N слов
top_classes = df_words['text'].value_counts().head(NUM_CLASSES).index.tolist()
df_small = df_words[df_words['text'].isin(top_classes)].copy()

print(f"\nКлассов: {len(top_classes)}")
print(f"Видео: {len(df_small)}")
print(f"\nПервые 10 слов: {top_classes[:10]}")
print(f"Последние 10 слов: {top_classes[-10:]}")


# ============================================================
# 3. Словарь классов
# ============================================================
label_to_idx = {text: i for i, text in enumerate(top_classes)}
idx_to_label = {i: text for text, i in label_to_idx.items()}


# ============================================================
# 4. Разбиваем на train/test ЗАНОВО (используем все видео)
# ============================================================
# Вместо df['train'] используем train_test_split
# Это даст больше данных для обучения
train_df, test_df = train_test_split(
    df_small,
    test_size=0.2,
    stratify=df_small['text'],
    random_state=42
)

print(f"\nTrain: {len(train_df)} видео")
print(f"Test: {len(test_df)} видео")
print(f"Train на класс: ~{len(train_df) / NUM_CLASSES:.1f}")
print(f"Test на класс: ~{len(test_df) / NUM_CLASSES:.1f}")


# ============================================================
# 5. Датасеты
# ============================================================
# ВАЖНО: все видео лежат либо в train/, либо в test/
# Если используем train_test_split, нужно указать правильный root_dir
# Проще: объединить все видео в одну папку ИЛИ использовать флаг train

# Вариант А: используем оригинальное разделение (train/test папки)
# train_df = df_small[df_small['train'] == True]
# test_df = df_small[df_small['train'] == False]

# Вариант Б: используем train_test_split (нужно, чтобы все видео были в одной папке)
# Если видео в train/ и test/, то нужно объединить их

# Пока используем оригинальное разделение
train_df = df_small[df_small['train'] == True]
test_df = df_small[df_small['train'] == False]

print(f"\nTrain: {len(train_df)} видео")
print(f"Test: {len(test_df)} видео")

train_dataset = SlovoDataset(
    df=train_df, root_dir=TRAIN_DIR, label_to_idx=label_to_idx,
    num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=True,
)

test_dataset = SlovoDataset(
    df=test_df, root_dir=TEST_DIR, label_to_idx=label_to_idx,
    num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=False,
)


# ============================================================
# 6. DataLoader
# ============================================================
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)


# ============================================================
# 7. Проверка батча
# ============================================================
videos, labels = next(iter(train_loader))
print(f"\nBatch shape: {videos.shape}")
print(f"Batch memory: {videos.element_size() * videos.nelement() / 1024**2:.2f} MB")


# ============================================================
# 8. Модель с замороженным backbone
# ============================================================
model = r3d_18(weights=R3D_18_Weights.DEFAULT)

# Замораживаем все слои
for param in model.parameters():
    param.requires_grad = False

# Заменяем fc и размораживаем только его
model.fc = torch.nn.Linear(model.fc.in_features, len(top_classes))
for param in model.fc.parameters():
    param.requires_grad = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# BatchNorm в eval
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm3d):
        module.eval()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nУстройство: {device}")
print(f"Обучаемых параметров: {trainable:,} из {total_params:,} "
      f"({100*trainable/total_params:.2f}%)")


# ============================================================
# 9. Оптимизатор и loss
# ============================================================
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')


# ============================================================
# 10. Цикл обучения с early stopping
# ============================================================
best_test_acc = 0.0
best_epoch = 0
patience = 3
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    
    # --- Обучение ---
    model.eval()  # BatchNorm в eval
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
    
    # --- Тест ---
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
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
          f"Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, "
          f"Test Acc: {test_acc:.2f}%")
    
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
        }, 'r3d18_slovo_100words_best.pth')
        print(f"  → Новая лучшая модель (Test Acc: {test_acc:.2f}%)")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping на эпохе {epoch+1}")
            break

print(f"\nОбучение завершено.")
print(f"Лучший Test Acc: {best_test_acc:.2f}% (эпоха {best_epoch})")
