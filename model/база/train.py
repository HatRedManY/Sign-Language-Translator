import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from dataset import SlovoDataset


# ============================================================
# 0. Настройки
# ============================================================
ANNOTATIONS_PATH = 'annotations.csv'
TRAIN_DIR = 'train'
TEST_DIR = 'test'
NUM_CLASSES = 20          # сколько частых классов берём
NUM_FRAMES = 16           # кадров на видео
IMG_SIZE = 112            # сторона кадра (R3D-18 обучался на 112)
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0           # на Windows лучше 0


# ============================================================
# 1. Загружаем разметку
# ============================================================
df = pd.read_csv(ANNOTATIONS_PATH, encoding='utf-8', sep='\t')

print(f"Всего строк: {len(df)}")
print(f"Уникальных жестов: {df['text'].nunique()}")


# ============================================================
# 2. Убираем служебный класс no_event
# ============================================================
# no_event — это пауза/переход, а не жест. Он только сбивает модель.
df = df[df['text'] != 'no_event'].copy()

print(f"После удаления no_event: {len(df)} строк")


# ============================================================
# 3. Фильтруем данные: берём только NUM_CLASSES частых классов
# ============================================================
top_classes = df['text'].value_counts().head(NUM_CLASSES).index.tolist()
df_small = df[df['text'].isin(top_classes)].copy()

print(f"Классов: {len(top_classes)}")
print(f"Видео: {len(df_small)}")
print(top_classes)


# ============================================================
# 4. Строим словарь: текст -> индекс
# ============================================================
label_to_idx = {text: i for i, text in enumerate(top_classes)}
idx_to_label = {i: text for text, i in label_to_idx.items()}

print(f"Словарь классов: {label_to_idx}")


# ============================================================
# 5. Разделяем на train и test
# ============================================================
train_df = df_small[df_small['train'] == True]
test_df = df_small[df_small['train'] == False]

print(f"Train: {len(train_df)} видео")
print(f"Test: {len(test_df)} видео")


# ============================================================
# 6. Создаём датасеты
# ============================================================
train_dataset = SlovoDataset(
    df=train_df,
    root_dir=TRAIN_DIR,
    label_to_idx=label_to_idx,
    num_frames=NUM_FRAMES,
    img_size=IMG_SIZE,
    training=True,      # включает случайное смещение начала
)

test_dataset = SlovoDataset(
    df=test_df,
    root_dir=TEST_DIR,
    label_to_idx=label_to_idx,
    num_frames=NUM_FRAMES,
    img_size=IMG_SIZE,
    training=False,     # строго равномерная выборка
)

print(f"Train dataset: {len(train_dataset)}")
print(f"Test dataset: {len(test_dataset)}")


# ============================================================
# 7. DataLoader
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,    # ускоряет передачу данных на GPU
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ============================================================
# 8. Проверка одного батча (перед обучением)
# ============================================================
videos, labels = next(iter(train_loader))
print(f"\nBatch shape: {videos.shape}")
print(f"Batch dtype: {videos.dtype}")
print(f"Batch memory: {videos.element_size() * videos.nelement() / 1024**2:.2f} MB")
print(f"Labels: {labels}")


# ============================================================
# 9. Модель
# ============================================================
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(top_classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"\nУстройство: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")


# ============================================================
# 10. Оптимизатор, функция потерь, scaler для mixed precision
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')   # для mixed precision (FP16)


# ============================================================
# 11. Цикл обучения
# ============================================================
for epoch in range(NUM_EPOCHS):
    
    # --- Обучение ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for videos, labels in train_loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision (новый API)
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
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
          f"Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%")
    
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
    print(f"          Test Acc: {test_acc:.2f}%")

print("\nОбучение завершено.")


# ============================================================
# 12. Сохраняем модель
# ============================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_frames': NUM_FRAMES,
    'img_size': IMG_SIZE,
}, 'r3d18_slovo_20classes.pth')

print("Модель сохранена в r3d18_slovo_20classes.pth")
