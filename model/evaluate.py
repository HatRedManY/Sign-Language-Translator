# evaluate.py
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_HOME"] = os.path.join(SCRIPT_DIR, "torch_cache")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, SCRIPT_DIR)
from dataset import SlovoDataset

# ============================================================
# Настройки (должны совпадать с train.py)
# ============================================================
ANNOTATIONS_PATH = os.path.join(SCRIPT_DIR, "annotations.csv")
TEST_DIR = os.path.join(SCRIPT_DIR, "test")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
CKPT_PATH = os.path.join(RUNS_DIR, "mvitv2_slovo_100words_best.pth")

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path, num_classes):
    model = mvit_v2_s(weights=None)
    in_features = model.head[1].in_features
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features, num_classes)
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    if not os.path.exists(CKPT_PATH):
        print(f"[!] Файл не найден: {CKPT_PATH}")
        sys.exit(1)

    # 1. Загрузка данных (повторяем логику train.py для test_df)
    df = pd.read_csv(ANNOTATIONS_PATH, encoding='utf-8', sep='\t')
    df = df[df['text'] != 'no_event'].copy()
    letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
    df_words = df[~df['text'].isin(letters)].copy()
    
    # Восстанавливаем словарь классов из чекпоинта
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]
    
    # Берем только тестовые видео, которые есть в словаре классов
    test_df = df_words[df_words['train'] == False]
    test_df = test_df[test_df['text'].isin(label_to_idx.keys())]

    print(f"Тестовых видео: {len(test_df)}")
    print(f"Классов: {len(label_to_idx)}")

    test_dataset = SlovoDataset(
        df=test_df, root_dir=TEST_DIR, label_to_idx=label_to_idx,
        num_frames=NUM_FRAMES, img_size=IMG_SIZE, training=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # 2. Загрузка модели
    print(f"Загрузка модели: {CKPT_PATH}")
    model, ckpt = load_model(CKPT_PATH, len(label_to_idx))
    print(f"Лучшая точность (из чекпоинта): {ckpt.get('test_acc', 0):.2f}%")

    # 3. Прогон по тесту
    all_preds = []
    all_labels = []

    print("Инференс...")
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(videos)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Метрики
    acc = 100 * (all_preds == all_labels).mean()
    print(f"\nТекущая точность на тесте: {acc:.2f}%\n")

    # 5. Топ-5 проблемных классов
    print("="*50)
    print("Топ-5 классов с наибольшим числом ошибок:")
    print("="*50)
    
    cm = confusion_matrix(all_labels, all_preds)
    # Считаем ошибки для каждого класса: сумма по строке минус диагональ
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    # Сортируем индексы по убыванию ошибок
    worst_classes = np.argsort(errors_per_class)[::-1][:5]
    
    for idx in worst_classes:
        if errors_per_class[idx] == 0:
            continue
        class_name = idx_to_label[idx]
        total = cm.sum(axis=1)[idx]
        correct = cm[idx, idx]
        print(f"  Класс '{class_name}': {errors_per_class[idx]} ошибок из {total} (верно {correct})")

    # 6. Построение Confusion Matrix
    # Матрица 100x100 слишком большая для вывода на экран,
    # поэтому сохраняем в файл как heatmap.
    print("\nПостроение Confusion Matrix...")
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix (Acc: {acc:.2f}%)")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    
    # Подписываем оси названиями классов (мелко, но информативно)
    labels_names = [idx_to_label[i] for i in range(len(label_to_idx))]
    plt.xticks(np.arange(len(labels_names)) + 0.5, labels_names, rotation=90, fontsize=4)
    plt.yticks(np.arange(len(labels_names)) + 0.5, labels_names, rotation=0, fontsize=4)
    
    plt.tight_layout()
    save_path = os.path.join(RUNS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Сохранено: {save_path}")
    print("\nГотово.")
