import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.video import r3d_18


# ============================================================
# 1. Загружаем сохранённую модель
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('r3d18_slovo_20classes.pth', map_location=device)

label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
num_frames = checkpoint['num_frames']
img_size = checkpoint['img_size']

print(f"Классов: {len(label_to_idx)}")
print(f"Классы: {list(label_to_idx.keys())}")
print(f"num_frames: {num_frames}, img_size: {img_size}")

model = r3d_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("Модель загружена.")


# ============================================================
# 2. Функция загрузки видео (нормализация на CPU)
# ============================================================
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        cap.release()
        return None
    
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    frames = np.array(frames[:num_frames])
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    tensor = (tensor - MEAN) / STD   # нормализация на CPU
    return tensor.unsqueeze(0)       # (1, C, T, H, W) — на CPU


# ============================================================
# 3. Предсказание
# ============================================================
def predict(video_path, top_k=3):
    video_tensor = load_video(video_path)
    
    if video_tensor is None:
        return None
    
    video_tensor = video_tensor.to(device)   # ← переносим весь батч на GPU
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = model(video_tensor)
        
        probs = torch.softmax(outputs, dim=1)[0]
    
    top_probs, top_indices = probs.topk(top_k)
    
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        results.append((idx_to_label[idx], float(prob)))
    
    return results


# ============================================================
# 4. Тестируем на 10 случайных видео из test
# ============================================================
df = pd.read_csv('annotations.csv', encoding='utf-8', sep='\t')
df = df[df['text'] != 'no_event']
top_classes = df['text'].value_counts().head(20).index.tolist()
df_test = df[(df['text'].isin(top_classes)) & (df['train'] == False)]

print("\n=== Проверка на 10 случайных тестовых видео ===\n")

correct = 0
total = 0

for i in range(min(100, len(df_test))):
    row = df_test.iloc[i]
    video_path = os.path.join('test', f"{row['attachment_id']}.mp4")
    
    if not os.path.exists(video_path):
        continue
    
    true_label = row['text']
    predictions = predict(video_path)
    
    if predictions is None:
        continue
    
    pred_label, pred_prob = predictions[0]
    is_correct = pred_label == true_label
    correct += is_correct
    total += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{status} Истина: {true_label} | Предсказано: {pred_label} ({pred_prob:.2%})")
    print(f"   Топ-3: {[(l, f'{p:.1%}') for l, p in predictions]}")
    print()

if total > 0:
    print(f"Точность на {total} примерах: {correct}/{total} = {100*correct/total:.1f}%")
