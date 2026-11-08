# predict_r3d.py
import os
import sys
import argparse

# Задаем TORCH_HOME до импорта torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_HOME"] = os.path.join(SCRIPT_DIR, "torch_cache")

import cv2
import numpy as np
import torch
from torchvision.models.video import r3d_18

# ============================================================
# Настройки
# ============================================================
# Указываем путь к вашему файлу R3D-18
CKPT_PATH = os.path.join(SCRIPT_DIR, "r3d18_slovo_100words_best.pth") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Статистики Kinetics-400
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Файл модели не найден: {ckpt_path}")
        
    print(f"Загрузка модели: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]
    num_frames = ckpt.get("num_frames", 16)
    img_size = ckpt.get("img_size", 112)
    
    print(f"Классов: {len(label_to_idx)}, Кадров: {num_frames}, Размер: {img_size}x{img_size}")
    print(f"Точность модели при обучении: {ckpt.get('test_acc', 'неизвестно')}%")
    
    model = r3d_18(weights=None)
    
    # ИСПРАВЛЕНИЕ: Голова должна быть простой Linear (без Sequential и Dropout),
    # так как в чекпоинте ключи 'fc.weight' и 'fc.bias'.
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_to_idx))
    
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    return model, idx_to_label, num_frames, img_size


def load_video(path, num_frames, img_size):
    """Загружает видео и применяет Resize + CenterCrop (как при обучении R3D)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Видео пустое или не читается: {path}")
    
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- ИСПРАВЛЕНИЕ: Resize + CenterCrop ---
            h, w = frame.shape[:2]
            if h < w:
                new_h = 128
                new_w = int(w * (128 / h))
            else:
                new_h = int(h * (128 / w))
                new_w = 128

            frame = cv2.resize(frame, (new_w, new_h))
            start_x = new_w // 2 - (img_size // 2)
            start_y = new_h // 2 - (img_size // 2)
            frame = frame[start_y:start_y + img_size, start_x:start_x + img_size]
            # ------------------------------------------
            
            frames.append(frame)
        frame_idx += 1
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    frames = np.array(frames[:num_frames])
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)


def predict(video_path, top_k=5):
    model, idx_to_label, num_frames, img_size = load_model(CKPT_PATH)
    
    print(f"\nЗагрузка видео: {video_path}")
    video_tensor = load_video(video_path, num_frames, img_size).to(DEVICE)
    
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(video_tensor)
        else:
            outputs = model(video_tensor)
            
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)
    
    print("\n" + "="*40)
    print(f"Топ-{top_k} предсказаний (R3D-18):")
    print("="*40)
    
    for prob, idx in zip(top_probs[0], top_indices[0]):
        label = idx_to_label[idx.item()]
        print(f"  {label:20s} — {prob.item()*100:6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предсказание жеста по видео (R3D-18)")
    parser.add_argument("--video", type=str, required=True, help="Путь к видеофайлу")
    parser.add_argument("--top_k", type=int, default=5, help="Сколько классов показать")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"[!] Видео не найдено: {args.video}")
        sys.exit(1)
        
    predict(args.video, args.top_k)
