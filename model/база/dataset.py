import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SlovoDataset(Dataset):
    """
    Датасет для Slovo (русский жестовый язык).
    
    Читает видео, сэмплирует num_frames кадров,
    возвращает тензор (C, T, H, W) и метку класса.
    
    Параметры:
        df          — DataFrame с колонками attachment_id, text, train
        root_dir    — папка с видео (train/ или test/)
        label_to_idx — словарь {текст: индекс}
        num_frames  — сколько кадров сэмплировать (по умолчанию 16)
        img_size    — размер стороны кадра (по умолчанию 112)
        training    — режим обучения (включает случайное смещение начала)
    """
    
    # Статистики Kinetics-400 (на которых обучался R3D-18)
    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    
    def __init__(self, df, root_dir, label_to_idx, num_frames=16,
                 img_size=112, training=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.label_to_idx = label_to_idx
        self.num_frames = num_frames
        self.img_size = img_size
        self.training = training
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        attachment_id = row['attachment_id']
        text = row['text']
        
        # Ищем видеофайл (Slovo хранит видео как .mp4)
        video_path = os.path.join(self.root_dir, f"{attachment_id}.mp4")
        
        # На всякий случай проверяем другие расширения
        if not os.path.exists(video_path):
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidate = os.path.join(self.root_dir, f"{attachment_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
        
        # Загружаем видео (только нужные кадры)
        frames = self._load_video(video_path)
        
        # (T, H, W, C) -> (C, T, H, W), нормализуем в [0, 1]
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        # Нормализация по статистикам Kinetics-400
        video_tensor = (video_tensor - self.MEAN) / self.STD
        
        # Метка класса
        label = self.label_to_idx[text]
        
        return video_tensor, label
    
    def _load_video(self, video_path):
        """
        Загружает видео и сэмплирует num_frames кадров.
        
        Читает только нужные кадры (не всё видео целиком), что 
        ускоряет загрузку в 5-10 раз.
        """
        size = self.img_size
        cap = cv2.VideoCapture(video_path)
        
        # Если видео не открылось — возвращаем чёрные кадры
        if not cap.isOpened():
            return np.zeros((self.num_frames, size, size, 3), dtype=np.uint8)
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total == 0:
            cap.release()
            return np.zeros((self.num_frames, size, size, 3), dtype=np.uint8)
        
        # Определяем, какие кадры нам нужны
        if self.training and total > self.num_frames:
            # Случайное смещение начала (аугментация)
            max_start = total - self.num_frames
            start = np.random.randint(0, max_start + 1)
            indices = set(range(start, start + self.num_frames))
        else:
            # Равномерная выборка (для теста и коротких видео)
            indices = set(
                np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()
            )
        
        # Читаем только нужные кадры
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                # OpenCV читает в BGR, переводим в RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size))
                frames.append(frame)
            frame_idx += 1
        
        cap.release()
        
        # Если кадров не хватило — дополняем последним
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((size, size, 3), dtype=np.uint8))
        
        return np.array(frames[:self.num_frames])
