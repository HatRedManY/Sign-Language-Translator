# dataset_r3d.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SlovoDatasetR3D(Dataset):
    MEAN = None
    STD = None
    
    def __init__(self, df, root_dir, label_to_idx, num_frames=16,
                 img_size=112, training=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.label_to_idx = label_to_idx
        self.num_frames = num_frames
        self.img_size = img_size
        self.training = training
        
        if SlovoDatasetR3D.MEAN is None:
            SlovoDatasetR3D.MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
            SlovoDatasetR3D.STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        attachment_id = row['attachment_id']
        text = row['text']
        video_path = os.path.join(self.root_dir, f"{attachment_id}.mp4")
        
        if not os.path.exists(video_path):
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidate = os.path.join(self.root_dir, f"{attachment_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
        
        frames = self._load_video(video_path)
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        mean = self.MEAN.to(video_tensor.device)
        std = self.STD.to(video_tensor.device)
        video_tensor = (video_tensor - mean) / std
        
        label = self.label_to_idx[text]
        return video_tensor, label
    
    def _load_video(self, video_path):
        size = self.img_size
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return np.zeros((self.num_frames, size, size, 3), dtype=np.uint8)
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return np.zeros((self.num_frames, size, size, 3), dtype=np.uint8)
        
        indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int).tolist())
        
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- Правильный препроцессинг для R3D-18: Resize + CenterCrop ---
                h, w = frame.shape[:2]
                if h < w:
                    new_h = 128
                    new_w = int(w * (128 / h))
                else:
                    new_h = int(h * (128 / w))
                    new_w = 128
                
                frame = cv2.resize(frame, (new_w, new_h))
                start_x = new_w // 2 - (size // 2)
                start_y = new_h // 2 - (size // 2)
                frame = frame[start_y:start_y + size, start_x:start_x + size]
                # -----------------------------------------------------------------
                
                frames.append(frame)
            frame_idx += 1
        
        cap.release()
        
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((size, size, 3), dtype=np.uint8))
        
        frames = np.array(frames[:self.num_frames])
        
        if self.training:
            frames = self._augment(frames)
        
        return frames

    def _augment(self, frames):
        # Для R3D аугментация может быть чуть проще, но оставим базовую
        h, w = frames.shape[1], frames.shape[2]
        if np.random.random() < 0.5:
            shift_x = np.random.randint(-w // 10, w // 10 + 1)
            shift_y = np.random.randint(-h // 10, h // 10 + 1)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            frames = np.array([cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE) for f in frames])
        
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return frames
