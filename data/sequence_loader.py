import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SequenceBufferDataset(Dataset):
    """
    Dataset loader for Fire and Ash Temporal Path.
    Loads 16-frame buffers from video files or directories of frames.
    Outputs tensors of shape (Channels, Time, Height, Width) for the 3D CNN.
    """
    def __init__(self, data_dir: str, sequence_length: int = 16, transform=None):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Determine classes from subdirectory names
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = self._gather_video_clips()

    def _gather_video_clips(self):
        """
        Scans data_dir for video files or frame sequences.
        For simplicity, this assumes each class folder contains .mp4 files.
        """
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            for file in os.listdir(cls_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    samples.append((os.path.join(cls_dir, file), self.class_to_idx[cls_name]))
        return samples

    def _extract_frames(self, video_path: str):
        """
        Extracts `sequence_length` frames from a video file.
        Uses evenly spaced frames if the video is longer than `sequence_length`.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Fallback if cv2 fails to get frame count
        if total_frames <= 0:
            total_frames = self.sequence_length
            
        frames = []
        if total_frames >= self.sequence_length:
            # Pick evenly spaced frames
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # CRITICAL RAM SAVER: Resize immediately before storing
                    frame = cv2.resize(frame, (64, 64))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        else:
            # Not enough frames, read what we can (up to sequence_length)
            for _ in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (64, 64))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
        cap.release()
        
        # Pad by duplicating the last frame if we didn't get enough
        if len(frames) > 0:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
        else:
            # Failsafe for entirely broken videos
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(self.sequence_length)]
            
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        
        # Apply spatial transformations (e.g., resizing) to each frame
        if self.transform:
            import PIL.Image
            frames = [self.transform(PIL.Image.fromarray(f)) for f in frames]
        else:
            # Default transformation: Resize to 64x64 and convert to tensor
            default_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
            frames = [default_transform(f) for f in frames]

        # Stack frames along the Time dimension
        # Individual frame tensor: (C, H, W) -> Stacked: (Time, C, H, W)
        sequence_tensor = torch.stack(frames)
        
        # Permute to (C, Time, H, W) as expected by PyTorch Conv3d
        sequence_tensor = sequence_tensor.permute(1, 0, 2, 3)
        
        return sequence_tensor, label

if __name__ == "__main__":
    # Example usage
    pass
