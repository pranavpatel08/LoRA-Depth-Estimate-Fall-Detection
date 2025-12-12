"""
Le2i Fall Detection Dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from collections import defaultdict


class Le2iDataset(Dataset):
    """
    Le2i Fall Detection Dataset with temporal windowing.
    
    Supports RGB and estimated depth (no real depth available).
    """
    
    def __init__(
        self,
        data_path: str = "data/le2i/processed",
        split: str = "train",
        window_size: int = 16,
        window_stride: int = 8,
        target_size: Tuple[int, int] = (224, 224),
        modality: str = "rgb",  # "rgb", "depth_zeroshot", "depth_lora"
        transform: Optional[callable] = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.target_size = target_size
        self.modality = modality
        self.transform = transform
        
        # Map modality to directory
        self.modality_dir_map = {
            "rgb": "rgb",
            "depth_zeroshot": "depth_zeroshot",
            "depth_lora": "depth_lora",
        }
        
        # Discover and split sequences
        self.sequences = self._discover_sequences()
        self.split_sequences = self._split_sequences(
            split_seed, train_ratio, val_ratio
        )
        
        # Build window index
        self.windows = self._build_windows()
        
        print(f"Le2i [{split}] ({modality}): {len(self.split_sequences)} sequences, "
              f"{len(self.windows)} windows")
    
    def _discover_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Discover all sequences."""
        sequences = {}
        
        frame_dir = self.modality_dir_map.get(self.modality, "rgb")
        
        for seq_dir in sorted(self.data_path.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            frames_dir = seq_dir / frame_dir
            
            if not frames_dir.exists():
                continue
            
            frames = sorted(frames_dir.glob("*.png"))
            
            if len(frames) < self.window_size:
                continue
            
            # Label from directory name
            is_fall = seq_dir.name.startswith("fall")
            
            sequences[seq_dir.name] = {
                "path": seq_dir,
                "frames": frames,
                "n_frames": len(frames),
                "label": 1 if is_fall else 0,
            }
        
        return sequences
    
    def _split_sequences(self, seed: int, train_ratio: float, val_ratio: float) -> List[str]:
        """Split sequences into train/val/test."""
        seq_names = sorted(self.sequences.keys())
        
        np.random.seed(seed)
        np.random.shuffle(seq_names)
        
        n_train = int(len(seq_names) * train_ratio)
        n_val = int(len(seq_names) * val_ratio)
        
        if self.split == "train":
            return seq_names[:n_train]
        elif self.split == "val":
            return seq_names[n_train:n_train + n_val]
        else:
            return seq_names[n_train + n_val:]
    
    def _build_windows(self) -> List[Tuple[str, int]]:
        """Build sliding window indices."""
        windows = []
        
        for seq_name in self.split_sequences:
            seq_info = self.sequences[seq_name]
            n_frames = seq_info["n_frames"]
            
            start = 0
            while start + self.window_size <= n_frames:
                windows.append((seq_name, start))
                start += self.window_stride
            
            # Final window
            if n_frames >= self.window_size:
                final_start = n_frames - self.window_size
                if (seq_name, final_start) not in windows:
                    windows.append((seq_name, final_start))
        
        return windows
    
    def _load_frame(self, frame_path: Path) -> np.ndarray:
        """Load and preprocess a frame."""
        is_depth = "depth" in self.modality
        
        if is_depth:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            frame = frame.astype(np.float32) / 65535.0
        else:
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
        
        frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        return frame
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, start_frame = self.windows[idx]
        seq_info = self.sequences[seq_name]
        
        frames = []
        for i in range(self.window_size):
            frame = self._load_frame(seq_info["frames"][start_frame + i])
            frames.append(frame)
        
        frames_stack = np.stack(frames, axis=0)
        
        if "depth" in self.modality:
            # (T, H, W) -> (T, 1, H, W)
            tensor = torch.from_numpy(frames_stack).unsqueeze(1).float()
            key = "depth"
        else:
            # (T, H, W, 3) -> (T, 3, H, W)
            tensor = torch.from_numpy(frames_stack).permute(0, 3, 1, 2).float()
            key = "rgb"
        
        return {
            key: tensor,
            "label": torch.tensor(seq_info["label"], dtype=torch.long),
            "sequence": seq_name,
        }


def get_le2i_dataloaders(
    data_path: str = "data/le2i/processed",
    batch_size: int = 8,
    window_size: int = 16,
    window_stride: int = 8,
    target_size: Tuple[int, int] = (224, 224),
    modality: str = "rgb",
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for Le2i dataset."""
    
    train_dataset = Le2iDataset(
        data_path=data_path, split="train", window_size=window_size,
        window_stride=window_stride, target_size=target_size, modality=modality,
    )
    
    val_dataset = Le2iDataset(
        data_path=data_path, split="val", window_size=window_size,
        window_stride=window_stride, target_size=target_size, modality=modality,
    )
    
    test_dataset = Le2iDataset(
        data_path=data_path, split="test", window_size=window_size,
        window_stride=window_stride, target_size=target_size, modality=modality,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing Le2i DataLoader...")
    
    try:
        train_loader, val_loader, test_loader = get_le2i_dataloaders(
            modality="rgb",
            num_workers=0,
        )
        
        batch = next(iter(train_loader))
        print(f"RGB shape: {batch['rgb'].shape}")
        print(f"Labels: {batch['label'].tolist()}")
        print("✓ Success")
    except Exception as e:
        print(f"Error: {e}")