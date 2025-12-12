"""
UR Fall Detection Dataset for temporal fall detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from collections import defaultdict


class URFallDataset(Dataset):
    """
    UR Fall Detection Dataset with temporal windowing.
    
    Returns sequences of frames for fall detection training.
    Supports both RGB and real/estimated depth inputs.
    """
    
    def __init__(
        self,
        data_path: str = "data/ur_fall",
        split: str = "train",
        window_size: int = 16,
        window_stride: int = 8,
        target_size: Tuple[int, int] = (224, 224),
        modality: str = "depth",
        depth_source: str = "real",  # "real", "zeroshot", or "lora"
        transform: Optional[callable] = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        """
        Args:
            data_path: Path to UR Fall dataset root
            split: 'train', 'val', or 'test'
            window_size: Number of frames per temporal window
            window_stride: Stride for sliding window
            target_size: Output frame size (H, W)
            modality: Which modality to return ("rgb", "depth", or "both")
            depth_source: Which depth to use ("real", "zeroshot", or "lora")
            transform: Optional augmentations
            split_seed: Random seed for reproducible splits
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
        """
        self.data_path = Path(data_path)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.target_size = target_size
        self.modality = modality
        self.depth_source = depth_source
        self.transform = transform
        
        # Mapping for depth directories - must be set before _discover_sequences
        self.depth_dir_map = {
            "real": "depth",
            "zeroshot": "depth_zeroshot",
            "lora": "depth_lora",
        }
        
        # Discover and split sequences
        self.sequences = self._discover_sequences()
        self.split_sequences = self._split_sequences(
            split_seed, train_ratio, val_ratio
        )
        
        # Build window index
        self.windows = self._build_windows()
        
        print(f"URFall [{split}] (depth={depth_source}): {len(self.split_sequences)} sequences, "
              f"{len(self.windows)} windows (size={window_size}, stride={window_stride})")
    
    def _discover_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Discover all sequences and their metadata."""
        sequences = {}
        
        depth_subdir = self.depth_dir_map.get(self.depth_source, "depth")
        
        for seq_dir in sorted(self.data_path.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name == "zips":
                continue
            
            # Filter cam1 for fair comparison
            if "cam1" in seq_dir.name:
                continue
            
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / depth_subdir
            
            if not (rgb_dir.exists() and depth_dir.exists()):
                continue
            
            rgb_frames = sorted(rgb_dir.glob("*.png"))
            depth_frames = sorted(depth_dir.glob("*.png"))
            
            if not (rgb_frames and depth_frames):
                continue
            
            is_fall = seq_dir.name.startswith("fall")
            
            sequences[seq_dir.name] = {
                "path": seq_dir,
                "rgb_frames": rgb_frames,
                "depth_frames": depth_frames,
                "n_frames": min(len(rgb_frames), len(depth_frames)),
                "label": 1 if is_fall else 0,
                "label_name": "fall" if is_fall else "adl",
            }
        
        return sequences
    
    def _split_sequences(
        self, 
        seed: int, 
        train_ratio: float, 
        val_ratio: float
    ) -> List[str]:
        """Split sequences into train/val/test by sequence ID."""
        
        # Group by base sequence to keep different camera views together
        base_sequences = defaultdict(list)
        for seq_name in self.sequences:
            parts = seq_name.rsplit("-cam", 1)
            base = parts[0]
            base_sequences[base].append(seq_name)
        
        # Split base sequences
        bases = sorted(base_sequences.keys())
        np.random.seed(seed)
        np.random.shuffle(bases)
        
        n_train = int(len(bases) * train_ratio)
        n_val = int(len(bases) * val_ratio)
        
        if self.split == "train":
            selected_bases = bases[:n_train]
        elif self.split == "val":
            selected_bases = bases[n_train:n_train + n_val]
        else:  # test
            selected_bases = bases[n_train + n_val:]
        
        # Expand back to full sequence names
        selected = []
        for base in selected_bases:
            selected.extend(base_sequences[base])
        
        return sorted(selected)
    
    def _build_windows(self) -> List[Tuple[str, int]]:
        """Build list of (sequence_name, start_frame) for all windows."""
        windows = []
        
        for seq_name in self.split_sequences:
            seq_info = self.sequences[seq_name]
            n_frames = seq_info["n_frames"]
            
            # Sliding window
            start = 0
            while start + self.window_size <= n_frames:
                windows.append((seq_name, start))
                start += self.window_stride
            
            # Include final window if sequence doesn't divide evenly
            if n_frames >= self.window_size:
                final_start = n_frames - self.window_size
                if (seq_name, final_start) not in windows:
                    windows.append((seq_name, final_start))
        
        return windows
    
    def _load_frame(
        self, 
        frame_path: Path, 
        is_depth: bool = False
    ) -> np.ndarray:
        """Load and preprocess a single frame."""
        if is_depth:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            # Normalize uint16 depth to [0, 1]
            frame = frame.astype(np.float32) / 65535.0
        else:
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
        
        # Resize
        frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        return frame
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, start_frame = self.windows[idx]
        seq_info = self.sequences[seq_name]
        
        frames_rgb = []
        frames_depth = []
        
        for i in range(self.window_size):
            frame_idx = start_frame + i
            
            if self.modality in ("rgb", "both"):
                rgb = self._load_frame(seq_info["rgb_frames"][frame_idx], is_depth=False)
                frames_rgb.append(rgb)
            
            if self.modality in ("depth", "both"):
                depth = self._load_frame(seq_info["depth_frames"][frame_idx], is_depth=True)
                frames_depth.append(depth)
        
        result = {
            "label": torch.tensor(seq_info["label"], dtype=torch.long),
            "sequence": seq_name,
            "start_frame": start_frame,
        }
        
        if frames_rgb:
            rgb_stack = np.stack(frames_rgb, axis=0)
            rgb_tensor = torch.from_numpy(rgb_stack).permute(0, 3, 1, 2).float()
            result["rgb"] = rgb_tensor
        
        if frames_depth:
            depth_stack = np.stack(frames_depth, axis=0)
            depth_tensor = torch.from_numpy(depth_stack).unsqueeze(1).float()
            result["depth"] = depth_tensor
        
        return result


def get_urfall_dataloaders(
    data_path: str = "data/ur_fall",
    batch_size: int = 8,
    window_size: int = 16,
    window_stride: int = 8,
    target_size: Tuple[int, int] = (224, 224),
    modality: str = "depth",
    depth_source: str = "real",
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders for UR Fall."""
    
    train_dataset = URFallDataset(
        data_path=data_path,
        split="train",
        window_size=window_size,
        window_stride=window_stride,
        target_size=target_size,
        modality=modality,
        depth_source=depth_source,
    )
    
    val_dataset = URFallDataset(
        data_path=data_path,
        split="val",
        window_size=window_size,
        window_stride=window_stride,
        target_size=target_size,
        modality=modality,
        depth_source=depth_source,
    )
    
    test_dataset = URFallDataset(
        data_path=data_path,
        split="test",
        window_size=window_size,
        window_stride=window_stride,
        target_size=target_size,
        modality=modality,
        depth_source=depth_source,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# Quick test
if __name__ == "__main__":
    print("Testing UR Fall DataLoader with different depth sources...")
    print("=" * 60)
    
    for depth_source in ["real", "zeroshot", "lora"]:
        print(f"\n--- Testing depth_source='{depth_source}' ---")
        try:
            train_loader, val_loader, test_loader = get_urfall_dataloaders(
                batch_size=4,
                window_size=16,
                window_stride=8,
                target_size=(224, 224),
                modality="depth",
                depth_source=depth_source,
                num_workers=0,
            )
            
            batch = next(iter(train_loader))
            print(f"  Depth shape: {batch['depth'].shape}")
            print(f"  Labels: {batch['label'].tolist()}")
            print(f"  ✓ Success")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")