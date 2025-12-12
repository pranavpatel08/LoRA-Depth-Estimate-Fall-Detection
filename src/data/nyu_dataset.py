"""
NYU Depth V2 Dataset for depth estimation training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class NYUDepthDataset(Dataset):
    """
    NYU Depth V2 Dataset.
    
    Returns RGB images and corresponding depth maps for training
    depth estimation models.
    """
    
    def __init__(
        self,
        data_path: str = "data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
        split: str = "train",
        transform: Optional[callable] = None,
        target_size: Tuple[int, int] = (480, 640),  # (H, W)
        max_depth: float = 10.0,
        train_ratio: float = 0.9,
    ):
        """
        Args:
            data_path: Path to the .mat file
            split: 'train' or 'val'
            transform: Optional transforms for data augmentation
            target_size: Output size (H, W) for resizing
            max_depth: Maximum depth value for clipping (meters)
            train_ratio: Fraction of data for training
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.max_depth = max_depth
        
        # Load data
        self._load_data(train_ratio)
    
    def _load_data(self, train_ratio: float):
        """Load and split the dataset."""
        with h5py.File(self.data_path, 'r') as f:
            # Based on your inspection: (N, C, W, H) for images, (N, W, H) for depths
            self.images = np.array(f['images'])  # (1449, 3, 640, 480)
            self.depths = np.array(f['depths'])  # (1449, 640, 480)
        
        n_samples = len(self.images)
        n_train = int(n_samples * train_ratio)
        
        # Fixed split for reproducibility
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        if self.split == "train":
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]
        
        print(f"NYU Depth V2 [{self.split}]: {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        
        # Get RGB: (3, W, H) -> (H, W, 3)
        rgb = self.images[real_idx]
        rgb = np.transpose(rgb, (2, 1, 0))  # (H, W, 3)
        
        # Get Depth: (W, H) -> (H, W)
        depth = self.depths[real_idx]
        depth = np.transpose(depth, (1, 0))  # (H, W)
        
        # Resize if needed
        if self.target_size != (rgb.shape[0], rgb.shape[1]):
            rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))
            depth = cv2.resize(depth, (self.target_size[1], self.target_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations
        if self.transform:
            rgb, depth = self.transform(rgb, depth)
        
        # Clip and normalize depth
        depth = np.clip(depth, 0, self.max_depth)
        
        # Create valid mask (NYU has some invalid regions)
        valid_mask = (depth > 0) & (depth < self.max_depth)
        
        # Convert to tensors
        # RGB: (H, W, 3) -> (3, H, W), normalized to [0, 1]
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        # Depth: (H, W) -> (1, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Valid mask: (H, W) -> (1, H, W)
        valid_mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float()
        
        return {
            "rgb": rgb_tensor,           # (3, H, W)
            "depth": depth_tensor,       # (1, H, W)
            "valid_mask": valid_mask_tensor,  # (1, H, W)
        }


class NYUDepthTransform:
    """Data augmentation for RGB-D pairs."""
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def __call__(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if not self.train:
            return rgb, depth
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
        
        # Random brightness/contrast (RGB only)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-20, 20)    # brightness
            rgb = np.clip(alpha * rgb + beta, 0, 255).astype(np.uint8)
        
        return rgb, depth


def get_nyu_dataloaders(
    data_path: str = "data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    batch_size: int = 8,
    target_size: Tuple[int, int] = (480, 640),
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders for NYU Depth V2.
    """
    train_dataset = NYUDepthDataset(
        data_path=data_path,
        split="train",
        transform=NYUDepthTransform(train=True),
        target_size=target_size,
    )
    
    val_dataset = NYUDepthDataset(
        data_path=data_path,
        split="val",
        transform=NYUDepthTransform(train=False),
        target_size=target_size,
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
    
    return train_loader, val_loader


# Quick test
if __name__ == "__main__":
    print("Testing NYU Depth DataLoader...")
    
    train_loader, val_loader = get_nyu_dataloaders(
        batch_size=4,
        target_size=(480, 640),
        num_workers=0,  # 0 for debugging
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  RGB shape: {batch['rgb'].shape}")
    print(f"  RGB range: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
    print(f"  Depth shape: {batch['depth'].shape}")
    print(f"  Depth range: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]m")
    print(f"  Valid mask shape: {batch['valid_mask'].shape}")
    print(f"  Valid ratio: {batch['valid_mask'].mean():.2%}")
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\n✓ NYU DataLoader test passed!")