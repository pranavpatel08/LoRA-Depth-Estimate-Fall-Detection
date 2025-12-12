"""
UP-Fall Detection Dataset for temporal fall detection.

Source: https://sites.google.com/up.edu.mx/har-up/

Structure: 
  - 17 Subjects
  - 11 Activities (5 Falls, 6 ADLs)
  - 3 Trials per activity
  - 2 Camera angles (Front, Side)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from collections import defaultdict


class UPFallDataset(Dataset):
    """
    UP-Fall Detection Dataset with temporal windowing.
    
    Key features:
    - Temporal trimming for fall sequences (skip initial standing phase)
    - Activity 11 (Laying) included as ADL to require temporal modeling
    - Subject-based splits to prevent data leakage
    """
    
    # Class constants
    FALL_ACTIVITIES = {1, 2, 3, 4, 5}  # Activities 1-5 are falls
    ADL_ACTIVITIES = {6, 7, 8, 9, 10, 11}  # Activities 6-11 are ADLs (11 = laying)
    
    # Known missing data
    MISSING_TRIALS = {
        (8, 11, 2),  # Subject 8, Activity 11, Trial 2
        (8, 11, 3),  # Subject 8, Activity 11, Trial 3
    }
    
    def __init__(
        self,
        data_path: str = "data/up_fall",
        split: str = "train",
        window_size: int = 16,
        window_stride: int = 8,
        target_size: Tuple[int, int] = (224, 224),
        modality: str = "depth",
        depth_source: str = "lora",  # "lora" or "zeroshot"
        transform: Optional[callable] = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        fall_trim_start: float = 0.25,  # Skip first 25% of fall sequences
        camera: int = 1,  # Which camera to use (1 = side view)
    ):
        """
        Args:
            data_path: Path to UP-Fall dataset root
            split: 'train', 'val', or 'test'
            window_size: Number of frames per temporal window
            window_stride: Stride for sliding window
            target_size: Output frame size (H, W)
            modality: Which modality to return ("rgb", "depth", or "both")
            depth_source: Which estimated depth to use ("lora" or "zeroshot")
            transform: Optional augmentations
            split_seed: Random seed for reproducible splits
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            fall_trim_start: Fraction of fall sequences to skip from start
            camera: Camera view to use (1 = side view recommended)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.target_size = target_size
        self.modality = modality
        self.depth_source = depth_source
        self.transform = transform
        self.fall_trim_start = fall_trim_start
        self.camera = camera
        
        # Depth directory mapping
        self.depth_dir_map = {
            "lora": "depth_lora",
            "zeroshot": "depth_zeroshot",
        }
        
        # Discover and split sequences
        self.sequences = self._discover_sequences()
        self.split_sequences = self._split_by_subject(
            split_seed, train_ratio, val_ratio
        )
        
        # Build window index with trimming applied
        self.windows = self._build_windows()
        
        # Print stats
        n_fall = sum(1 for s in self.split_sequences if self.sequences[s]["label"] == 1)
        n_adl = len(self.split_sequences) - n_fall
        print(f"UPFall [{split}] (depth={depth_source}): "
              f"{len(self.split_sequences)} sequences ({n_fall} fall, {n_adl} ADL), "
              f"{len(self.windows)} windows (size={window_size}, stride={window_stride})")
    
    def _parse_sequence_path(self, path: Path) -> Optional[Tuple[int, int, int]]:
        """
        Parse subject, activity, trial from path.
        Returns None if parsing fails.
        
        Expected structure: .../SubjectX/ActivityY/TrialZ/CameraW/
        """
        try:
            # Navigate up from camera folder
            camera_dir = path
            trial_dir = camera_dir.parent
            activity_dir = trial_dir.parent
            subject_dir = activity_dir.parent
            
            subject = int(subject_dir.name.replace("Subject", ""))
            activity = int(activity_dir.name.replace("Activity", ""))
            trial = int(trial_dir.name.replace("Trial", ""))
            
            return subject, activity, trial
        except (ValueError, AttributeError):
            return None
    
    def _discover_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Discover all valid sequences with RGB frames."""
        sequences = {}
        
        depth_subdir = self.depth_dir_map.get(self.depth_source, "depth_lora")
        camera_name = f"Camera{self.camera}"
        
        # Iterate through subject directories
        for subject_dir in sorted(self.data_path.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("Subject"):
                continue
            
            for activity_dir in sorted(subject_dir.iterdir()):
                if not activity_dir.is_dir() or not activity_dir.name.startswith("Activity"):
                    continue
                
                for trial_dir in sorted(activity_dir.iterdir()):
                    if not trial_dir.is_dir() or not trial_dir.name.startswith("Trial"):
                        continue
                    
                    camera_dir = trial_dir / camera_name
                    if not camera_dir.exists():
                        continue
                    
                    # Parse identifiers
                    parsed = self._parse_sequence_path(camera_dir)
                    if parsed is None:
                        continue
                    
                    subject, activity, trial = parsed
                    
                    # Skip known missing trials
                    if (subject, activity, trial) in self.MISSING_TRIALS:
                        continue
                    
                    # Check for RGB frames
                    # TODO: Verify actual frame naming convention in your dataset
                    # Common patterns: frame_0001.png, rgb_0001.png, 0001.png
                    rgb_frames = sorted(camera_dir.glob("*.png"))
                    if not rgb_frames:
                        # Try jpg if no png
                        rgb_frames = sorted(camera_dir.glob("*.jpg"))
                    
                    if not rgb_frames:
                        continue
                    
                    # Check for depth (will be generated later, but check if exists)
                    depth_dir = camera_dir / depth_subdir
                    depth_frames = []
                    if depth_dir.exists():
                        depth_frames = sorted(depth_dir.glob("*.png"))
                    
                    # Determine label
                    is_fall = activity in self.FALL_ACTIVITIES
                    
                    # Create unique sequence ID
                    seq_id = f"S{subject:02d}_A{activity:02d}_T{trial}_C{self.camera}"
                    
                    sequences[seq_id] = {
                        "path": camera_dir,
                        "subject": subject,
                        "activity": activity,
                        "trial": trial,
                        "rgb_frames": rgb_frames,
                        "depth_frames": depth_frames,
                        "depth_dir": depth_dir,
                        "n_frames": len(rgb_frames),
                        "label": 1 if is_fall else 0,
                        "label_name": "fall" if is_fall else "adl",
                        "is_laying": activity == 11,  # Track for analysis
                    }
        
        print(f"Discovered {len(sequences)} total sequences")
        return sequences
    
    def _split_by_subject(
        self,
        seed: int,
        train_ratio: float,
        val_ratio: float,
    ) -> List[str]:
        """
        Split sequences by SUBJECT to prevent data leakage.
        All sequences from a subject go into the same split.
        """
        # Group sequences by subject
        subjects = defaultdict(list)
        for seq_id, info in self.sequences.items():
            subjects[info["subject"]].append(seq_id)
        
        # Sort and shuffle subjects
        subject_ids = sorted(subjects.keys())
        np.random.seed(seed)
        np.random.shuffle(subject_ids)
        
        # Split subjects
        n_subjects = len(subject_ids)
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        if self.split == "train":
            selected_subjects = subject_ids[:n_train]
        elif self.split == "val":
            selected_subjects = subject_ids[n_train:n_train + n_val]
        else:  # test
            selected_subjects = subject_ids[n_train + n_val:]
        
        # Collect all sequences for selected subjects
        selected = []
        for subj in selected_subjects:
            selected.extend(subjects[subj])
        
        print(f"  Split '{self.split}': subjects {sorted(selected_subjects)} "
              f"({len(selected)} sequences)")
        
        return sorted(selected)
    
    def _get_frame_range(self, seq_info: Dict[str, Any]) -> Tuple[int, int]:
        """
        Get valid frame range for a sequence, applying trimming for falls.
        
        Returns:
            (start_frame, end_frame) - end is exclusive
        """
        n_frames = seq_info["n_frames"]
        
        if seq_info["label"] == 1:  # Fall sequence
            # Skip first fall_trim_start fraction
            start = int(n_frames * self.fall_trim_start)
        else:
            start = 0
        
        return start, n_frames
    
    def _build_windows(self) -> List[Tuple[str, int]]:
        """Build list of (sequence_id, start_frame) for all windows."""
        windows = []
        
        for seq_id in self.split_sequences:
            seq_info = self.sequences[seq_id]
            
            # Get valid frame range (with trimming applied)
            frame_start, frame_end = self._get_frame_range(seq_info)
            valid_frames = frame_end - frame_start
            
            # Skip if not enough frames for a window
            if valid_frames < self.window_size:
                print(f"  Warning: {seq_id} has only {valid_frames} valid frames, "
                      f"need {self.window_size}. Skipping.")
                continue
            
            # Sliding window within valid range
            pos = frame_start
            while pos + self.window_size <= frame_end:
                windows.append((seq_id, pos))
                pos += self.window_stride
            
            # Include final window if needed
            final_start = frame_end - self.window_size
            if final_start >= frame_start and (seq_id, final_start) not in windows:
                windows.append((seq_id, final_start))
        
        return windows
    
    def _load_frame(
        self,
        frame_path: Path,
        is_depth: bool = False,
    ) -> np.ndarray:
        """Load and preprocess a single frame."""
        if is_depth:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise FileNotFoundError(f"Could not load depth: {frame_path}")
            # Handle different bit depths
            if frame.dtype == np.uint16:
                frame = frame.astype(np.float32) / 65535.0
            else:
                frame = frame.astype(np.float32) / 255.0
        else:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Could not load RGB: {frame_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
        
        # Resize
        frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        return frame
    
    def _get_depth_frame_path(
        self,
        seq_info: Dict[str, Any],
        frame_idx: int,
    ) -> Path:
        """Get corresponding depth frame path for an RGB frame index."""
        rgb_path = seq_info["rgb_frames"][frame_idx]
        depth_dir = seq_info["depth_dir"]
        
        # Depth frames should mirror RGB naming
        depth_path = depth_dir / rgb_path.name
        
        return depth_path
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_id, start_frame = self.windows[idx]
        seq_info = self.sequences[seq_id]
        
        frames_rgb = []
        frames_depth = []
        
        for i in range(self.window_size):
            frame_idx = start_frame + i
            
            if self.modality in ("rgb", "both"):
                rgb = self._load_frame(
                    seq_info["rgb_frames"][frame_idx],
                    is_depth=False
                )
                frames_rgb.append(rgb)
            
            if self.modality in ("depth", "both"):
                depth_path = self._get_depth_frame_path(seq_info, frame_idx)
                depth = self._load_frame(depth_path, is_depth=True)
                frames_depth.append(depth)
        
        result = {
            "label": torch.tensor(seq_info["label"], dtype=torch.long),
            "sequence": seq_id,
            "start_frame": start_frame,
            "activity": seq_info["activity"],
            "is_laying": seq_info["is_laying"],
        }
        
        if frames_rgb:
            rgb_stack = np.stack(frames_rgb, axis=0)  # (T, H, W, C)
            rgb_tensor = torch.from_numpy(rgb_stack).permute(0, 3, 1, 2).float()
            result["rgb"] = rgb_tensor  # (T, 3, H, W)
        
        if frames_depth:
            depth_stack = np.stack(frames_depth, axis=0)  # (T, H, W)
            depth_tensor = torch.from_numpy(depth_stack).unsqueeze(1).float()
            result["depth"] = depth_tensor  # (T, 1, H, W)
        
        return result


def get_upfall_dataloaders(
    data_path: str = "data/up_fall",
    batch_size: int = 8,
    window_size: int = 16,
    window_stride: int = 8,
    target_size: Tuple[int, int] = (224, 224),
    modality: str = "depth",
    depth_source: str = "lora",
    num_workers: int = 4,
    fall_trim_start: float = 0.25,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders for UP-Fall."""
    
    common_args = dict(
        data_path=data_path,
        window_size=window_size,
        window_stride=window_stride,
        target_size=target_size,
        modality=modality,
        depth_source=depth_source,
        fall_trim_start=fall_trim_start,
    )
    
    train_dataset = UPFallDataset(split="train", **common_args)
    val_dataset = UPFallDataset(split="val", **common_args)
    test_dataset = UPFallDataset(split="test", **common_args)
    
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


# === Testing / Exploration ===
if __name__ == "__main__":
    print("=" * 60)
    print("UP-Fall Dataset Explorer")
    print("=" * 60)
    
    # Phase 1: Just test sequence discovery (RGB only, no depth needed yet)
    print("\n[Phase 1] Testing sequence discovery with RGB...")
    
    try:
        dataset = UPFallDataset(
            data_path="data/up_fall",
            split="train",
            modality="rgb",  # RGB only for initial test
            window_size=16,
            window_stride=8,
            fall_trim_start=0.25,
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Total windows: {len(dataset)}")
        
        # Show sequence distribution
        print(f"\n[Sequence Distribution]")
        activities = defaultdict(int)
        for seq_id in dataset.split_sequences:
            act = dataset.sequences[seq_id]["activity"]
            activities[act] += 1
        
        for act in sorted(activities.keys()):
            label = "Fall" if act <= 5 else "ADL"
            laying = " (LAYING)" if act == 11 else ""
            print(f"  Activity {act:2d} ({label}{laying}): {activities[act]} sequences")
        
        # Test loading a single sample
        print(f"\n[Test Loading Sample]")
        sample = dataset[0]
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Label: {sample['label'].item()} ({dataset.sequences[sample['sequence']]['label_name']})")
        print(f"  Sequence: {sample['sequence']}")
        print(f"  Start frame: {sample['start_frame']}")
        
        print(f"\n✓ Phase 1 complete - sequence discovery working!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)