"""
Generate LoRA-adapted depth maps for UP-Fall dataset.
Optimized for H200 GPU with batch processing and resume capability.

Usage:
    python data/scripts/generate_upfall_depth.py --resume
    python data/scripts/generate_upfall_depth.py --batch_size 64
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.depth_lora import DepthAnythingLoRA


class RGBFrameDataset(Dataset):
    """Simple dataset for batch processing RGB frames."""
    
    def __init__(
        self,
        frame_paths: List[Path],
        output_paths: List[Path],
        input_size: Tuple[int, int] = (518, 518),
    ):
        self.frame_paths = frame_paths
        self.output_paths = output_paths
        self.input_size = input_size
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        
        # Load and preprocess
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original size for later
        orig_h, orig_w = img.shape[:2]
        
        # Resize for model input
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        
        # To tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return {
            "image": img,
            "idx": idx,
            "orig_size": (orig_h, orig_w),
        }


def collect_frames_to_process(
    data_path: Path,
    output_subdir: str = "depth_lora",
    camera: int = 1,
    resume: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    Collect all RGB frames that need processing.
    
    Returns:
        (rgb_paths, output_paths) - parallel lists
    """
    rgb_paths = []
    output_paths = []
    skipped = 0
    
    camera_name = f"Camera{camera}"
    
    for subject_dir in sorted(data_path.iterdir()):
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
                
                # Create output directory
                output_dir = camera_dir / output_subdir
                output_dir.mkdir(exist_ok=True)
                
                # Find RGB frames
                frames = sorted(camera_dir.glob("*.png"))
                if not frames:
                    frames = sorted(camera_dir.glob("*.jpg"))
                
                for frame_path in frames:
                    output_path = output_dir / frame_path.name
                    
                    # Skip if exists and resume mode
                    if resume and output_path.exists():
                        skipped += 1
                        continue
                    
                    rgb_paths.append(frame_path)
                    output_paths.append(output_path)
    
    print(f"Found {len(rgb_paths)} frames to process, {skipped} already exist (skipped)")
    return rgb_paths, output_paths


def save_depth_map(depth: np.ndarray, path: Path):
    """
    Save depth map as 16-bit PNG.
    Normalizes to [0, 65535] range for maximum precision.
    """
    # Normalize to [0, 1] based on min/max
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth)
    
    # Convert to 16-bit
    depth_uint16 = (depth_norm * 65535).astype(np.uint16)
    
    cv2.imwrite(str(path), depth_uint16)


@torch.inference_mode()
def process_batch(
    model: torch.nn.Module,
    batch: dict,
    output_paths: List[Path],
    device: torch.device,
):
    """Process a batch of frames and save results."""
    images = batch["image"].to(device, non_blocking=True)
    indices = batch["idx"]
    orig_sizes = batch["orig_size"]  # (H_list, W_list)
    
    # Forward pass
    depth_pred = model(images)  # (B, H, W)
    
    # Process each prediction
    for i, idx in enumerate(indices):
        depth = depth_pred[i]  # (H, W)
        orig_h, orig_w = orig_sizes[0][i].item(), orig_sizes[1][i].item()
        
        # Resize to original resolution
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        ).squeeze()
        
        # Save
        depth_np = depth.cpu().numpy()
        save_depth_map(depth_np, output_paths[idx])


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for UP-Fall")
    parser.add_argument("--data_path", type=str, default="data/up_fall")
    parser.add_argument("--checkpoint", type=str, default="outputs/depth_lora/checkpoints/best",
                        help="LoRA checkpoint path, or 'none' for zero-shot")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size (H200 can handle 48-64)")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--resume", action="store_true", help="Skip existing depth maps")
    parser.add_argument("--output_subdir", type=str, default="depth_lora")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (faster but slower startup)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Collect frames
    print(f"\nScanning {args.data_path}...")
    rgb_paths, output_paths = collect_frames_to_process(
        Path(args.data_path),
        output_subdir=args.output_subdir,
        camera=args.camera,
        resume=args.resume,
    )
    
    if len(rgb_paths) == 0:
        print("No frames to process. Done!")
        return
    
    # Load model
    print(f"\nLoading model...")
    model = DepthAnythingLoRA(
        lora_r=16,
        lora_alpha=32,
        lora_target_modules=["query", "key", "value", "dense"],
    )
    
    if args.checkpoint.lower() != "none":
        print(f"  Loading LoRA weights from {args.checkpoint}")
        model.load_lora(args.checkpoint)
    else:
        print(f"  Using ZERO-SHOT (no LoRA weights)")
    
    model = model.to(device)
    model.eval()
    
    # Optional: torch.compile for faster inference
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Enable TF32 for H200 (faster matmuls)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create dataset and dataloader
    dataset = RGBFrameDataset(rgb_paths, output_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    # Process
    print(f"\nProcessing {len(rgb_paths)} frames (batch_size={args.batch_size})...")
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in tqdm(dataloader, desc="Generating depth"):
            process_batch(model, batch, output_paths, device)
    
    # Verify
    n_created = sum(1 for p in output_paths if p.exists())
    print(f"\n✓ Complete! Created {n_created}/{len(output_paths)} depth maps")
    print(f"  Output: {args.data_path}/Subject*/Activity*/Trial*/Camera{args.camera}/{args.output_subdir}/")


if __name__ == "__main__":
    main()