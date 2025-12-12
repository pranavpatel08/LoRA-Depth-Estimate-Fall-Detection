"""
Generate estimated depth for Le2i dataset.
"""

import os
os.environ["USE_TF"] = "0"

import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.depth_lora import DepthAnythingLoRA


def generate_depth_for_sequence(model, rgb_dir: Path, output_dir: Path, device: torch.device):
    """Generate depth maps for all frames in a sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_frames = sorted(rgb_dir.glob("*.png"))
    
    for rgb_path in rgb_frames:
        output_path = output_dir / rgb_path.name
        
        if output_path.exists():
            continue
        
        # Load and preprocess
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        rgb_resized = cv2.resize(rgb, (518, 518))
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
        
        # Estimate depth
        with torch.no_grad():
            depth_pred = model(rgb_tensor)
            if hasattr(depth_pred, 'predicted_depth'):
                depth_pred = depth_pred.predicted_depth
        
        # Resize to original size
        h, w = rgb.shape[:2]
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=True
        ).squeeze().cpu().numpy()
        
        # Normalize and save
        depth_min, depth_max = depth_pred.min(), depth_pred.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_pred - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_pred)
        
        depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_uint16)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    le2i_path = Path("data/le2i_frames")
    
    if not le2i_path.exists():
        print(f"Le2i frames not found at {le2i_path}")
        print("Run scripts/prepare_le2i.py first")
        return
    
    # Load models
    print("\nLoading models...")
    
    zs_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)
    zs_model.eval()
    
    lora_model = DepthAnythingLoRA()
    lora_model.load_lora("outputs/depth_lora/checkpoints/best")
    lora_model = lora_model.to(device)
    lora_model.eval()
    
    # Process sequences
    sequences = sorted([d for d in le2i_path.iterdir() if d.is_dir()])
    print(f"Found {len(sequences)} sequences")
    
    for seq_dir in tqdm(sequences, desc="Processing"):
        rgb_dir = seq_dir / "rgb"
        
        if not rgb_dir.exists():
            continue
        
        # Zero-shot depth
        zs_dir = seq_dir / "depth_zeroshot"
        generate_depth_for_sequence(zs_model, rgb_dir, zs_dir, device)
        
        # LoRA depth
        lora_dir = seq_dir / "depth_lora"
        generate_depth_for_sequence(lora_model, rgb_dir, lora_dir, device)
    
    print("\n✓ Depth generation complete!")


if __name__ == "__main__":
    main()