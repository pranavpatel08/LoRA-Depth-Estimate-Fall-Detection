"""
Generate estimated depth maps from RGB using Depth Anything V2.
Compares zero-shot vs LoRA-adapted models.
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


def generate_depth_for_sequence(
    model,
    rgb_dir: Path,
    output_dir: Path,
    target_size: tuple = (480, 640),
    device: torch.device = torch.device("cuda"),
):
    """
    Generate depth maps for all RGB frames in a sequence.
    
    Args:
        model: Depth estimation model
        rgb_dir: Directory containing RGB frames
        output_dir: Directory to save estimated depth
        target_size: Output size (H, W)
        device: Torch device
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_frames = sorted(rgb_dir.glob("*.png"))
    
    for rgb_path in rgb_frames:
        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model (resize to model input size)
        rgb_resized = cv2.resize(rgb, (518, 518))
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
        
        # Estimate depth
        with torch.no_grad():
            depth_pred = model(rgb_tensor)
            
            # Handle different output types
            if hasattr(depth_pred, 'predicted_depth'):
                depth_pred = depth_pred.predicted_depth
        
        # Resize to target size
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=target_size,
            mode='bilinear',
            align_corners=True,
        ).squeeze().cpu().numpy()
        
        # Normalize to uint16 range (like original Kinect depth)
        depth_min, depth_max = depth_pred.min(), depth_pred.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_pred - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_pred)
        
        depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        
        # Save with matching filename
        output_name = rgb_path.name.replace("-rgb-", "-d-").replace("rgb", "d")
        # Handle various naming conventions
        if output_name == rgb_path.name:
            output_name = rgb_path.stem + "_depth.png"
        
        cv2.imwrite(str(output_dir / output_name), depth_uint16)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    urfall_path = Path("data/ur_fall")
    
    # =========================================================
    # Load models
    # =========================================================
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    
    # Zero-shot model
    print("\nLoading zero-shot Depth Anything V2...")
    zs_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)
    zs_model.eval()
    
    # LoRA-adapted model
    print("Loading LoRA-adapted model...")
    lora_model = DepthAnythingLoRA()
    lora_model.load_lora("outputs/depth_lora/checkpoints/best")
    lora_model = lora_model.to(device)
    lora_model.eval()
    
    # =========================================================
    # Process sequences
    # =========================================================
    print("\n" + "="*60)
    print("Generating Estimated Depth Maps")
    print("="*60)
    
    # Find all sequences (cam0 only)
    sequences = sorted([
        d for d in urfall_path.iterdir()
        if d.is_dir() and d.name != "zips" and "cam1" not in d.name
    ])
    
    print(f"Found {len(sequences)} sequences")
    
    for seq_dir in tqdm(sequences, desc="Processing sequences"):
        rgb_dir = seq_dir / "rgb"
        
        if not rgb_dir.exists():
            continue
        
        # Generate zero-shot depth
        zs_output_dir = seq_dir / "depth_zeroshot"
        if not zs_output_dir.exists() or len(list(zs_output_dir.glob("*.png"))) == 0:
            generate_depth_for_sequence(
                model=zs_model,
                rgb_dir=rgb_dir,
                output_dir=zs_output_dir,
                device=device,
            )
        
        # Generate LoRA depth
        lora_output_dir = seq_dir / "depth_lora"
        if not lora_output_dir.exists() or len(list(lora_output_dir.glob("*.png"))) == 0:
            generate_depth_for_sequence(
                model=lora_model,
                rgb_dir=rgb_dir,
                output_dir=lora_output_dir,
                device=device,
            )
    
    # =========================================================
    # Verify outputs
    # =========================================================
    print("\n" + "="*60)
    print("Verification")
    print("="*60)
    
    # Check a sample sequence
    sample_seq = sequences[0]
    
    real_count = len(list((sample_seq / "depth").glob("*.png")))
    zs_count = len(list((sample_seq / "depth_zeroshot").glob("*.png")))
    lora_count = len(list((sample_seq / "depth_lora").glob("*.png")))
    
    print(f"\nSample sequence: {sample_seq.name}")
    print(f"  Real depth frames: {real_count}")
    print(f"  Zero-shot depth frames: {zs_count}")
    print(f"  LoRA depth frames: {lora_count}")
    
    # Count total
    total_real = sum(len(list((s / "depth").glob("*.png"))) for s in sequences if (s / "depth").exists())
    total_zs = sum(len(list((s / "depth_zeroshot").glob("*.png"))) for s in sequences if (s / "depth_zeroshot").exists())
    total_lora = sum(len(list((s / "depth_lora").glob("*.png"))) for s in sequences if (s / "depth_lora").exists())
    
    print(f"\nTotal frames across all sequences:")
    print(f"  Real depth: {total_real}")
    print(f"  Zero-shot: {total_zs}")
    print(f"  LoRA: {total_lora}")
    
    print("\n✓ Depth generation complete!")
    print(f"Generated depth saved to: {urfall_path}/*/depth_zeroshot and depth_lora")


if __name__ == "__main__":
    main()