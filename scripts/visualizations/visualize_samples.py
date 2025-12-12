"""
Visualize samples from NYU Depth V2 and UR Fall datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from pathlib import Path

NYU_PATH = Path("data/nyu_depth_v2/nyu_depth_v2_labeled.mat")
URFALL_PATH = Path("data/ur_fall")
OUTPUT_DIR = Path("outputs/visualizations")


def visualize_nyu_sample(idx=0):
    """Visualize a sample from NYU Depth V2."""
    print(f"Loading NYU sample {idx}...")
    
    with h5py.File(NYU_PATH, 'r') as f:
        # Shape: (N, C, W, H) based on your earlier output
        rgb = np.array(f['images'][idx])  # (3, W, H)
        depth = np.array(f['depths'][idx])  # (W, H)
    
    # Transpose to (H, W, C) for visualization
    rgb = np.transpose(rgb, (2, 1, 0))  # (H, W, 3)
    depth = np.transpose(depth, (1, 0))  # (H, W)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title(f"NYU RGB (idx={idx})")
    axes[0].axis('off')
    
    im = axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title(f"NYU Depth (idx={idx})\nRange: {depth.min():.2f}m - {depth.max():.2f}m")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (m)')
    
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"nyu_sample_{idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    return rgb, depth


def visualize_urfall_sequence(seq_name="fall-01-cam0", frame_indices=None):
    """Visualize frames from a UR Fall sequence."""
    seq_dir = URFALL_PATH / seq_name
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    
    if not seq_dir.exists():
        print(f"Sequence not found: {seq_dir}")
        return
    
    rgb_frames = sorted(rgb_dir.glob("*.png"))
    depth_frames = sorted(depth_dir.glob("*.png"))
    
    print(f"Loading UR Fall sequence: {seq_name}")
    print(f"  RGB frames: {len(rgb_frames)}, Depth frames: {len(depth_frames)}")
    
    # Default: show first, middle, and last frames
    if frame_indices is None:
        n = len(rgb_frames)
        frame_indices = [0, n // 2, n - 1]
    
    fig, axes = plt.subplots(2, len(frame_indices), figsize=(4 * len(frame_indices), 7))
    
    for col, idx in enumerate(frame_indices):
        if idx >= len(rgb_frames):
            continue
        
        # Load RGB (OpenCV loads as BGR)
        rgb = cv2.imread(str(rgb_frames[idx]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load Depth
        depth = cv2.imread(str(depth_frames[idx]), cv2.IMREAD_UNCHANGED)
        
        # Normalize depth for visualization
        depth_vis = depth.astype(np.float32)
        depth_vis[depth_vis == 0] = np.nan  # Mark invalid as NaN
        
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"Frame {idx + 1}/{len(rgb_frames)}")
        axes[0, col].axis('off')
        
        im = axes[1, col].imshow(depth_vis, cmap='plasma')
        valid_mask = ~np.isnan(depth_vis)
        if valid_mask.any():
            vmin, vmax = np.nanmin(depth_vis), np.nanmax(depth_vis)
            axes[1, col].set_title(f"Depth: {vmin:.0f}-{vmax:.0f}")
        axes[1, col].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel("RGB", fontsize=12)
    axes[1, 0].set_ylabel("Depth", fontsize=12)
    
    plt.suptitle(f"UR Fall: {seq_name}", fontsize=14)
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"urfall_{seq_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def visualize_fall_vs_adl():
    """Side-by-side comparison of fall vs ADL sequences."""
    fall_seq = "fall-01-cam0"
    adl_seq = "adl-01-cam0"
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    for row, (seq_name, label) in enumerate([(fall_seq, "FALL"), (adl_seq, "ADL")]):
        seq_dir = URFALL_PATH / seq_name
        rgb_frames = sorted((seq_dir / "rgb").glob("*.png"))
        depth_frames = sorted((seq_dir / "depth").glob("*.png"))
        
        n = len(rgb_frames)
        indices = [0, n // 3, 2 * n // 3, n - 1]
        
        for col, idx in enumerate(indices):
            rgb = cv2.imread(str(rgb_frames[idx]))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f"t={idx + 1}/{n}")
            axes[row, col].axis('off')
        
        axes[row, 0].set_ylabel(label, fontsize=14, fontweight='bold')
    
    plt.suptitle("Fall vs ADL Sequence Comparison", fontsize=14)
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fall_vs_adl_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("="*60)
    print("Dataset Visualization")
    print("="*60)
    
    # NYU Depth V2
    print("\n[1/3] NYU Depth V2")
    if NYU_PATH.exists():
        visualize_nyu_sample(idx=0)
        visualize_nyu_sample(idx=100)
    else:
        print(f"  Skipped: {NYU_PATH} not found")
    
    # UR Fall single sequence
    print("\n[2/3] UR Fall Sequence")
    visualize_urfall_sequence("fall-01-cam0")
    visualize_urfall_sequence("adl-01-cam0")
    
    # Fall vs ADL comparison
    print("\n[3/3] Fall vs ADL Comparison")
    visualize_fall_vs_adl()
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*60)