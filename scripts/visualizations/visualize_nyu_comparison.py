"""
Visualize NYU Depth V2 samples: RGB, Ground Truth, Zero-shot, LoRA-adapted.
Generates Figure 2 for the paper.

Note: Adjust the dataset loading section if your NYUDepthDataset has different interface.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import h5py  # NYU .mat files are HDF5 format

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.depth_lora import DepthAnythingLoRA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_nyu_samples(mat_path="data/nyu_depth_v2/nyu_depth_v2_labeled.mat", indices=[0, 50, 100]):
    """Load samples directly from NYU .mat file (HDF5 format)."""
    print(f"Loading NYU data from {mat_path}...")
    
    with h5py.File(mat_path, 'r') as f:
        print(f"  Keys in file: {list(f.keys())}")
        
        images = f['images']
        depths = f['depths']
        
        print(f"  Images shape: {images.shape}")
        print(f"  Depths shape: {depths.shape}")
        
        # Determine number of samples (last dimension)
        n_samples = images.shape[0] if images.shape[0] < images.shape[-1] else images.shape[-1]
        print(f"  N samples: {n_samples}")
        
        samples = []
        for idx in indices:
            if idx >= n_samples:
                print(f"  Warning: index {idx} out of range, skipping")
                continue
            
            # Load and handle HDF5 transposition
            # HDF5 stores in (N, 3, H, W) or (3, H, W, N) depending on version
            if images.shape[0] == n_samples:
                # Shape is (N, 3, H, W)
                rgb = np.array(images[idx])  # (3, H, W)
                rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)
            else:
                # Shape is (3, H, W, N)
                rgb = np.array(images[:, :, :, idx])  # (3, H, W)
                rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)
            
            if depths.shape[0] == n_samples:
                depth = np.array(depths[idx])  # (H, W)
            else:
                depth = np.array(depths[:, :, idx])
                depth = np.transpose(depth)
            
            # Ensure correct orientation (NYU images may need rotation)
            # Check if image looks correct, otherwise try transposing
            if rgb.shape[0] > rgb.shape[1]:  # Height > Width, likely needs rotation
                rgb = np.rot90(rgb, k=3)
                depth = np.rot90(depth, k=3)
            
            # Convert to tensor format  
            rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()
            if rgb_tensor.max() > 1.0:
                rgb_tensor = rgb_tensor / 255.0
            
            depth_tensor = torch.from_numpy(depth.copy()).unsqueeze(0).float()
            
            samples.append((rgb_tensor, depth_tensor))
            print(f"  Loaded sample {idx}: RGB {rgb_tensor.shape}, Depth {depth_tensor.shape}")
    
    return samples


def load_models():
    """Load zero-shot and LoRA-adapted models."""
    # Zero-shot (no LoRA loaded)
    model_zeroshot = DepthAnythingLoRA(
        lora_r=16, lora_alpha=32,
        lora_target_modules=["query", "key", "value", "dense"]
    ).to(DEVICE)
    model_zeroshot.eval()
    
    # LoRA-adapted
    model_lora = DepthAnythingLoRA(
        lora_r=16, lora_alpha=32,
        lora_target_modules=["query", "key", "value", "dense"]
    ).to(DEVICE)
    model_lora.load_lora("outputs/depth_lora/checkpoints/best")
    model_lora.eval()
    
    return model_zeroshot, model_lora


@torch.inference_mode()
def predict_depth(model, rgb_tensor):
    """Run depth prediction."""
    rgb = rgb_tensor.unsqueeze(0).to(DEVICE)
    depth = model(rgb)
    return depth.squeeze().cpu().numpy()


def normalize_depth(depth, vmin=None, vmax=None):
    """Normalize depth for visualization."""
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()
    return (depth - vmin) / (vmax - vmin + 1e-8)


def create_comparison_figure(samples, model_zeroshot, model_lora, n_samples=3):
    """Create a grid comparison figure."""
    fig = plt.figure(figsize=(14, 3.5 * n_samples))
    gs = GridSpec(n_samples, 4, figure=fig, wspace=0.05, hspace=0.15)
    
    for i, (rgb, gt_depth) in enumerate(samples[:n_samples]):
        # Predict
        pred_zeroshot = predict_depth(model_zeroshot, rgb)
        pred_lora = predict_depth(model_lora, rgb)
        
        # Resize predictions to match GT
        h, w = gt_depth.shape[-2:]
        pred_zeroshot = cv2.resize(pred_zeroshot, (w, h))
        pred_lora = cv2.resize(pred_lora, (w, h))
        
        # Convert tensors to numpy for display
        rgb_np = rgb.permute(1, 2, 0).numpy()
        gt_np = gt_depth.squeeze().numpy()
        
        # Normalize depths for display (use GT range for consistency)
        vmin, vmax = gt_np.min(), gt_np.max()
        
        # Row: RGB, GT, Zero-shot, LoRA
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(rgb_np)
        ax1.axis('off')
        if i == 0:
            ax1.set_title('RGB Input', fontsize=12, fontweight='bold')
        
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(gt_np, cmap='magma')
        ax2.axis('off')
        if i == 0:
            ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(normalize_depth(pred_zeroshot), cmap='magma')
        ax3.axis('off')
        if i == 0:
            ax3.set_title('Zero-shot', fontsize=12, fontweight='bold')
        
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(normalize_depth(pred_lora), cmap='magma')
        ax4.axis('off')
        if i == 0:
            ax4.set_title('LoRA-adapted', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    print("Loading models...")
    model_zeroshot, model_lora = load_models()
    
    print("Loading NYU samples...")
    
    # Try using existing NYUDepthDataset first
    try:
        from src.data.nyu_dataset import NYUDepthDataset
        print("  Using NYUDepthDataset class...")
        dataset = NYUDepthDataset(
            data_path="data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
            split="val",
        )
        indices = [0, 30, 80, 50, 100]
        samples = []
        for idx in indices[:5]:
            if idx < len(dataset):
                sample = dataset[idx]
                samples.append((sample["rgb"], sample["depth"]))
        print(f"  Loaded {len(samples)} samples from NYUDepthDataset")
    except Exception as e:
        print(f"  NYUDepthDataset not available ({e}), using direct HDF5 loading...")
        samples = load_nyu_samples(
            mat_path="data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
            indices=[0, 200, 500, 800, 1000]
        )
    
    if not samples:
        print("No samples loaded! Check mat file path.")
        return
    
    print(f"Creating comparison figure with {min(3, len(samples))} samples...")
    fig = create_comparison_figure(samples, model_zeroshot, model_lora, n_samples=3)
    
    # Save
    save_path = OUTPUT_DIR / "nyu_depth_comparison.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    # Also save PNG for quick viewing
    fig.savefig(OUTPUT_DIR / "nyu_depth_comparison.png", dpi=150, bbox_inches='tight')
    
    # Also copy to figures/ for paper
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "nyu_depth_comparison.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'nyu_depth_comparison.pdf'}")
    
    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()