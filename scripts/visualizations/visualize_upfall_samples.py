"""
Visualize UP-Fall samples: RGB sequences and corresponding LoRA depth.
Shows fall vs ADL (especially lying down) comparison.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.upfall_dataset import UPFallDataset

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_samples_by_activity(dataset, target_activities, n_per_activity=1):
    """Find sample indices for specific activities."""
    samples = {act: [] for act in target_activities}
    
    for idx in range(len(dataset)):
        seq_id, _ = dataset.windows[idx]
        activity = dataset.sequences[seq_id]["activity"]
        
        if activity in target_activities and len(samples[activity]) < n_per_activity:
            samples[activity].append(idx)
        
        # Check if we have enough
        if all(len(v) >= n_per_activity for v in samples.values()):
            break
    
    return samples


def visualize_sequence(rgb_seq, depth_seq, title, n_frames=8):
    """Visualize a sequence of RGB and depth frames."""
    # Sample frames evenly
    T = rgb_seq.shape[0]
    indices = np.linspace(0, T-1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(2, n_frames, figsize=(2*n_frames, 4))
    
    for i, idx in enumerate(indices):
        # RGB
        rgb = rgb_seq[idx].permute(1, 2, 0).numpy()
        axes[0, i].imshow(rgb)
        axes[0, i].axis('off')
        axes[0, i].set_title(f't={idx}', fontsize=8)
        
        # Depth
        depth = depth_seq[idx, 0].numpy()
        axes[1, i].imshow(depth, cmap='magma')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('RGB', fontsize=10, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Depth', fontsize=10, rotation=0, ha='right', va='center')
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def create_activity_comparison(dataset_rgb, dataset_depth):
    """Create comparison figure for Fall vs Lying Down vs Walking."""
    # Activities to compare
    activities = {
        2: "Fall (backward)",
        11: "Lying Down (ADL)",
        6: "Walking (ADL)",
    }
    
    samples_rgb = find_samples_by_activity(dataset_rgb, list(activities.keys()), n_per_activity=1)
    
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 2, figure=fig, wspace=0.1, hspace=0.3)
    
    n_frames = 8
    
    for row, (act_id, act_name) in enumerate(activities.items()):
        if not samples_rgb[act_id]:
            continue
        
        idx = samples_rgb[act_id][0]
        
        # Get RGB sample
        sample_rgb = dataset_rgb[idx]
        rgb_seq = sample_rgb["rgb"]
        
        # Get corresponding depth sample (same index should work)
        sample_depth = dataset_depth[idx]
        depth_seq = sample_depth["depth"]
        
        T = rgb_seq.shape[0]
        frame_indices = np.linspace(0, T-1, n_frames, dtype=int)
        
        # RGB row
        for col, fidx in enumerate(frame_indices):
            ax = fig.add_subplot(gs[row, 0])
            if col == 0:
                # Create a strip of frames
                strip_rgb = np.concatenate([
                    rgb_seq[i].permute(1, 2, 0).numpy() 
                    for i in frame_indices
                ], axis=1)
                ax.imshow(strip_rgb)
                ax.set_ylabel(act_name, fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title('RGB Sequence', fontsize=12)
        
        # Depth row  
        ax = fig.add_subplot(gs[row, 1])
        strip_depth = np.concatenate([
            depth_seq[i, 0].numpy()
            for i in frame_indices
        ], axis=1)
        ax.imshow(strip_depth, cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title('LoRA Depth Sequence', fontsize=12)
    
    plt.suptitle('UP-Fall: Fall vs ADL Comparison (8 frames shown per sequence)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    print("Loading UP-Fall datasets...")
    
    dataset_rgb = UPFallDataset(
        data_path="data/up_fall",
        split="test",
        modality="rgb",
        depth_source="lora",
        window_size=16,
        window_stride=8,
    )
    
    dataset_depth = UPFallDataset(
        data_path="data/up_fall",
        split="test",
        modality="depth",
        depth_source="lora",
        window_size=16,
        window_stride=8,
    )
    
    # Create activity comparison
    print("Creating activity comparison figure...")
    fig = create_activity_comparison(dataset_rgb, dataset_depth)
    
    save_path = OUTPUT_DIR / "upfall_activity_comparison.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    fig.savefig(OUTPUT_DIR / "upfall_activity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual sequence visualizations
    print("Creating individual sequence visualizations...")
    
    activities_to_show = {
        2: "fall_backward",
        11: "lying_down", 
        6: "walking",
    }
    
    samples = find_samples_by_activity(dataset_rgb, list(activities_to_show.keys()))
    
    for act_id, name in activities_to_show.items():
        if not samples[act_id]:
            continue
        
        idx = samples[act_id][0]
        sample_rgb = dataset_rgb[idx]
        sample_depth = dataset_depth[idx]
        
        fig = visualize_sequence(
            sample_rgb["rgb"], 
            sample_depth["depth"],
            f"Activity {act_id}: {name.replace('_', ' ').title()}"
        )
        
        save_path = OUTPUT_DIR / f"upfall_sequence_{name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    print("Done!")


if __name__ == "__main__":
    main()