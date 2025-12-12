"""
Generate per-activity accuracy bar chart for RGB vs Depth.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from diagnostic results
ACTIVITIES = {
    1: ("Fall forward", "Fall"),
    2: ("Fall backward", "Fall"),
    3: ("Fall lateral L", "Fall"),
    4: ("Fall lateral R", "Fall"),
    5: ("Fall sitting", "Fall"),
    6: ("Walking", "ADL"),
    7: ("Standing", "ADL"),
    8: ("Sitting", "ADL"),
    9: ("Picking up", "ADL"),
    10: ("Jumping", "ADL"),
    11: ("Lying down", "ADL"),
}

RGB_ACC = {
    1: 0.588, 2: 0.812, 3: 0.678, 4: 0.547, 5: 0.452,
    6: 0.983, 7: 1.000, 8: 1.000, 9: 1.000, 10: 1.000, 11: 0.646
}

DEPTH_ACC = {
    1: 0.819, 2: 0.960, 3: 0.911, 4: 0.995, 5: 0.985,
    6: 0.570, 7: 0.983, 8: 0.988, 9: 0.970, 10: 0.928, 11: 0.019
}


def create_grouped_bar_chart():
    """Create grouped bar chart for per-activity accuracy."""
    activities = list(ACTIVITIES.keys())
    labels = [f"A{i}" for i in activities]
    
    rgb_vals = [RGB_ACC[i] for i in activities]
    depth_vals = [DEPTH_ACC[i] for i in activities]
    
    x = np.arange(len(activities))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create bars
    bars_rgb = ax.bar(x - width/2, rgb_vals, width, label='RGB', color='#1f77b4', alpha=0.8)
    bars_depth = ax.bar(x + width/2, depth_vals, width, label='LoRA Depth', color='#d62728', alpha=0.8)
    
    # Customize
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Activity', fontsize=12)
    ax.set_title('Per-Activity Accuracy: RGB vs LoRA Depth', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add Fall/ADL separator
    ax.axvline(x=4.5, color='black', linestyle='-', alpha=0.3)
    ax.text(2, 1.05, 'Falls', ha='center', fontsize=11, style='italic')
    ax.text(7.5, 1.05, 'ADLs', ha='center', fontsize=11, style='italic')
    
    # Highlight Activity 11
    ax.annotate('Lying down\n(critical test)', 
                xy=(10, 0.1), xytext=(9, 0.3),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # Add value labels on bars
    for bar, val in zip(bars_rgb, rgb_vals):
        if val < 0.1:
            continue
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    for bar, val in zip(bars_depth, depth_vals):
        if val < 0.1:
            continue
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    return fig


def create_fall_vs_adl_summary():
    """Create summary bar chart: Fall detection vs ADL recognition."""
    # Aggregate by type
    fall_rgb = np.mean([RGB_ACC[i] for i in range(1, 6)])
    fall_depth = np.mean([DEPTH_ACC[i] for i in range(1, 6)])
    
    adl_rgb = np.mean([RGB_ACC[i] for i in range(6, 12)])
    adl_depth = np.mean([DEPTH_ACC[i] for i in range(6, 12)])
    
    # Activity 11 specifically
    lay_rgb = RGB_ACC[11]
    lay_depth = DEPTH_ACC[11]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x = np.array([0, 1])
    width = 0.35
    
    # Fall detection
    axes[0].bar(x - width/2, [fall_rgb], width, label='RGB', color='#1f77b4')
    axes[0].bar(x[:1] + width/2, [fall_depth], width, label='LoRA Depth', color='#d62728')
    axes[0].set_ylabel('Mean Accuracy')
    axes[0].set_title('Fall Detection\n(Activities 1-5)', fontweight='bold')
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([''])
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].bar_label(axes[0].containers[0], fmt='%.2f', padding=3)
    axes[0].bar_label(axes[0].containers[1], fmt='%.2f', padding=3)
    
    # ADL recognition  
    axes[1].bar(x[:1] - width/2, [adl_rgb], width, label='RGB', color='#1f77b4')
    axes[1].bar(x[:1] + width/2, [adl_depth], width, label='LoRA Depth', color='#d62728')
    axes[1].set_title('ADL Recognition\n(Activities 6-11)', fontweight='bold')
    axes[1].set_xticks([0])
    axes[1].set_xticklabels([''])
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].bar_label(axes[1].containers[0], fmt='%.2f', padding=3)
    axes[1].bar_label(axes[1].containers[1], fmt='%.2f', padding=3)
    
    # Activity 11 only
    axes[2].bar(x[:1] - width/2, [lay_rgb], width, label='RGB', color='#1f77b4')
    axes[2].bar(x[:1] + width/2, [lay_depth], width, label='LoRA Depth', color='#d62728')
    axes[2].set_title('Lying Down\n(Activity 11 - Critical)', fontweight='bold')
    axes[2].set_xticks([0])
    axes[2].set_xticklabels([''])
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].bar_label(axes[2].containers[0], fmt='%.2f', padding=3)
    axes[2].bar_label(axes[2].containers[1], fmt='%.2f', padding=3)
    
    plt.tight_layout()
    return fig


def main():
    print("Creating per-activity visualizations...")
    
    # Grouped bar chart
    fig = create_grouped_bar_chart()
    fig.savefig(OUTPUT_DIR / "per_activity_accuracy.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "per_activity_accuracy.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'per_activity_accuracy.pdf'}")
    plt.close()
    
    # Summary chart
    fig = create_fall_vs_adl_summary()
    fig.savefig(OUTPUT_DIR / "fall_vs_adl_summary.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fall_vs_adl_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'fall_vs_adl_summary.pdf'}")
    plt.close()
    
    print("Done!")


if __name__ == "__main__":
    main()