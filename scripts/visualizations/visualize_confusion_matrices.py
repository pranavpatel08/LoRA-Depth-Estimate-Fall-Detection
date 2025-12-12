"""
Generate confusion matrices for RGB and Depth models.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.upfall_dataset import UPFallDataset
from src.models.fall_detector import FallDetector

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(condition):
    """Load trained model."""
    in_ch = 3 if condition == "rgb" else 1
    model = FallDetector(
        in_channels=in_ch,
        spatial_channels=64,
        temporal_channels=[64, 128],
        dropout=0.5
    ).to(DEVICE)
    
    state = torch.load(
        f"outputs/upfall_ablation/{condition}/best_model.pt",
        map_location=DEVICE
    )
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def get_predictions(model, loader, modality):
    """Get all predictions."""
    all_preds, all_labels = [], []
    
    for batch in loader:
        x = batch[modality].to(DEVICE, non_blocking=True)
        labels = batch["label"]
        
        with autocast(dtype=torch.bfloat16):
            output = model(x)
        
        preds = output["probs"].argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, title, ax, labels=["ADL", "Fall"]):
    """Plot a single confusion matrix."""
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False,
        annot_kws={"size": 14}
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')


def plot_normalized_confusion_matrix(cm, title, ax, labels=["ADL", "Fall"]):
    """Plot normalized confusion matrix (percentages)."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_norm, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False,
        annot_kws={"size": 12}
    )
    
    # Add percentage symbol
    for t in ax.texts:
        t.set_text(t.get_text() + '%')
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')


def main():
    print("Loading models and data...")
    
    # Load models
    model_rgb = load_model("rgb")
    model_depth = load_model("depth_lora")
    
    # Load test data
    test_rgb = UPFallDataset(
        data_path="data/up_fall", split="test", modality="rgb",
        depth_source="lora", window_size=16, window_stride=8
    )
    test_depth = UPFallDataset(
        data_path="data/up_fall", split="test", modality="depth",
        depth_source="lora", window_size=16, window_stride=8
    )
    
    loader_rgb = DataLoader(test_rgb, batch_size=64, num_workers=8, pin_memory=True)
    loader_depth = DataLoader(test_depth, batch_size=64, num_workers=8, pin_memory=True)
    
    # Get predictions
    print("Running inference...")
    preds_rgb, labels_rgb = get_predictions(model_rgb, loader_rgb, "rgb")
    preds_depth, labels_depth = get_predictions(model_depth, loader_depth, "depth")
    
    # Compute confusion matrices
    cm_rgb = confusion_matrix(labels_rgb, preds_rgb)
    cm_depth = confusion_matrix(labels_depth, preds_depth)
    
    # Create figure with raw counts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    plot_confusion_matrix(cm_rgb, "RGB Model", axes[0])
    plot_confusion_matrix(cm_depth, "LoRA Depth Model", axes[1])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrices.pdf'}")
    plt.close()
    
    # Create figure with percentages
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    plot_normalized_confusion_matrix(cm_rgb, "RGB Model", axes[0])
    plot_normalized_confusion_matrix(cm_depth, "LoRA Depth Model", axes[1])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices_normalized.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "confusion_matrices_normalized.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrices_normalized.pdf'}")
    plt.close()
    
    # Print stats
    print("\n" + "="*50)
    print("CONFUSION MATRIX STATS")
    print("="*50)
    
    for name, cm in [("RGB", cm_rgb), ("Depth", cm_depth)]:
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name}:")
        print(f"  True Negatives (ADL correct):  {tn}")
        print(f"  False Positives (ADL→Fall):    {fp}")
        print(f"  False Negatives (Fall→ADL):    {fn}")
        print(f"  True Positives (Fall correct): {tp}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()