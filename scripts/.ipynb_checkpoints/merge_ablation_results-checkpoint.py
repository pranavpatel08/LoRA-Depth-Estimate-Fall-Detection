"""
Merge ablation results from parallel runs + run temporal shuffle tests.

Usage:
    python scripts/merge_ablation_results.py
    python scripts/merge_ablation_results.py --test_shuffle
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.upfall_dataset import UPFallDataset
from src.models.fall_detector import FallDetector


OUTPUT_DIR = Path("outputs/upfall_ablation")
CONDITIONS = ["rgb", "depth_lora", "depth_zeroshot"]


def load_wandb_summary(condition: str) -> dict:
    """Try to load metrics from wandb or saved JSON."""
    json_path = OUTPUT_DIR / condition / "metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    shuffle_temporal: bool = False,
) -> dict:
    """Evaluate with optional temporal shuffle."""
    model.eval()
    all_preds, all_labels = [], []
    
    for batch in loader:
        x = batch[modality].to(device, non_blocking=True)
        labels = batch["label"]
        
        if shuffle_temporal:
            idx = torch.randperm(x.shape[1])
            x = x[:, idx]
        
        with autocast(dtype=torch.bfloat16):
            output = model(x)
        
        preds = output["probs"].argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_temporal_shuffle_test(device: torch.device) -> dict:
    """Test temporal dependency for both conditions."""
    results = {}
    
    for condition in CONDITIONS:
        model_path = OUTPUT_DIR / condition / "best_model.pt"
        if not model_path.exists():
            print(f"  {condition}: model not found, skipping")
            continue
        
        # Setup
        if condition == "rgb":
            modality, in_ch = "rgb", 3
        else:
            modality, in_ch = "depth", 1
        
        # Load model
        model = FallDetector(in_channels=in_ch, spatial_channels=64,
                            temporal_channels=[64, 128], dropout=0.5).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        
        # Data
        test_ds = UPFallDataset(
            data_path="data/up_fall", split="test", modality=modality,
            depth_source="lora", window_size=16, window_stride=8,
        )
        test_loader = DataLoader(test_ds, batch_size=64, num_workers=8, pin_memory=True)
        
        # Evaluate both ways
        normal = evaluate_model(model, test_loader, device, modality, shuffle_temporal=False)
        shuffled = evaluate_model(model, test_loader, device, modality, shuffle_temporal=True)
        
        results[condition] = {
            "normal_acc": normal["accuracy"],
            "shuffled_acc": shuffled["accuracy"],
            "acc_drop": normal["accuracy"] - shuffled["accuracy"],
            "normal_f1": normal["f1"],
            "shuffled_f1": shuffled["f1"],
            "f1_drop": normal["f1"] - shuffled["f1"],
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_shuffle", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check both models exist
    print("Checking for trained models...")
    all_exist = True
    for cond in CONDITIONS:
        model_path = OUTPUT_DIR / cond / "best_model.pt"
        exists = model_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {cond}: {status} {model_path}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠ Not all models found. Run both conditions first:")
        print("  python scripts/run_upfall_ablation.py --condition rgb")
        print("  python scripts/run_upfall_ablation.py --condition depth_lora")
        return
    
    # Re-evaluate both models on test set for consistent comparison
    print("\n" + "="*60)
    print("RE-EVALUATING MODELS ON TEST SET")
    print("="*60)
    
    results = {}
    for condition in CONDITIONS:
        model_path = OUTPUT_DIR / condition / "best_model.pt"
        
        if condition == "rgb":
            modality, in_ch = "rgb", 3
        else:
            modality, in_ch = "depth", 1
        
        model = FallDetector(in_channels=in_ch, spatial_channels=64,
                            temporal_channels=[64, 128], dropout=0.5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        test_ds = UPFallDataset(
            data_path="data/up_fall", split="test", modality=modality,
            depth_source="lora", window_size=16, window_stride=8,
        )
        test_loader = DataLoader(test_ds, batch_size=64, num_workers=8, pin_memory=True)
        
        metrics = evaluate_model(model, test_loader, device, modality)
        results[condition] = metrics
        
        print(f"\n{condition.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(f"{'Condition':<15} | {'Accuracy':>10} | {'F1':>10}")
    print("-"*42)
    for cond in CONDITIONS:
        if cond in results:
            m = results[cond]
            print(f"{cond:<15} | {m['accuracy']:>10.4f} | {m['f1']:>10.4f}")
    
    # Pairwise comparisons
    print("-"*42)
    if "depth_lora" in results and "rgb" in results:
        d_acc = results["depth_lora"]["accuracy"] - results["rgb"]["accuracy"]
        d_f1 = results["depth_lora"]["f1"] - results["rgb"]["f1"]
        print(f"{'Δ lora-rgb':<15} | {d_acc:>+10.4f} | {d_f1:>+10.4f}")
    
    if "depth_lora" in results and "depth_zeroshot" in results:
        d_acc = results["depth_lora"]["accuracy"] - results["depth_zeroshot"]["accuracy"]
        d_f1 = results["depth_lora"]["f1"] - results["depth_zeroshot"]["f1"]
        print(f"{'Δ lora-zeroshot':<15} | {d_acc:>+10.4f} | {d_f1:>+10.4f}")
    
    if "depth_zeroshot" in results and "rgb" in results:
        d_acc = results["depth_zeroshot"]["accuracy"] - results["rgb"]["accuracy"]
        d_f1 = results["depth_zeroshot"]["f1"] - results["rgb"]["f1"]
        print(f"{'Δ zeroshot-rgb':<15} | {d_acc:>+10.4f} | {d_f1:>+10.4f}")
    
    # Summary message
    if "depth_lora" in results and "rgb" in results:
        delta_f1 = results["depth_lora"]["f1"] - results["rgb"]["f1"]
        if delta_f1 > 0:
            print(f"\n✓ LoRA Depth outperforms RGB by {delta_f1:.1%} F1")
        else:
            print(f"\n⚠ RGB outperforms LoRA Depth (unexpected)")
    
    # Temporal shuffle test
    if args.test_shuffle:
        print("\n" + "="*60)
        print("TEMPORAL SHUFFLE TEST")
        print("="*60)
        print("(Tests if model relies on temporal ordering)\n")
        
        shuffle_results = run_temporal_shuffle_test(device)
        
        print(f"{'Condition':<15} | {'Normal Acc':>10} | {'Shuffled':>10} | {'Drop':>10}")
        print("-"*55)
        for cond, r in shuffle_results.items():
            print(f"{cond:<15} | {r['normal_acc']:>10.4f} | {r['shuffled_acc']:>10.4f} | {r['acc_drop']:>+10.4f}")
        
        # Interpret
        for cond, r in shuffle_results.items():
            if r["acc_drop"] > 0.05:
                print(f"\n✓ {cond}: Model uses temporal info (>{r['acc_drop']:.1%} accuracy drop)")
            else:
                print(f"\n⚠ {cond}: Model may not use temporal info effectively")
    
    # Save combined results
    comparisons = {}
    if "depth_lora" in results and "rgb" in results:
        comparisons["lora_vs_rgb"] = {
            "delta_accuracy": results["depth_lora"]["accuracy"] - results["rgb"]["accuracy"],
            "delta_f1": results["depth_lora"]["f1"] - results["rgb"]["f1"],
        }
    if "depth_lora" in results and "depth_zeroshot" in results:
        comparisons["lora_vs_zeroshot"] = {
            "delta_accuracy": results["depth_lora"]["accuracy"] - results["depth_zeroshot"]["accuracy"],
            "delta_f1": results["depth_lora"]["f1"] - results["depth_zeroshot"]["f1"],
        }
    
    combined = {
        "test_results": results,
        "comparisons": comparisons,
    }
    if args.test_shuffle:
        combined["temporal_shuffle"] = shuffle_results
    
    with open(OUTPUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()