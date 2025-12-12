"""
UP-Fall Ablation: RGB vs LoRA Depth for fall detection.
Optimized for H200 + 12 cores.

Usage:
    # Run full ablation (both conditions)
    python scripts/run_upfall_ablation.py
    
    # Single condition
    python scripts/run_upfall_ablation.py --condition depth_lora
    
    # With temporal shuffle test
    python scripts/run_upfall_ablation.py --test_shuffle
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.upfall_dataset import UPFallDataset
from src.models.fall_detector import FallDetector


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    # Data
    data_path: str = "data/up_fall"
    window_size: int = 16
    window_stride: int = 8
    target_size: tuple = (224, 224)
    fall_trim_start: float = 0.25
    
    # Model
    spatial_channels: int = 64
    temporal_channels: List[int] = field(default_factory=lambda: [64, 128])
    dropout: float = 0.5
    
    # Training - optimized for H200 80GB
    epochs: int = 10
    batch_size: int = 512
    lr: float = 0.01
    weight_decay: float = 1e-4
    num_workers: int = 12
    
    # Early stopping
    patience: int = 6
    min_delta: float = 0.005
    
    # Paths
    output_dir: str = "outputs/upfall_ablation"
    
    # Experiment
    seed: int = 42
    project: str = "lora-depth-fall"
    use_compile: bool = False  # Disable by default (slow startup)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Faster
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels


def get_dataloaders(
    cfg: AblationConfig,
    modality: str,
    depth_source: str = "lora",
    shuffle_temporal: bool = False,
) -> tuple:
    """Create optimized dataloaders."""
    common = dict(
        data_path=cfg.data_path,
        window_size=cfg.window_size,
        window_stride=cfg.window_stride,
        target_size=cfg.target_size,
        modality=modality,
        depth_source=depth_source,
        fall_trim_start=cfg.fall_trim_start,
    )
    
    train_ds = UPFallDataset(split="train", **common)
    val_ds = UPFallDataset(split="val", **common)
    test_ds = UPFallDataset(split="test", **common)
    
    # Optimized loader settings for 12 cores
    loader_cfg = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )
    
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_cfg)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_cfg)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_cfg)
    
    return train_loader, val_loader, test_loader


def create_model(cfg: AblationConfig, in_channels: int, device: torch.device) -> nn.Module:
    """Create and optimize model for H200."""
    model = FallDetector(
        in_channels=in_channels,
        spatial_channels=cfg.spatial_channels,
        temporal_channels=cfg.temporal_channels,
        num_classes=2,
        dropout=cfg.dropout,
    ).to(device)
    
    if cfg.use_compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    return model


def compute_class_weights(dataset: UPFallDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse frequency class weights from dataset metadata (fast)."""
    # Count from windows metadata - no iteration needed
    counts = np.zeros(2, dtype=np.int64)
    for seq_id, _ in dataset.windows:
        label = dataset.sequences[seq_id]["label"]
        counts[label] += 1
    
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 2  # Normalize
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    shuffle_temporal: bool = False,
) -> Dict[str, float]:
    """Evaluate model with optional temporal shuffling."""
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in loader:
        x = batch[modality].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        # Optional: shuffle temporal order to test if model uses it
        if shuffle_temporal:
            idx = torch.randperm(x.shape[1])
            x = x[:, idx]
        
        with autocast(dtype=torch.bfloat16):
            output = model(x)
        
        probs = output["probs"][:, 1]  # P(fall)
        preds = output["probs"].argmax(dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    acc = (all_preds == all_labels).mean()
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    modality: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with mixed precision."""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch in pbar:
        x = batch[modality].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(dtype=torch.bfloat16):
            output = model(x)
            loss = criterion(output["logits"], labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Stats
        total_loss += loss.item() * x.size(0)
        total_correct += (output["probs"].argmax(1) == labels).sum().item()
        total_samples += x.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_correct/total_samples:.3f}")
    
    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,
        "lr": scheduler.get_last_lr()[0],
    }


def run_condition(
    cfg: AblationConfig,
    condition: str,
    device: torch.device,
    resume: bool = False,
) -> Dict[str, float]:
    """Run a single experimental condition."""
    
    # Parse condition
    if condition == "rgb":
        modality, depth_source, in_channels = "rgb", None, 3
    elif condition == "depth_lora":
        modality, depth_source, in_channels = "depth", "lora", 1
    elif condition == "depth_zeroshot":
        modality, depth_source, in_channels = "depth", "zeroshot", 1
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    run_name = f"upfall_{condition}"
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition.upper()}")
    print(f"{'='*60}")
    
    # Init wandb
    wandb.init(
        project=cfg.project,
        name=run_name,
        config={
            "condition": condition,
            "modality": modality,
            **cfg.__dict__,
        },
        reinit=True,
    )
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, modality=modality, depth_source=depth_source or "lora"
    )
    
    # Model
    set_seed(cfg.seed)
    model = create_model(cfg, in_channels, device)
    
    # Training setup
    class_weights = compute_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    
    scaler = GradScaler()
    
    # Checkpoint path
    save_dir = Path(cfg.output_dir) / condition
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "checkpoint.pt"
    
    # Resume from checkpoint if exists and requested
    start_epoch = 0
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    if resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        patience_counter = ckpt["patience_counter"]
        best_model_state = ckpt["best_model_state"]
        print(f"  Resumed at epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, modality, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, modality)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        epoch_time = time.time() - t0
        
        # Log
        metrics = {**train_metrics, **val_metrics, "epoch_time": epoch_time}
        wandb.log(metrics, step=epoch)
        
        print(f"Epoch {epoch+1:2d} | "
              f"Loss: {train_metrics['train_loss']:.4f} | "
              f"Train Acc: {train_metrics['train_acc']:.3f} | "
              f"Val F1: {val_metrics['val_f1']:.3f} | "
              f"Val Acc: {val_metrics['val_accuracy']:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping on val F1
        if val_metrics["val_f1"] > best_val_f1 + cfg.min_delta:
            best_val_f1 = val_metrics["val_f1"]
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        # Save checkpoint after every epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "patience_counter": patience_counter,
            "best_model_state": best_model_state,
        }, ckpt_path)
        
        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for testing
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # Test evaluation
    test_metrics = evaluate(model, test_loader, device, modality)
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    
    print(f"\nTEST RESULTS ({condition}):")
    print(f"  Accuracy:    {test_metrics['test_accuracy']:.4f}")
    print(f"  F1:          {test_metrics['test_f1']:.4f}")
    print(f"  Precision:   {test_metrics['test_precision']:.4f}")
    print(f"  Recall:      {test_metrics['test_recall']:.4f}")
    print(f"  Specificity: {test_metrics['test_specificity']:.4f}")
    
    wandb.log(test_metrics)
    
    # Save best model (save_dir already created above)
    torch.save(best_model_state, save_dir / "best_model.pt")
    
    wandb.finish()
    
    return test_metrics


def test_temporal_shuffle(
    cfg: AblationConfig,
    condition: str,
    device: torch.device,
):
    """Test if model relies on temporal ordering."""
    print(f"\n{'='*60}")
    print(f"TEMPORAL SHUFFLE TEST: {condition.upper()}")
    print(f"{'='*60}")
    
    # Load best model
    save_dir = Path(cfg.output_dir) / condition
    model_path = save_dir / "best_model.pt"
    
    if not model_path.exists():
        print(f"No model found at {model_path}, skipping shuffle test")
        return
    
    # Setup
    if condition == "rgb":
        modality, depth_source, in_channels = "rgb", "lora", 3
    else:
        modality, depth_source, in_channels = "depth", "lora", 1
    
    model = FallDetector(
        in_channels=in_channels,
        spatial_channels=cfg.spatial_channels,
        temporal_channels=cfg.temporal_channels,
        num_classes=2,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    _, _, test_loader = get_dataloaders(cfg, modality=modality, depth_source=depth_source)
    
    # Normal evaluation
    normal_metrics = evaluate(model, test_loader, device, modality, shuffle_temporal=False)
    
    # Shuffled evaluation
    shuffled_metrics = evaluate(model, test_loader, device, modality, shuffle_temporal=True)
    
    print(f"\nNormal order:   Acc={normal_metrics['accuracy']:.4f}, F1={normal_metrics['f1']:.4f}")
    print(f"Shuffled order: Acc={shuffled_metrics['accuracy']:.4f}, F1={shuffled_metrics['f1']:.4f}")
    print(f"Accuracy drop:  {normal_metrics['accuracy'] - shuffled_metrics['accuracy']:.4f}")
    
    if normal_metrics["accuracy"] - shuffled_metrics["accuracy"] > 0.05:
        print("✓ Model uses temporal information (>5% accuracy drop)")
    else:
        print("⚠ Model may not be using temporal information effectively")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None,
                        choices=["rgb", "depth_lora", "depth_zeroshot"], help="Run single condition")
    parser.add_argument("--test_shuffle", action="store_true", help="Test temporal shuffling")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (slow startup, faster training)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    # Config - start with defaults, override with CLI args if provided
    cfg = AblationConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.patience is not None:
        cfg.patience = args.patience
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.compile:
        cfg.use_compile = True
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Output dir
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run conditions
    conditions = [args.condition] if args.condition else ["depth_zeroshot", "depth_lora"]
    results = {}
    
    for condition in conditions:
        results[condition] = run_condition(cfg, condition, device, resume=args.resume)
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for cond, metrics in results.items():
        print(f"{cond:15s} | Acc: {metrics['test_accuracy']:.4f} | F1: {metrics['test_f1']:.4f}")
    
    if "depth_lora" in results and "depth_zeroshot" in results:
        delta_acc = results["depth_lora"]["test_accuracy"] - results["depth_zeroshot"]["test_accuracy"]
        delta_f1 = results["depth_lora"]["test_f1"] - results["depth_zeroshot"]["test_f1"]
        print(f"{'Δ (lora-zeroshot)':<18} | Acc: {delta_acc:+.4f} | F1: {delta_f1:+.4f}")
    
    # Temporal shuffle test
    if args.test_shuffle:
        for condition in conditions:
            test_temporal_shuffle(cfg, condition, device)


if __name__ == "__main__":
    main()