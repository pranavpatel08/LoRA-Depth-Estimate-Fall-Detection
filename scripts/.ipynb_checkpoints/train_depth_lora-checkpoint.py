"""
Train LoRA-adapted Depth Anything V2 on NYU Depth V2.

Usage:
    USE_TF=0 python scripts/train_depth_lora.py
    USE_TF=0 python scripts/train_depth_lora.py --config configs/depth_lora.yaml
"""

import os
os.environ["USE_TF"] = "0"

import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.depth_lora import DepthAnythingLoRA, DepthLoss, DepthMetrics
from src.data.nyu_dataset import get_nyu_dataloaders


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics like delta1 (higher is better),
                  'min' for metrics like loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler with warmup."""
    warmup_steps = config['training']['warmup_epochs'] * steps_per_epoch
    total_steps = config['training']['epochs'] * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, dataloader, loss_fn, device, use_amp=True):
    """Run evaluation and return metrics."""
    model.eval()
    
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            rgb = batch['rgb'].to(device)
            depth_gt = batch['depth'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            with autocast(enabled=use_amp):
                depth_pred = model(rgb)
                losses = loss_fn(depth_pred, depth_gt, valid_mask)
            
            total_loss += losses['loss'].item()
            
            metrics = DepthMetrics.compute(depth_pred, depth_gt, valid_mask)
            all_metrics.append(metrics)
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics


def evaluate_zero_shot(config, device):
    """Evaluate zero-shot (no LoRA) performance for comparison."""
    print("\n" + "="*60)
    print("Evaluating Zero-Shot Performance (baseline)")
    print("="*60)
    
    from transformers import AutoModelForDepthEstimation
    
    base_model = AutoModelForDepthEstimation.from_pretrained(
        config['model']['name']
    ).to(device)
    base_model.eval()
    
    class BaseModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x).predicted_depth
    
    wrapped_model = BaseModelWrapper(base_model)
    
    _, val_loader = get_nyu_dataloaders(
        data_path=config['data']['nyu_path'],
        batch_size=config['data']['batch_size'],
        target_size=tuple(config['data']['target_size']),
        num_workers=config['data']['num_workers'],
    )
    
    loss_fn = DepthLoss(
        l1_weight=config['training']['l1_loss_weight'],
        grad_weight=config['training']['grad_loss_weight'],
    )
    
    metrics = evaluate(wrapped_model, val_loader, loss_fn, device, config['training']['use_amp'])
    
    print(f"\nZero-shot metrics:")
    print(f"  Loss:    {metrics['loss']:.4f}")
    print(f"  AbsRel:  {metrics['abs_rel']:.4f}")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  δ1:      {metrics['delta1']:.4f}")
    print(f"  δ2:      {metrics['delta2']:.4f}")
    print(f"  δ3:      {metrics['delta3']:.4f}")
    
    del base_model, wrapped_model
    torch.cuda.empty_cache()
    
    return metrics


def train(config):
    """Main training function."""
    
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    wandb.init(
        project=config['logging']['project'],
        name=config['logging']['run_name'],
        config=config,
    )
    
    zero_shot_metrics = evaluate_zero_shot(config, device)
    wandb.log({"zero_shot/" + k: v for k, v in zero_shot_metrics.items()})
    
    print("\n" + "="*60)
    print("Creating LoRA Model")
    print("="*60)
    
    model = DepthAnythingLoRA(
        model_name=config['model']['name'],
        lora_r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        lora_target_modules=config['model']['lora_target_modules'],
    ).to(device)
    
    print(f"Trainable params: {model.get_trainable_params():,}")
    print(f"Total params: {model.get_total_params():,}")
    print(f"Trainable %: {100 * model.get_trainable_params() / model.get_total_params():.2f}%")
    
    print("\nLoading data...")
    train_loader, val_loader = get_nyu_dataloaders(
        data_path=config['data']['nyu_path'],
        batch_size=config['data']['batch_size'],
        target_size=tuple(config['data']['target_size']),
        num_workers=config['data']['num_workers'],
    )
    
    loss_fn = DepthLoss(
        l1_weight=config['training']['l1_loss_weight'],
        grad_weight=config['training']['grad_loss_weight'],
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Early stopping setup
    early_stopping = None
    if config['training'].get('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 7),
            min_delta=config['training'].get('min_delta', 0.001),
            mode="max"  # We're tracking delta1 (higher is better)
        )
        print(f"Early stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_delta1 = 0
    global_step = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(device)
            depth_gt = batch['depth'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=config['training']['use_amp']):
                depth_pred = model(rgb)
                losses = loss_fn(depth_pred, depth_gt, valid_mask)
                loss = losses['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            if global_step % config['logging']['log_every'] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/l1_loss": losses['l1_loss'].item(),
                    "train/grad_loss": losses['grad_loss'].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + batch_idx / len(train_loader),
                }, step=global_step)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % config['training']['eval_every'] == 0:
            print("Running evaluation...")
            val_metrics = evaluate(model, val_loader, loss_fn, device, config['training']['use_amp'])
            
            print(f"Val metrics:")
            print(f"  Loss:    {val_metrics['loss']:.4f}")
            print(f"  AbsRel:  {val_metrics['abs_rel']:.4f}")
            print(f"  RMSE:    {val_metrics['rmse']:.4f}")
            print(f"  δ1:      {val_metrics['delta1']:.4f}")
            
            wandb.log({
                "val/" + k: v for k, v in val_metrics.items()
            }, step=global_step)
            
            # Save best model
            if val_metrics['delta1'] > best_delta1:
                best_delta1 = val_metrics['delta1']
                print(f"New best δ1: {best_delta1:.4f}, saving...")
                model.save_lora(str(checkpoint_dir / "best"))
                wandb.log({"val/best_delta1": best_delta1}, step=global_step)
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_metrics['delta1']):
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best δ1: {early_stopping.best_score:.4f}")
                    break
                else:
                    epochs_without_improvement = early_stopping.counter
                    print(f"Epochs without improvement: {epochs_without_improvement}/{early_stopping.patience}")
        
        # Regular checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            model.save_lora(str(checkpoint_dir / f"epoch_{epoch+1}"))
    
    # Final save
    model.save_lora(str(checkpoint_dir / "final"))
    
    # Final comparison
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    
    final_metrics = evaluate(model, val_loader, loss_fn, device, config['training']['use_amp'])
    
    print("\nZero-shot vs LoRA-adapted:")
    print(f"{'Metric':<12} {'Zero-shot':>12} {'LoRA':>12} {'Δ':>12}")
    print("-" * 50)
    for key in ['abs_rel', 'rmse', 'delta1', 'delta2', 'delta3']:
        zs = zero_shot_metrics[key]
        lora = final_metrics[key]
        delta = lora - zs
        # For abs_rel and rmse, lower is better; for delta, higher is better
        if key in ['abs_rel', 'rmse']:
            better = "↑" if delta < 0 else "↓"
        else:
            better = "↑" if delta > 0 else "↓"
        print(f"{key:<12} {zs:>12.4f} {lora:>12.4f} {delta:>+11.4f} {better}")
    
    wandb.log({
        "final/" + k: v for k, v in final_metrics.items()
    })
    
    wandb.finish()
    print(f"\nTraining complete! Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/depth_lora.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)