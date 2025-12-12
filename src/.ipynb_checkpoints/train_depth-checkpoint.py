"""
Training script for Depth Anything V2 with LoRA adaptation.
"""

import os
import sys
import argparse
import yaml
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from transformers import AutoModelForDepthEstimation
from peft import LoraConfig, get_peft_model

# Local imports
from src.data.nyu_dataset import get_nyu_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class DepthLoss(nn.Module):
    """
    Combined loss for depth estimation.
    
    Components:
    - Scale-invariant loss (SILog)
    - Gradient loss for edge preservation
    """
    
    def __init__(self, si_weight: float = 1.0, grad_weight: float = 0.5):
        super().__init__()
        self.si_weight = si_weight
        self.grad_weight = grad_weight
    
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss for edge-aware depth."""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # Compute gradients
        pred_grad_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
        target_grad_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_grad_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)
        
        # Masked L1 loss on gradients
        mask_eroded = mask[:, :, 1:-1, 1:-1]  # Account for conv padding
        diff_x = torch.abs(pred_grad_x - target_grad_x)[:, :, 1:-1, 1:-1]
        diff_y = torch.abs(pred_grad_y - target_grad_y)[:, :, 1:-1, 1:-1]
        
        loss = (diff_x * mask_eroded).sum() + (diff_y * mask_eroded).sum()
        loss = loss / (mask_eroded.sum() + 1e-8)
        
        return loss
    
    def silog_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scale-invariant logarithmic loss."""
        # Add small epsilon to avoid log(0)
        pred_log = torch.log(pred + 1e-8)
        target_log = torch.log(target + 1e-8)
        
        diff = (pred_log - target_log) * mask
        
        n_valid = mask.sum() + 1e-8
        
        loss = torch.sqrt((diff ** 2).sum() / n_valid - 0.5 * (diff.sum() ** 2) / (n_valid ** 2))
        
        return loss
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> dict:
        """
        Args:
            pred: (B, 1, H, W) predicted depth
            target: (B, 1, H, W) ground truth depth
            mask: (B, 1, H, W) valid pixel mask
        """
        si_loss = self.silog_loss(pred, target, mask)
        grad_loss = self.gradient_loss(pred, target, mask)
        
        total_loss = self.si_weight * si_loss + self.grad_weight * grad_loss
        
        return {
            "loss": total_loss,
            "si_loss": si_loss,
            "grad_loss": grad_loss,
        }


class DepthMetrics:
    """Compute standard depth estimation metrics."""
    
    @staticmethod
    def compute(
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> dict:
        """
        Compute depth metrics.
        
        Args:
            pred: (B, 1, H, W) predicted depth
            target: (B, 1, H, W) ground truth depth  
            mask: (B, 1, H, W) valid pixel mask
        
        Returns:
            Dictionary with AbsRel, RMSE, delta1, delta2, delta3
        """
        pred = pred[mask > 0]
        target = target[mask > 0]
        
        if len(pred) == 0:
            return {
                "abs_rel": 0.0,
                "rmse": 0.0,
                "delta1": 0.0,
                "delta2": 0.0,
                "delta3": 0.0,
            }
        
        # Absolute relative error
        abs_rel = torch.mean(torch.abs(pred - target) / target).item()
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
        
        # Threshold accuracy (delta)
        thresh = torch.max(pred / target, target / pred)
        delta1 = (thresh < 1.25).float().mean().item()
        delta2 = (thresh < 1.25 ** 2).float().mean().item()
        delta3 = (thresh < 1.25 ** 3).float().mean().item()
        
        return {
            "abs_rel": abs_rel,
            "rmse": rmse,
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
        }


def align_depth(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Align predicted depth to target using least squares (scale and shift).
    
    Depth Anything outputs relative depth; we need to align to metric depth.
    """
    pred_flat = pred[mask > 0]
    target_flat = target[mask > 0]
    
    if len(pred_flat) < 10:
        return pred
    
    # Solve: target = scale * pred + shift
    # Using least squares: [pred, 1] @ [scale, shift]^T = target
    A = torch.stack([pred_flat, torch.ones_like(pred_flat)], dim=1)
    b = target_flat

    A = A.float()
    b = b.float()
    
    # Least squares solution
    result = torch.linalg.lstsq(A, b.unsqueeze(1))
    params = result.solution.squeeze()
    
    scale, shift = params[0], params[1]
    
    aligned = scale * pred + shift
    aligned = torch.clamp(aligned, min=0.001)  # Ensure positive depth
    
    return aligned


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    loss_fn: DepthLoss,
    scaler: GradScaler,
    device: torch.device,
    config: dict,
    epoch: int,
    wandb_run=None,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_si_loss = 0.0
    total_grad_loss = 0.0
    n_batches = 0
    
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    log_every = config['logging']['log_every_n_steps']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device)
        depth_gt = batch['depth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['training']['use_amp']):
            output = model(rgb)
            depth_pred = output.predicted_depth.unsqueeze(1)  # (B, 1, H, W)
            
            # Resize prediction to match GT if needed
            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                depth_pred = torch.nn.functional.interpolate(
                    depth_pred, 
                    size=depth_gt.shape[-2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Align depth (scale/shift)
            depth_pred_aligned = align_depth(depth_pred, depth_gt, valid_mask)
            
            # Compute loss
            losses = loss_fn(depth_pred_aligned, depth_gt, valid_mask)
            loss = losses['loss'] / grad_accum_steps

        print(f"Loss shape: {loss.shape}, value: {loss.item()}")
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Logging
        total_loss += losses['loss'].item()
        total_si_loss += losses['si_loss'].item()
        total_grad_loss += losses['grad_loss'].item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # WandB logging
        if wandb_run and batch_idx % log_every == 0:
            wandb_run.log({
                "train/loss": losses['loss'].item(),
                "train/si_loss": losses['si_loss'].item(),
                "train/grad_loss": losses['grad_loss'].item(),
                "train/lr": scheduler.get_last_lr()[0],
            })
    
    return {
        "loss": total_loss / n_batches,
        "si_loss": total_si_loss / n_batches,
        "grad_loss": total_grad_loss / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    loss_fn: DepthLoss,
    device: torch.device,
    config: dict,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    all_metrics = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        rgb = batch['rgb'].to(device)
        depth_gt = batch['depth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # Forward
        output = model(rgb)
        depth_pred = output.predicted_depth.unsqueeze(1)
        
        # Resize if needed
        if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
            depth_pred = torch.nn.functional.interpolate(
                depth_pred, 
                size=depth_gt.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Align
        depth_pred_aligned = align_depth(depth_pred, depth_gt, valid_mask)
        
        # Loss
        losses = loss_fn(depth_pred_aligned, depth_gt, valid_mask)
        total_loss += losses['loss'].item()
        
        # Metrics
        metrics = DepthMetrics.compute(depth_pred_aligned, depth_gt, valid_mask)
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(val_loader)
    
    return avg_metrics


@torch.no_grad()
def evaluate_zero_shot(
    model_name: str,
    val_loader,
    loss_fn: DepthLoss,
    device: torch.device,
) -> dict:
    """Evaluate zero-shot (no LoRA) performance."""
    print("\nEvaluating zero-shot performance...")
    
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    return evaluate(model, val_loader, loss_fn, device, {"training": {"use_amp": True}})


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: dict,
    path: Path,
):
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")
    
    # Also save just the LoRA adapter
    adapter_path = path.parent / f"lora_adapter_epoch{epoch}"
    model.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter: {adapter_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    wandb_run = None
    if config['logging']['use_wandb']:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['logging']['project_name'],
                name=config['logging']['run_name'],
                config=config,
            )
        except ImportError:
            print("WandB not available, skipping...")
    
    # Create output dirs
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader = get_nyu_dataloaders(
        data_path=config['data']['nyu_path'],
        batch_size=config['training']['batch_size'],
        target_size=(config['data']['input_size'], config['data']['input_size']),
        num_workers=4,
    )
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForDepthEstimation.from_pretrained(config['model']['name'])
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model = model.to(device)
    
    # Loss function
    loss_fn = DepthLoss(si_weight=1.0, grad_weight=0.5)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Scheduler (warmup + cosine decay)
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = len(train_loader) * config['training']['warmup_epochs']
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=num_warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_training_steps - num_warmup_steps
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Evaluate zero-shot baseline
    print("\n" + "="*60)
    print("Zero-shot baseline evaluation")
    print("="*60)
    zero_shot_metrics = evaluate_zero_shot(
        config['model']['name'], 
        val_loader, 
        loss_fn, 
        device
    )
    print(f"Zero-shot metrics:")
    print(f"  AbsRel: {zero_shot_metrics['abs_rel']:.4f}")
    print(f"  RMSE:   {zero_shot_metrics['rmse']:.4f}")
    print(f"  δ1:     {zero_shot_metrics['delta1']:.4f}")
    
    if wandb_run:
        wandb_run.log({"zero_shot/" + k: v for k, v in zero_shot_metrics.items()})
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)
    
    best_delta1 = 0.0
    start_epoch = 1
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['training']['num_epochs']} ---")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, scaler, device, config, epoch, wandb_run
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        if epoch % config['training']['eval_every_n_epochs'] == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, device, config)
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"AbsRel: {val_metrics['abs_rel']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}, "
                  f"δ1: {val_metrics['delta1']:.4f}")
            
            if wandb_run:
                wandb_run.log({
                    "val/loss": val_metrics['loss'],
                    "val/abs_rel": val_metrics['abs_rel'],
                    "val/rmse": val_metrics['rmse'],
                    "val/delta1": val_metrics['delta1'],
                    "val/delta2": val_metrics['delta2'],
                    "val/delta3": val_metrics['delta3'],
                    "epoch": epoch,
                })
            
            # Save best model
            if val_metrics['delta1'] > best_delta1:
                best_delta1 = val_metrics['delta1']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, config,
                    checkpoint_dir / "best_model.pt"
                )
        
        # Periodic checkpoint
        if epoch % config['training']['save_every_n_epochs'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, config,
                checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
            )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    final_metrics = evaluate(model, val_loader, loss_fn, device, config)
    
    print(f"\nZero-shot → LoRA-adapted:")
    print(f"  AbsRel: {zero_shot_metrics['abs_rel']:.4f} → {final_metrics['abs_rel']:.4f}")
    print(f"  RMSE:   {zero_shot_metrics['rmse']:.4f} → {final_metrics['rmse']:.4f}")
    print(f"  δ1:     {zero_shot_metrics['delta1']:.4f} → {final_metrics['delta1']:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch, final_metrics, config,
        checkpoint_dir / "final_model.pt"
    )
    
    if wandb_run:
        wandb_run.finish()
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()