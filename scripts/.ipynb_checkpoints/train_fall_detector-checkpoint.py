"""
Train fall detection model on UR Fall dataset.
"""

import os
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fall_detector import FallDetector
from src.data.urfall_dataset import get_urfall_dataloaders


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.005, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
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


def compute_class_weights(dataloader) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    labels = []
    for batch in dataloader:
        labels.extend(batch['label'].tolist())
    
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(counter)
    
    weights = []
    for i in range(num_classes):
        weight = total / (num_classes * counter.get(i, 1))
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, dataloader, criterion, device, use_amp=True):
    """Run evaluation and compute metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            depth = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(enabled=use_amp):
                output = model(depth)
                loss = criterion(output['logits'], labels)
            
            total_loss += loss.item()
            
            probs = output['probs'][:, 1]  # Probability of fall (class 1)
            preds = output['logits'].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # Default if only one class present
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
    }
    
    return metrics


def train(config):
    """Main training function."""
    
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Initialize wandb
    wandb.init(
        project=config['logging']['project'],
        name=config['logging']['run_name'],
        config=config,
    )
    
    # Data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_urfall_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
        window_size=config['data']['window_size'],
        window_stride=config['data']['window_stride'],
        target_size=tuple(config['data']['target_size']),
        modality=config['data']['modality'],
        num_workers=config['data']['num_workers'],
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Compute class weights
    if config['training']['use_class_weights']:
        print("\nComputing class weights...")
        class_weights = compute_class_weights(train_loader).to(device)
        print(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = None
    
    # Create model
    print("\nCreating model...")
    model = FallDetector(
        in_channels=config['model']['in_channels'],
        spatial_channels=config['model']['spatial_channels'],
        temporal_channels=config['model']['temporal_channels'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Early stopping
    early_stopping = None
    if config['training'].get('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 15),
            min_delta=config['training'].get('min_delta', 0.005),
            mode="max"
        )
        print(f"\nEarly stopping: patience={early_stopping.patience}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_f1 = 0
    global_step = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            depth = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=config['training']['use_amp']):
                output = model(depth)
                loss = criterion(output['logits'], labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds = output['logits'].argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{epoch_correct/epoch_total:.3f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            if global_step % config['logging']['log_every'] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + batch_idx / len(train_loader),
                }, step=global_step)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = epoch_correct / epoch_total
        print(f"\nEpoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_acc": epoch_acc,
        }, step=global_step)
        
        # Evaluation
        if (epoch + 1) % config['training']['eval_every'] == 0:
            print("Running validation...")
            val_metrics = evaluate(model, val_loader, criterion, device, config['training']['use_amp'])
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"P: {val_metrics['precision']:.4f}, "
                  f"R: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}")
            
            wandb.log({
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
                "val/auc": val_metrics['auc'],
            }, step=global_step)
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                print(f"New best F1: {best_f1:.4f}, saving...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1': best_f1,
                    'config': config,
                }, checkpoint_dir / "best.pt")
                wandb.log({"val/best_f1": best_f1}, step=global_step)
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_metrics['f1']):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best F1: {early_stopping.best_score:.4f}")
                    break
                else:
                    print(f"Patience: {early_stopping.counter}/{early_stopping.patience}")
        
        # Regular checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_dir / f"epoch_{epoch+1}.pt")
    
    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_dir / "best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, config['training']['use_amp'])
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {test_metrics['confusion_matrix']}")
    
    wandb.log({
        "test/accuracy": test_metrics['accuracy'],
        "test/precision": test_metrics['precision'],
        "test/recall": test_metrics['recall'],
        "test/f1": test_metrics['f1'],
        "test/auc": test_metrics['auc'],
    })
    
    # Save final results
    results = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'best_val_f1': best_f1,
    }
    
    with open(output_dir / "results.yaml", 'w') as f:
        yaml.dump(results, f)
    
    wandb.finish()
    print(f"\nTraining complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fall_detection.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)