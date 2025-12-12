"""
Run ablation study comparing fall detection with:
1. Real Kinect depth
2. Zero-shot estimated depth
3. LoRA-adapted estimated depth

Usage:
    python scripts/ablations/run_ablation.py
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fall_detector import FallDetector
from src.data.urfall_dataset import get_urfall_dataloaders


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            depth = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            output = model(depth)
            
            probs = output['probs'][:, 1]
            preds = output['logits'].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
    }


def train_and_evaluate(depth_source: str, config: dict, device: torch.device):
    """Train a model on given depth source and evaluate."""
    print(f"\n{'='*60}")
    print(f"Training with depth_source = '{depth_source}'")
    print('='*60)
    
    # Data loaders
    train_loader, val_loader, test_loader = get_urfall_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
        window_size=config['data']['window_size'],
        window_stride=config['data']['window_stride'],
        target_size=tuple(config['data']['target_size']),
        modality="depth",
        depth_source=depth_source,
        num_workers=config['data']['num_workers'],
    )
    
    print(f"Train: {len(train_loader.dataset)} windows")
    print(f"Val: {len(val_loader.dataset)} windows")
    print(f"Test: {len(test_loader.dataset)} windows")
    
    # Model
    model = FallDetector(
        in_channels=1,
        spatial_channels=64,
        temporal_channels=[64, 128],
        num_classes=2,
        dropout=0.5,
    ).to(device)
    
    # Loss with class weights
    from collections import Counter
    labels = [batch['label'].item() for batch in train_loader.dataset]
    counter = Counter(labels)
    total = len(labels)
    weights = torch.tensor([total / (2 * counter[0]), total / (2 * counter[1])]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    patience = 8
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            depth = batch['depth'].to(device)
            labels_batch = batch['label'].to(device)
            
            optimizer.zero_grad()
            output = model(depth)
            loss = criterion(output['logits'], labels_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
              f"Val F1={val_metrics['f1']:.4f}, Val AUC={val_metrics['auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device)
    
    return test_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    config = {
        'data': {
            'data_path': 'data/ur_fall',
            'batch_size': 8,
            'window_size': 16,
            'window_stride': 8,
            'target_size': [224, 224],
            'num_workers': 4,
        }
    }
    
    results = {}
    
    # Run for each depth source
    for depth_source in ["real", "zeroshot", "lora"]:
        try:
            metrics = train_and_evaluate(depth_source, config, device)
            results[depth_source] = metrics
            
            print(f"\n{depth_source.upper()} Test Results:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"Error with {depth_source}: {e}")
            results[depth_source] = None
    
    # Summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Depth Source':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-"*70)
    
    for source in ["real", "zeroshot", "lora"]:
        if results.get(source):
            m = results[source]
            print(f"{source:<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}")
        else:
            print(f"{source:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    print("-"*70)
    
    # Save results
    output_dir = Path("outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for YAML serialization
    results_serializable = {}
    for k, v in results.items():
        if v:
            results_serializable[k] = {
                key: val.tolist() if hasattr(val, 'tolist') else val
                for key, val in v.items()
            }
    
    with open(output_dir / "results.yaml", 'w') as f:
        yaml.dump(results_serializable, f)
    
    print(f"\nResults saved to: {output_dir / 'results.yaml'}")


if __name__ == "__main__":
    main()
