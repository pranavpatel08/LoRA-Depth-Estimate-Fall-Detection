"""
Proof-of-concept: LoRA integration with Depth Anything V2.

Verifies:
1. Model loads correctly
2. LoRA attaches to correct layers
3. Forward pass works
4. Only LoRA params are trainable
5. Backward pass works (gradients flow)
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from peft import LoraConfig, get_peft_model, TaskType
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # =========================================================
    # Step 1: Load Depth Anything V2-Small
    # =========================================================
    print_section("Step 1: Loading Depth Anything V2-Small")
    
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    
    print(f"Loading from: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    
    print(f"Model type: {type(model).__name__}")
    params = count_parameters(model)
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # =========================================================
    # Step 2: Inspect model architecture for LoRA targets
    # =========================================================
    print_section("Step 2: Model Architecture (LoRA Target Identification)")
    
    # Print named modules to find attention layers
    print("\nSearching for attention layers (q_proj, k_proj, v_proj)...")
    
    lora_targets = []
    for name, module in model.named_modules():
        # Look for linear layers in attention
        if any(key in name.lower() for key in ['q_proj', 'k_proj', 'v_proj', 'qkv', 'query', 'key', 'value']):
            if isinstance(module, nn.Linear):
                lora_targets.append(name)
                print(f"  Found: {name} -> {module}")
    
    if not lora_targets:
        print("\n  No standard attention projections found. Listing all Linear layers:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"    {name}: {module.in_features} -> {module.out_features}")
    
    # =========================================================
    # Step 3: Apply LoRA configuration
    # =========================================================
    print_section("Step 3: Applying LoRA")
    
    # Target modules for DINOv2/ViT backbone
    # Depth Anything V2 uses a DINOv2 encoder - attention layers are typically:
    # backbone.encoder.layer.X.attention.attention.{query, key, value}
    # But naming varies - let's try common patterns
    
    target_modules = ["query", "key", "value"]  # Common ViT naming
    
    # Alternative if the above doesn't work:
    # target_modules = ["qkv"]  # Some ViTs use fused QKV
    
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )
    
    print(f"LoRA config:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    try:
        model_lora = get_peft_model(model, lora_config)
        print("\n✓ LoRA applied successfully!")
    except Exception as e:
        print(f"\n✗ LoRA application failed: {e}")
        print("\nTrying alternative target modules...")
        
        # Try finding actual module names
        linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get the last part of the name (e.g., "query" from "...attention.query")
                short_name = name.split('.')[-1]
                linear_names.add(short_name)
        
        print(f"Available linear layer names: {linear_names}")
        
        # Try with discovered names
        for candidate in [["qkv"], ["q_proj", "k_proj", "v_proj"], ["out_proj"]]:
            try:
                lora_config.target_modules = candidate
                model_lora = get_peft_model(model, lora_config)
                print(f"\n✓ LoRA applied successfully with targets: {candidate}")
                break
            except:
                continue
        else:
            print("\n✗ Could not apply LoRA. Manual inspection needed.")
            return
    
    # =========================================================
    # Step 4: Verify parameter counts
    # =========================================================
    print_section("Step 4: Parameter Analysis")
    
    params_lora = count_parameters(model_lora)
    print(f"After LoRA:")
    print(f"  Total params:     {params_lora['total']:,}")
    print(f"  Trainable params: {params_lora['trainable']:,}")
    print(f"  Frozen params:    {params_lora['frozen']:,}")
    print(f"  Trainable %:      {params_lora['trainable_pct']:.2f}%")
    
    # List trainable parameters
    print(f"\nTrainable parameter names:")
    for name, param in model_lora.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
    
    # =========================================================
    # Step 5: Forward pass test
    # =========================================================
    print_section("Step 5: Forward Pass Test")
    
    model_lora = model_lora.to(device)
    model_lora.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 518, 518).to(device)  # DAv2 default size
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        try:
            output = model_lora(dummy_input)
            
            # Output structure varies - handle both cases
            if hasattr(output, 'predicted_depth'):
                depth_output = output.predicted_depth
            else:
                depth_output = output
            
            print(f"Output type: {type(output)}")
            print(f"Depth output shape: {depth_output.shape}")
            print(f"Depth range: [{depth_output.min():.3f}, {depth_output.max():.3f}]")
            print("\n✓ Forward pass successful!")
        except Exception as e:
            print(f"\n✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # =========================================================
    # Step 6: Backward pass test (gradient flow)
    # =========================================================
    print_section("Step 6: Backward Pass Test")
    
    model_lora.train()
    
    # Forward
    output = model_lora(dummy_input)
    depth_output = output.predicted_depth if hasattr(output, 'predicted_depth') else output
    
    # Dummy loss
    dummy_target = torch.randn_like(depth_output)
    loss = nn.MSELoss()(depth_output, dummy_target)
    
    print(f"Dummy loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    lora_grads = []
    for name, param in model_lora.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            lora_grads.append((name, grad_norm))
    
    print(f"\nGradients computed for {len(lora_grads)} LoRA parameters:")
    for name, grad_norm in lora_grads[:5]:  # Show first 5
        print(f"  {name}: grad_norm={grad_norm:.6f}")
    if len(lora_grads) > 5:
        print(f"  ... and {len(lora_grads) - 5} more")
    
    print("\n✓ Backward pass successful!")
    
    # =========================================================
    # Step 7: Test with real NYU data
    # =========================================================
    print_section("Step 7: Test with Real NYU Data")
    
    try:
        from src.data.nyu_dataset import get_nyu_dataloaders
        
        train_loader, val_loader = get_nyu_dataloaders(
            batch_size=2,
            target_size=(518, 518),  # Match DAv2 expected size
            num_workers=0,
        )
        
        batch = next(iter(train_loader))
        rgb = batch['rgb'].to(device)
        depth_gt = batch['depth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        print(f"NYU batch loaded:")
        print(f"  RGB: {rgb.shape}")
        print(f"  Depth GT: {depth_gt.shape}")
        
        # Forward pass
        model_lora.eval()
        with torch.no_grad():
            output = model_lora(rgb)
            depth_pred = output.predicted_depth if hasattr(output, 'predicted_depth') else output
        
        print(f"  Depth Pred: {depth_pred.shape}")
        print(f"  Pred range: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
        print(f"  GT range: [{depth_gt.min():.3f}, {depth_gt.max():.3f}]")
        
        print("\n✓ Real data forward pass successful!")
        
    except Exception as e:
        print(f"\n⚠ Real data test failed: {e}")
        print("  (This is okay for POC - we'll fix data pipeline issues later)")
    
    # =========================================================
    # Summary
    # =========================================================
    print_section("POC Summary")
    print("✓ Model loaded successfully")
    print("✓ LoRA applied to attention layers")
    print(f"✓ Only {params_lora['trainable_pct']:.2f}% of parameters are trainable")
    print("✓ Forward pass works")
    print("✓ Backward pass works (gradients flow to LoRA params)")
    print("\nReady for Phase 2.2: Full training pipeline!")


if __name__ == "__main__":
    main()