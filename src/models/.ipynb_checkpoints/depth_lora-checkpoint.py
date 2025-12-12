"""
LoRA-adapted Depth Anything V2 for indoor depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Tuple


class DepthAnythingLoRA(nn.Module):
    """
    Depth Anything V2 with LoRA adapters for domain adaptation.
    """
    
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Load base model
        self.base_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        
        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["query", "key", "value"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Store config for saving
        self.lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": lora_target_modules,
        }
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: (B, 3, H, W) RGB images, normalized to [0, 1]
            
        Returns:
            Predicted depth map (B, H, W)
        """
        output = self.model(pixel_values)
        return output.predicted_depth
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def save_lora(self, path: str):
        """Save only the LoRA adapters."""
        self.model.save_pretrained(path)
    
    def load_lora(self, path: str):
        """Load LoRA adapters."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)


class DepthLoss(nn.Module):
    """
    Combined loss for depth estimation with scale-shift alignment.
    
    Handles the scale ambiguity between predicted relative depth
    and ground truth metric depth.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        grad_weight: float = 0.5,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
    
    def align_pred_to_gt(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Align prediction to ground truth using least squares.
        Finds scale (s) and shift (t) such that: aligned = s * pred + t
        
        IMPORTANT: Scale and shift are computed with no_grad to prevent
        degenerate optimization where the model games the alignment.
        """
        with torch.no_grad():
            pred_flat = pred[mask > 0]
            target_flat = target[mask > 0]
            
            if len(pred_flat) < 10:
                scale = torch.tensor(1.0, device=pred.device)
                shift = torch.tensor(0.0, device=pred.device)
            else:
                pred_mean = pred_flat.mean()
                target_mean = target_flat.mean()
                
                pred_centered = pred_flat - pred_mean
                target_centered = target_flat - target_mean
                
                scale = (pred_centered * target_centered).sum() / (pred_centered ** 2).sum().clamp(min=1e-8)
                shift = target_mean - scale * pred_mean
                
                scale = scale.clamp(min=0.01, max=100.0)
        
        aligned = scale * pred + shift
        return aligned
    
    def l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Simple L1 loss on aligned predictions."""
        diff = torch.abs(pred - target) * mask
        return diff.sum() / mask.sum().clamp(min=1.0)
    
    def gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient matching loss for edge preservation."""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        
        loss_dx = torch.abs(pred_dx - target_dx) * mask_dx
        loss_dy = torch.abs(pred_dy - target_dy) * mask_dy
        
        n_valid = mask_dx.sum() + mask_dy.sum()
        
        return (loss_dx.sum() + loss_dy.sum()) / n_valid.clamp(min=1.0)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss with scale-shift alignment."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, 
                size=target.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        if mask is None:
            mask = (target > 0).float()
        
        pred_aligned = self.align_pred_to_gt(pred, target, mask)
        
        l1_loss = self.l1_loss(pred_aligned, target, mask)
        grad_loss = self.gradient_loss(pred_aligned, target, mask)
        
        total_loss = self.l1_weight * l1_loss + self.grad_weight * grad_loss
        
        return {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "grad_loss": grad_loss,
        }


class DepthMetrics:
    """Standard depth estimation metrics with scale alignment."""
    
    @staticmethod
    def align_pred_to_gt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Align prediction to GT using least squares (no gradients)."""
        with torch.no_grad():
            pred_flat = pred[mask > 0]
            target_flat = target[mask > 0]
            
            if len(pred_flat) < 10:
                return pred
            
            pred_mean = pred_flat.mean()
            target_mean = target_flat.mean()
            
            pred_centered = pred_flat - pred_mean
            target_centered = target_flat - target_mean
            
            scale = (pred_centered * target_centered).sum() / (pred_centered ** 2).sum().clamp(min=1e-8)
            shift = target_mean - scale * pred_mean
            
            scale = scale.clamp(min=0.01, max=100.0)
        
        return scale * pred + shift
    
    @staticmethod
    def compute(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute depth metrics with scale alignment."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        
        # Resize if needed
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        if mask is None:
            mask = (target > 0).float()
        
        # Align predictions
        pred_aligned = DepthMetrics.align_pred_to_gt(pred, target, mask)
        
        # Flatten for computation
        pred_flat = pred_aligned[mask > 0]
        target_flat = target[mask > 0]
        
        if len(pred_flat) == 0:
            return {"abs_rel": 0, "rmse": 0, "delta1": 0, "delta2": 0, "delta3": 0}
        
        # Clamp to valid range
        pred_flat = pred_flat.clamp(min=1e-3)
        target_flat = target_flat.clamp(min=1e-3)
        
        # Absolute Relative Error
        abs_rel = torch.mean(torch.abs(pred_flat - target_flat) / target_flat).item()
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred_flat - target_flat) ** 2)).item()
        
        # Threshold accuracy (delta)
        ratio = torch.max(pred_flat / target_flat, target_flat / pred_flat)
        delta1 = (ratio < 1.25).float().mean().item()
        delta2 = (ratio < 1.25 ** 2).float().mean().item()
        delta3 = (ratio < 1.25 ** 3).float().mean().item()
        
        return {
            "abs_rel": abs_rel,
            "rmse": rmse,
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
        }


# Quick test
if __name__ == "__main__":
    import os
    os.environ["USE_TF"] = "0"
    
    print("Testing DepthAnythingLoRA...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DepthAnythingLoRA(lora_r=8).to(device)
    print(f"Trainable params: {model.get_trainable_params():,}")
    print(f"Total params: {model.get_total_params():,}")
    print(f"Trainable %: {100 * model.get_trainable_params() / model.get_total_params():.2f}%")
    
    # Test forward
    x = torch.randn(2, 3, 518, 518).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    
    # Test loss
    loss_fn = DepthLoss()
    target = torch.rand(2, 1, 480, 640).to(device) * 10
    mask = torch.ones_like(target)
    losses = loss_fn(y, target, mask)
    print(f"Loss: {losses['loss'].item():.4f}")
    
    # Test metrics
    metrics = DepthMetrics.compute(y, target, mask)
    print(f"Metrics: {metrics}")
    
    print("\n✓ All tests passed!")