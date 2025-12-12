"""
Temporal fall detection model using depth sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatiotemporal modeling.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Combined gates for efficiency
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input tensor
            hidden: Tuple of (h, c) each (B, hidden_channels, H, W)
            
        Returns:
            (h_new, c_new): New hidden and cell states
        """
        B, _, H, W = x.shape
        
        if hidden is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        else:
            h, c = hidden
        
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: list,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        
        layers = []
        for i in range(self.num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            layers.append(ConvLSTMCell(in_ch, hidden_channels[i], kernel_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (B, T, C, H, W) input sequence
            hidden: List of (h, c) tuples for each layer
            
        Returns:
            output: (B, T, hidden_channels[-1], H, W) output sequence
            hidden: Updated hidden states
        """
        B, T, C, H, W = x.shape
        
        if hidden is None:
            hidden = [None] * self.num_layers
        
        # Process sequence
        outputs = []
        for t in range(T):
            x_t = x[:, t]  # (B, C, H, W)
            
            for layer_idx, layer in enumerate(self.layers):
                x_t, c_t = layer(x_t, hidden[layer_idx])
                hidden[layer_idx] = (x_t, c_t)
            
            outputs.append(x_t)
        
        output = torch.stack(outputs, dim=1)  # (B, T, C, H, W)
        
        return output, hidden


class SpatialEncoder(nn.Module):
    """
    Lightweight CNN encoder for depth or RGB frames.
    Extracts spatial features from each frame independently.
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # 1 for depth, 3 for RGB
        base_channels: int = 32,
        out_channels: int = 64,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: (in_channels, H, W) -> (32, H/2, W/2)
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Block 2: (32, H/2, W/2) -> (64, H/4, W/4)
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Block 3: (64, H/4, W/4) -> (out_channels, H/8, W/8)
            nn.Conv2d(base_channels * 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) sequence or (B, C, H, W) single frame
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self.encoder(x)
            _, C_out, H_out, W_out = features.shape
            features = features.view(B, T, C_out, H_out, W_out)
        else:
            features = self.encoder(x)
        
        return features


class FallDetector(nn.Module):
    """
    Complete fall detection model combining spatial and temporal processing.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        spatial_channels: int = 64,
        temporal_channels: list = None,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_channels: Input depth channels (1 for depth)
            spatial_channels: Output channels from spatial encoder
            temporal_channels: List of ConvLSTM hidden channels
            num_classes: Number of output classes (2 for fall/no-fall)
            dropout: Dropout rate for classification head
        """
        super().__init__()
        
        if temporal_channels is None:
            temporal_channels = [64, 128]
        
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            base_channels=32,
            out_channels=spatial_channels,
        )
        
        self.temporal_model = ConvLSTM(
            input_channels=spatial_channels,
            hidden_channels=temporal_channels,
            kernel_size=3,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 1, H, W) depth sequence
            return_features: If True, also return intermediate features
            
        Returns:
            Dictionary with 'logits', 'probs', and optionally 'features'
        """
        B, T, C, H, W = x.shape
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(x)  # (B, T, C, h, w)
        
        # Temporal modeling
        temporal_features, _ = self.temporal_model(spatial_features)  # (B, T, C, h, w)
        
        # Use last timestep for classification
        final_features = temporal_features[:, -1]  # (B, C, h, w)
        
        # Classification
        logits = self.classifier(final_features)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
        }
        
        if return_features:
            output['spatial_features'] = spatial_features
            output['temporal_features'] = temporal_features
        
        return output
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.
        
        Returns:
            predictions: (B,) predicted class indices
            confidences: (B,) confidence scores for predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = output['probs']
            confidences, predictions = probs.max(dim=-1)
        return predictions, confidences


# Quick test
if __name__ == "__main__":
    print("Testing FallDetector...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = FallDetector(
        in_channels=1,
        spatial_channels=64,
        temporal_channels=[64, 128],
        num_classes=2,
        dropout=0.5,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Test forward pass
    B, T, C, H, W = 4, 16, 1, 224, 224
    x = torch.randn(B, T, C, H, W).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    output = model(x, return_features=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Spatial features shape: {output['spatial_features'].shape}")
    print(f"Temporal features shape: {output['temporal_features'].shape}")
    
    # Test prediction
    preds, confs = model.predict(x)
    print(f"\nPredictions: {preds.tolist()}")
    print(f"Confidences: {[f'{c:.3f}' for c in confs.tolist()]}")
    
    # Test backward pass
    model.train()
    output = model(x)
    loss = F.cross_entropy(output['logits'], torch.randint(0, 2, (B,)).to(device))
    loss.backward()
    print(f"\nBackward pass successful, loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")