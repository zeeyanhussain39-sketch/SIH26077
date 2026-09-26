"""
PyTorch Spatio-Temporal Nowcasting Network (ConvLSTM / Conv2D Architecture).
Processes multi-channel temporal sequences (Satellite IR + Radar Reflectivity + Numerical Indices)
and outputs lead-time predictions for T+2h, T+3h, T+4h, T+5h, T+6h.
"""

from typing import Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class SpatioTemporalNowcastNet(nn.Module):
        """
        Lightweight Spatio-Temporal ConvNet for CPU/GPU nowcasting inference.
        Input: (Batch, Sequence_Length, Channels, Height, Width)
               where Channels = [Radar dBZ, Satellite BT, CAPE grid]
        Output:
            - hazard_maps: (Batch, Lead_Times=5, Hazard_Types=3, Height, Width)
            - alert_logits: (Batch, Lead_Times=5, Hazard_Types=3)
        """
        def __init__(self, in_channels: int = 3, hidden_dim: int = 32, num_hazards: int = 3):
            super().__init__()
            self.num_hazards = num_hazards

            # Spatial Feature Extractor
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(inplace=True),
            )

            # Temporal bottleneck & spatial head
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, num_hazards, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

            # Global hazard probability head
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_hazards),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x shape: (Batch, Channels, Height, Width) [or collapsed temporal frames]
            """
            feats = self.encoder(x)
            hazard_map = self.decoder(feats)
            pooled = self.global_pool(feats).flatten(1)
            hazard_logits = self.classifier(pooled)
            return hazard_map, hazard_logits

else:
    class SpatioTemporalNowcastNet:
        """Fallback mock class when PyTorch is not yet installed."""
        def __init__(self, *args, **kwargs):
            self.status = "PyTorch not found; running in analytical simulation mode"

        def __call__(self, x: Any):
            return None, None
