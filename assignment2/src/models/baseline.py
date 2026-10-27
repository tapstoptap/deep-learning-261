from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for Food-101.
    Architecture per Proposal Section 8:
        Input image (3, 224, 224)
            ↓
        Convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
            ↓
        Global Average Pooling
            ↓
        Fully Connected Classification Head
            ↓
        101-class output
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 101,
        channels: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers: list[nn.Module] = []
        current_in = in_channels
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            current_in = out_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)
