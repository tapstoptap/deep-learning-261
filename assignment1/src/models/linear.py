from __future__ import annotations

import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened_images = images.flatten(start_dim=1)
        return self.classifier(flattened_images)
