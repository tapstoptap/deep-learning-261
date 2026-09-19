from __future__ import annotations

import torch
from torch import nn


class ConvolutionalClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_channels: list[int],
        kernel_size: int,
        pooling_size: int,
        classifier_hidden_dim: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 2:
            raise ValueError("The CNN config must provide exactly two conv channel sizes")

        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels[0], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
        )

        feature_size = 28 // (pooling_size**2)
        flattened_feature_dim = conv_channels[1] * feature_size * feature_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_feature_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)
