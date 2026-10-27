from __future__ import annotations

from typing import Any
import torch
from torch import nn

from models.baseline import SimpleCNN
from models.pretrained import ResNet50Classifier


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_model(config: dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate models based on configuration dictionary.
    """
    model_name = config.get("name", "simple_cnn").lower()
    num_classes = config.get("num_classes", 101)

    if model_name in {"simple_cnn", "baseline", "cnn"}:
        channels = config.get("channels", [32, 64, 128, 256])
        dropout = config.get("dropout", 0.3)
        return SimpleCNN(
            in_channels=config.get("in_channels", 3),
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
        )
    elif model_name in {"resnet50", "resnet_50", "pretrained"}:
        pretrained = config.get("pretrained", True)
        freeze_backbone = config.get("freeze_backbone", False)
        dropout = config.get("dropout", 0.2)
        return ResNet50Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model name in config: {model_name}")


__all__ = ["SimpleCNN", "ResNet50Classifier", "create_model", "count_parameters"]
