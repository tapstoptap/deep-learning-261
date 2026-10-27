from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Classifier(nn.Module):
    """
    Pretrained ResNet-50 classifier for Food-101.
    Per Proposal Section 9:
      - E2 (freeze_backbone=True): Feature extractor mode; only classification head is trained.
      - E3 (freeze_backbone=False): Full fine-tuning of all layers.
    """

    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the backbone parameters while keeping fc trainable."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
