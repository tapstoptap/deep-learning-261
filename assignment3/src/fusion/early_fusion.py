"""Simple fusion: L2-normalised image and question features, concatenated -> MLP.

The fusion point is after the modality encoders and before the classifier
(intermediate, feature-level fusion).
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.head import mlp_head


class ConcatFusion(nn.Module):
    def __init__(self, n_classes, hidden=1024, dropout=0.5, image_dim=2048, text_dim=768):
        super().__init__()
        self.head = mlp_head(image_dim + text_dim, n_classes, hidden, dropout)

    def forward(self, image, text):
        joint = torch.cat([F.normalize(image, dim=-1), F.normalize(text, dim=-1)], dim=-1)
        return self.head(joint)
