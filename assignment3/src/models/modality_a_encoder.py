"""Unimodal A (image only): frozen ResNet-50 pooled feature -> MLP. The question is ignored."""
import torch.nn.functional as F
from torch import nn

from models.head import mlp_head


class ImageOnly(nn.Module):
    def __init__(self, n_classes, hidden=1024, dropout=0.5, image_dim=2048):
        super().__init__()
        self.head = mlp_head(image_dim, n_classes, hidden, dropout)

    def forward(self, image, text):
        return self.head(F.normalize(image, dim=-1))
