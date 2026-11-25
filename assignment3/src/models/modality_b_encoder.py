"""Unimodal B (question only): frozen BERT-base [CLS] feature -> MLP. The image is ignored."""
import torch.nn.functional as F
from torch import nn

from models.head import mlp_head


class TextOnly(nn.Module):
    def __init__(self, n_classes, hidden=1024, dropout=0.5, text_dim=768):
        super().__init__()
        self.head = mlp_head(text_dim, n_classes, hidden, dropout)

    def forward(self, image, text):
        return self.head(F.normalize(text, dim=-1))
