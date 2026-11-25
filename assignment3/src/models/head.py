"""Classifier head shared by the unimodal and fusion models."""
from torch import nn


def mlp_head(in_dim, n_classes, hidden=1024, dropout=0.5):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )
