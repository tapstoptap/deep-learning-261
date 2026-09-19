from __future__ import annotations

import torch
from torch import nn


def create_activation(activation_name: str) -> nn.Module:
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation_name}")


class MultilayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLP requires at least one hidden layer")

        layers: list[nn.Module] = [nn.Flatten()]
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    create_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)
