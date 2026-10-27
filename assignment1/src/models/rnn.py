from __future__ import annotations

import torch
from torch import nn

from .representations import images_to_sequence


class RecurrentImageClassifier(nn.Module):
    def __init__(
        self,
        recurrent_type: str,
        representation: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.patch_size = patch_size

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        recurrent_type = recurrent_type.lower()
        if recurrent_type == "gru":
            recurrent_class = nn.GRU
        elif recurrent_type == "lstm":
            recurrent_class = nn.LSTM
        else:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")

        self.recurrent_layer = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=recurrent_dropout,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(output_dim, num_classes)

    def images_to_sequence(self, images: torch.Tensor) -> torch.Tensor:
        return images_to_sequence(images, self.representation, self.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sequence = self.images_to_sequence(images)
        sequence_outputs, _ = self.recurrent_layer(sequence)
        final_timestep = sequence_outputs[:, -1, :]
        return self.classifier(final_timestep)
