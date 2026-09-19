from __future__ import annotations

import torch
from torch import nn


class TransformerImageClassifier(nn.Module):
    def __init__(
        self,
        representation: str,
        sequence_length: int,
        token_input_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        feedforward_dim: int,
        dropout: float,
        pooling: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.representation = representation
        self.pooling = pooling
        self.input_projection = nn.Linear(token_input_dim, embedding_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, embedding_dim)
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def images_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        image_rows = images.squeeze(1)
        if self.representation == "image_rows":
            return image_rows
        if self.representation == "image_columns":
            return image_rows.transpose(1, 2)
        raise ValueError(f"Unsupported image representation: {self.representation}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.images_to_tokens(images)
        token_embeddings = self.input_projection(tokens)
        token_embeddings = token_embeddings + self.positional_embedding[:, : tokens.size(1)]
        encoded_tokens = self.encoder(token_embeddings)

        if self.pooling == "mean":
            image_representation = encoded_tokens.mean(dim=1)
        elif self.pooling == "last":
            image_representation = encoded_tokens[:, -1, :]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        return self.classifier(image_representation)
