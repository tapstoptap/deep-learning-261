from __future__ import annotations

import torch


def images_to_sequence(
    images: torch.Tensor,
    representation: str,
    patch_size: int = 4,
) -> torch.Tensor:
    """Convert a batch of 1-channel images into sequence tokens.

    The returned tensor has shape ``(batch, sequence_length, token_width)``.
    Rows and columns both produce 28 tokens of width 28 for Fashion-MNIST.
    Non-overlapping patches produce ``(28 / patch_size) ** 2`` tokens, each
    with ``patch_size ** 2`` values.
    """
    if images.ndim != 4 or images.size(1) != 1:
        raise ValueError(
            "Expected images with shape (batch, 1, height, width), "
            f"received {tuple(images.shape)}"
        )

    image_plane = images[:, 0]
    if representation == "image_rows":
        return image_plane
    if representation == "image_columns":
        return image_plane.transpose(1, 2)
    if representation == "image_patches":
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        height, width = image_plane.shape[-2:]
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Image shape {(height, width)} is not divisible by patch_size "
                f"{patch_size}"
            )
        patches = image_plane.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        return patches.contiguous().view(
            image_plane.size(0), -1, patch_size * patch_size
        )

    raise ValueError(f"Unsupported image representation: {representation}")
