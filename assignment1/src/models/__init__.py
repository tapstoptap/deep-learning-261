from __future__ import annotations

from typing import Any

from torch import nn

from .cnn import ConvolutionalClassifier
from .linear import LinearClassifier
from .mlp import MultilayerPerceptron
from .rnn import RecurrentImageClassifier
from .transformer import TransformerImageClassifier


def create_model(model_config: dict[str, Any]) -> nn.Module:
    model_name = model_config["name"]
    parameters = model_config["parameters"]

    if model_name == "linear":
        return LinearClassifier(**parameters)
    if model_name == "mlp":
        return MultilayerPerceptron(**parameters)
    if model_name == "cnn":
        return ConvolutionalClassifier(**parameters)
    if model_name == "rnn":
        parameters = dict(parameters)
        parameters.pop("sequence_length", None)
        return RecurrentImageClassifier(**parameters)
    if model_name == "transformer":
        return TransformerImageClassifier(**parameters)

    raise ValueError(f"Unsupported model: {model_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["count_trainable_parameters", "create_model"]
