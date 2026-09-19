from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def calculate_accuracy(targets: Sequence[int], predictions: Sequence[int]) -> float:
    return float(accuracy_score(targets, predictions))


def calculate_macro_f1(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[int],
    zero_division: int = 0,
) -> float:
    return float(
        f1_score(
            targets,
            predictions,
            labels=list(labels),
            average="macro",
            zero_division=zero_division,
        )
    )


def calculate_confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int], labels: Sequence[int]
) -> np.ndarray:
    return confusion_matrix(targets, predictions, labels=list(labels))


def calculate_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[int],
    zero_division: int = 0,
) -> dict[str, float]:
    return {
        "accuracy": calculate_accuracy(targets, predictions),
        "macro_f1": calculate_macro_f1(
            targets, predictions, labels, zero_division=zero_division
        ),
    }
