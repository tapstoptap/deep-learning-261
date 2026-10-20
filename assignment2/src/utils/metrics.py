from __future__ import annotations

from typing import Sequence
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def calculate_accuracy(targets: Sequence[int], predictions: Sequence[int]) -> float:
    return float(accuracy_score(targets, predictions))


def calculate_macro_f1(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[int] | None = None,
    zero_division: int = 0,
) -> float:
    return float(
        f1_score(
            targets,
            predictions,
            labels=list(labels) if labels is not None else None,
            average="macro",
            zero_division=zero_division,
        )
    )


def calculate_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    return confusion_matrix(
        targets,
        predictions,
        labels=list(labels) if labels is not None else None,
    )


def calculate_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[int] | None = None,
    zero_division: int = 0,
) -> dict[str, float]:
    return {
        "accuracy": calculate_accuracy(targets, predictions),
        "macro_f1": calculate_macro_f1(
            targets, predictions, labels=labels, zero_division=zero_division
        ),
    }


def calculate_per_class_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    target_names: Sequence[str] | None = None,
) -> dict:
    """Return dictionary report with per-class precision, recall, f1-score, and support."""
    return classification_report(
        targets,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def get_top_confused_pairs(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    top_k: int = 10,
) -> list[dict]:
    """Find the top-K pairs of (true_class, predicted_class) that are most frequently confused."""
    confusions: dict[tuple[int, int], int] = {}
    for t, p in zip(targets, predictions):
        if t != p:
            pair = (int(t), int(p))
            confusions[pair] = confusions.get(pair, 0) + 1

    sorted_pairs = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "true_class": class_names[true_idx] if true_idx < len(class_names) else str(true_idx),
            "pred_class": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
            "count": count,
        }
        for (true_idx, pred_idx), count in sorted_pairs
    ]
