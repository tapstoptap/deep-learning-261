from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.fashion_mnist import FASHION_MNIST_CLASS_NAMES


def save_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], label="Validation"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(
        epochs, [row["train_macro_f1"] for row in history], label="Train"
    )
    axes[1].plot(
        epochs,
        [row["validation_macro_f1"] for row in history],
        label="Validation",
    )
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_confusion_matrix(
    confusion_values: np.ndarray, output_path: Path, csv_path: Path
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *FASHION_MNIST_CLASS_NAMES])
        for class_name, row in zip(FASHION_MNIST_CLASS_NAMES, confusion_values):
            writer.writerow([class_name, *row.tolist()])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=FASHION_MNIST_CLASS_NAMES,
        yticklabels=FASHION_MNIST_CLASS_NAMES,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def collect_prediction_examples(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    examples_per_group: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    correct_examples: list[dict[str, Any]] = []
    incorrect_examples: list[dict[str, Any]] = []
    model.eval()

    with torch.inference_mode():
        for images, targets in data_loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)

            for image, target, prediction, class_probabilities in zip(
                images, targets, predictions, probabilities
            ):
                example = {
                    "image": image.cpu(),
                    "target": int(target),
                    "prediction": int(prediction),
                    "confidence": float(class_probabilities[prediction]),
                }
                if prediction == target and len(correct_examples) < examples_per_group:
                    correct_examples.append(example)
                elif prediction != target and len(incorrect_examples) < examples_per_group:
                    incorrect_examples.append(example)

            if (
                len(correct_examples) >= examples_per_group
                and len(incorrect_examples) >= examples_per_group
            ):
                break

    return {"correct": correct_examples, "incorrect": incorrect_examples}


def save_prediction_examples(
    examples: list[dict[str, Any]],
    output_path: Path,
    title: str,
    normalization_mean: float | None = None,
    normalization_std: float | None = None,
) -> None:
    if not examples:
        return

    columns = 4
    rows = (len(examples) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_array = np.asarray(axes).reshape(-1)

    for axis, example in zip(axes_array, examples):
        image = example["image"].squeeze(0).numpy()
        if normalization_mean is not None and normalization_std is not None:
            image = image * normalization_std + normalization_mean
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"True: {FASHION_MNIST_CLASS_NAMES[example['target']]}\n"
            f"Pred: {FASHION_MNIST_CLASS_NAMES[example['prediction']]} "
            f"({example['confidence']:.2f})"
        )
        axis.axis("off")
    for axis in axes_array[len(examples) :]:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
