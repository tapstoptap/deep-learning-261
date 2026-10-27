from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
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


def save_combined_training_curves(
    histories_by_model: dict[str, dict[int, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    """Plot epoch-wise mean and sample SD using only epochs shared by all seeds."""
    if not histories_by_model:
        raise ValueError("No training histories were provided")

    model_order = [
        model for model in ("linear", "mlp", "cnn", "rnn", "transformer")
        if model in histories_by_model
    ]
    model_order += sorted(set(histories_by_model) - set(model_order))
    colors = {
        "linear": "#526174",
        "mlp": "#B85B40",
        "cnn": "#087E83",
        "rnn": "#AE8023",
        "transformer": "#7653A6",
    }
    names = {"linear": "Linear", "mlp": "MLP", "cnn": "CNN", "rnn": "GRU", "transformer": "Transformer"}
    fallback_colors = plt.get_cmap("tab10")

    figure, (f1_axis, loss_axis) = plt.subplots(2, 1, figsize=(10.8, 8.2))
    fields_by_axis = (
        (f1_axis, "train_macro_f1", "validation_macro_f1", "Macro-F1"),
        (loss_axis, "train_loss", "validation_loss", "Cross-entropy loss"),
    )
    seed_counts: set[int] = set()
    max_shared_epoch = 0
    lowest_f1 = 1.0

    for model_index, model in enumerate(model_order):
        histories = histories_by_model[model]
        if not histories or any(not history for history in histories.values()):
            raise ValueError(f"Missing training history for {model}")
        seed_counts.add(len(histories))
        common_epochs = min(len(history) for history in histories.values())
        max_shared_epoch = max(max_shared_epoch, common_epochs)
        epochs = np.arange(1, common_epochs + 1)
        color = colors.get(model, fallback_colors(model_index % 10))

        for axis, train_field, validation_field, title in fields_by_axis:
            for field, line_style, opacity, band_opacity in (
                (validation_field, "-", 1.0, 0.14),
                (train_field, (0, (4, 2)), 0.82, 0.055),
            ):
                values = np.array(
                    [
                        [float(row[field]) for row in history[:common_epochs]]
                        for _, history in sorted(histories.items())
                    ],
                    dtype=float,
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {field} values for {model}")
                mean = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                if axis is f1_axis:
                    lowest_f1 = min(lowest_f1, float(np.min(mean - sd)))
                axis.plot(epochs, mean, color=color, linestyle=line_style, linewidth=2.1,
                          alpha=opacity)
                if len(values) > 1:
                    axis.fill_between(epochs, mean - sd, mean + sd, color=color,
                                      alpha=band_opacity, linewidth=0)

            axis.set_ylabel(title)
            axis.grid(axis="y", color="#DCE3EA", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_facecolor("#F8FAFC")

    if len(seed_counts) != 1:
        raise ValueError("Models have different numbers of training seeds")
    seed_count = seed_counts.pop()
    subtitle = (
        f"Mean ±1 sample SD across {seed_count} seeds; each model ends at its last shared epoch"
        if seed_count > 1 else "Single seed; no SD band"
    )
    for axis in (f1_axis, loss_axis):
        axis.set_xlim(1, max_shared_epoch)
    f1_axis.set_ylim(max(0, lowest_f1 - 0.04), 1.01)
    loss_axis.set_ylim(bottom=0)
    loss_axis.set_xlabel("Completed epoch")
    figure.suptitle("Training dynamics across model families", fontsize=16, weight="semibold")
    figure.text(
        0.5, 0.935,
        subtitle,
        ha="center", fontsize=9, color="#526174",
    )
    model_handles = [
        Line2D([0], [0], color=colors.get(model, fallback_colors(index % 10)),
               linewidth=2.5, label=names.get(model, model))
        for index, model in enumerate(model_order)
    ]
    style_handles = [
        Line2D([0], [0], color="#2D3642", linewidth=2, label="Validation"),
        Line2D([0], [0], color="#2D3642", linewidth=2, linestyle=(0, (4, 2)),
               label="Training"),
    ]
    figure.legend(handles=model_handles + style_handles, loc="lower center", ncol=7,
                  frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.015))
    figure.tight_layout(rect=(0.035, 0.065, 0.985, 0.93), h_pad=2.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def save_representation_ablation_curves(
    histories_by_variant: dict[str, dict[int, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    """Plot training curves for the GRU/Transformer representation ablation."""
    if not histories_by_variant:
        raise ValueError("No ablation training histories were provided")

    variant_order = [
        "rnn/image_rows",
        "rnn/image_columns",
        "rnn/image_patches",
        "transformer/image_rows",
        "transformer/image_columns",
        "transformer/image_patches",
    ]
    variant_order = [
        variant for variant in variant_order if variant in histories_by_variant
    ]
    variant_order += sorted(set(histories_by_variant) - set(variant_order))
    colors = {
        "rnn/image_rows": "#C45A4A",
        "rnn/image_columns": "#D98B38",
        "rnn/image_patches": "#E0B84C",
        "transformer/image_rows": "#3568A8",
        "transformer/image_columns": "#5D8BC5",
        "transformer/image_patches": "#89A9D6",
    }
    labels = {
        "rnn/image_rows": "GRU / rows",
        "rnn/image_columns": "GRU / columns",
        "rnn/image_patches": "GRU / 4×4 patches",
        "transformer/image_rows": "Transformer / rows",
        "transformer/image_columns": "Transformer / columns",
        "transformer/image_patches": "Transformer / 4×4 patches",
    }
    fallback_colors = plt.get_cmap("tab10")

    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    axes = (
        (loss_axis, "train_loss", "validation_loss", "Cross-entropy loss"),
        (f1_axis, "train_macro_f1", "validation_macro_f1", "Macro-F1"),
    )
    seed_counts: set[int] = set()
    max_shared_epoch = 0
    lowest_f1 = 1.0

    for variant_index, variant in enumerate(variant_order):
        histories = histories_by_variant[variant]
        if not histories or any(not history for history in histories.values()):
            raise ValueError(f"Missing training history for {variant}")
        seed_counts.add(len(histories))
        common_epochs = min(len(history) for history in histories.values())
        max_shared_epoch = max(max_shared_epoch, common_epochs)
        epochs = np.arange(1, common_epochs + 1)
        color = colors.get(variant, fallback_colors(variant_index % 10))

        for axis, train_field, validation_field, title in axes:
            for field, line_style, line_alpha, band_alpha in (
                (validation_field, "-", 1.0, 0.14),
                (train_field, (0, (4, 2)), 0.78, 0.055),
            ):
                values = np.array(
                    [
                        [float(row[field]) for row in history[:common_epochs]]
                        for _, history in sorted(histories.items())
                    ],
                    dtype=float,
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {field} values for {variant}")
                mean = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                if axis is f1_axis:
                    lowest_f1 = min(lowest_f1, float(np.min(mean - sd)))
                axis.plot(
                    epochs,
                    mean,
                    color=color,
                    linestyle=line_style,
                    linewidth=2.0,
                    alpha=line_alpha,
                )
                if len(values) > 1:
                    axis.fill_between(
                        epochs,
                        mean - sd,
                        mean + sd,
                        color=color,
                        alpha=band_alpha,
                        linewidth=0,
                    )

            axis.set_xlabel("Epoch")
            axis.set_ylabel(title)
            axis.grid(axis="y", color="#DCE3EA", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_facecolor("#F8FAFC")

    if len(seed_counts) != 1:
        raise ValueError("Ablation variants have different numbers of seeds")
    seed_count = seed_counts.pop()
    for axis in (loss_axis, f1_axis):
        axis.set_xlim(1, max_shared_epoch)
    loss_axis.set_ylim(bottom=0)
    f1_axis.set_ylim(max(0, lowest_f1 - 0.04), 1.01)

    figure.suptitle("Representation ablation on sequence models", fontsize=15, weight="semibold")
    figure.text(
        0.5,
        0.925,
        f"Mean ±1 sample SD across {seed_count} seeds; curves stop at each variant's last shared epoch",
        ha="center",
        fontsize=9,
        color="#526174",
    )
    variant_handles = [
        Line2D(
            [0],
            [0],
            color=colors.get(variant, fallback_colors(index % 10)),
            linewidth=2.4,
            label=labels.get(variant, variant),
        )
        for index, variant in enumerate(variant_order)
    ]
    style_handles = [
        Line2D([0], [0], color="#2D3642", linewidth=2, label="Validation"),
        Line2D(
            [0],
            [0],
            color="#2D3642",
            linewidth=2,
            linestyle=(0, (4, 2)),
            label="Training",
        ),
    ]
    figure.legend(
        handles=variant_handles + style_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0.035, 0.12, 0.985, 0.9), w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
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
