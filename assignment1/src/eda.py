from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms

from config import load_experiment_config
from datasets.fashion_mnist import (
    FASHION_MNIST_CLASS_NAMES,
    calculate_normalization_statistics,
    load_or_create_split_manifest,
)


def save_class_distribution(class_counts: dict[int, int], output_path: Path) -> None:
    labels = [FASHION_MNIST_CLASS_NAMES[class_index] for class_index in class_counts]
    counts = [class_counts[class_index] for class_index in class_counts]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=labels, y=counts, color="steelblue")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Number of samples")
    plt.title("Fashion-MNIST training split class distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_representative_samples(
    dataset: datasets.FashionMNIST,
    train_indices: list[int],
    output_path: Path,
) -> None:
    first_index_by_class: dict[int, int] = {}
    for dataset_index in train_indices:
        class_index = int(dataset.targets[dataset_index])
        if class_index not in first_index_by_class:
            first_index_by_class[class_index] = dataset_index
        if len(first_index_by_class) == len(FASHION_MNIST_CLASS_NAMES):
            break

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    for class_index, axis in enumerate(axes.flat):
        image, _ = dataset[first_index_by_class[class_index]]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(FASHION_MNIST_CLASS_NAMES[class_index])
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Fashion-MNIST EDA outputs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    assignment_root = Path(__file__).resolve().parent.parent
    config = load_experiment_config(arguments.config)
    data_config = config["data"]

    dataset = datasets.FashionMNIST(
        root=assignment_root / data_config["root"],
        train=True,
        download=arguments.download,
        transform=transforms.ToTensor(),
    )
    targets = [int(target) for target in dataset.targets]
    split_manifest = load_or_create_split_manifest(
        split_file=assignment_root / data_config["split_file"],
        targets=targets,
        train_size=data_config["train_size"],
        validation_size=data_config["validation_size"],
        split_seed=data_config["split_seed"],
    )

    train_targets = [targets[index] for index in split_manifest["train_indices"]]
    validation_targets = [
        targets[index] for index in split_manifest["validation_indices"]
    ]
    train_class_counts = dict(sorted(Counter(train_targets).items()))
    validation_class_counts = dict(sorted(Counter(validation_targets).items()))
    mean, std = calculate_normalization_statistics(
        dataset, split_manifest["train_indices"]
    )

    output_dir = assignment_root / arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_class_distribution(train_class_counts, output_dir / "class_distribution.png")
    save_representative_samples(
        dataset,
        split_manifest["train_indices"],
        output_dir / "representative_samples.png",
    )

    summary = {
        "dataset": "FashionMNIST",
        "image_shape": [1, 28, 28],
        "number_of_classes": 10,
        "official_training_samples": len(dataset),
        "training_samples": len(train_targets),
        "validation_samples": len(validation_targets),
        "test_samples": data_config["test_size"],
        "training_class_counts": train_class_counts,
        "validation_class_counts": validation_class_counts,
        "training_subset_mean": mean,
        "training_subset_std": std,
        "split_seed": data_config["split_seed"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"EDA outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
