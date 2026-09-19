from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass
class FashionMnistDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    split_manifest: dict[str, Any]


def calculate_split_manifest_hash(split_manifest: dict[str, Any]) -> str:
    serialized_manifest = json.dumps(
        split_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized_manifest).hexdigest()


def create_image_transform(
    normalization_enabled: bool,
    normalization_mean: float | None,
    normalization_std: float | None,
    augmentation_enabled: bool,
) -> transforms.Compose:
    transform_steps: list[Any] = []
    if augmentation_enabled:
        transform_steps.extend(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transform_steps.append(transforms.ToTensor())

    if normalization_enabled:
        if normalization_mean is None or normalization_std is None:
            raise ValueError(
                "Normalization is enabled, but mean or std is missing from protocol.yaml"
            )
        transform_steps.append(
            transforms.Normalize((normalization_mean,), (normalization_std,))
        )
    return transforms.Compose(transform_steps)


def create_split_manifest(
    targets: list[int],
    validation_size: int,
    split_seed: int,
) -> dict[str, Any]:
    all_indices = list(range(len(targets)))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_size,
        random_state=split_seed,
        shuffle=True,
        stratify=targets,
    )
    return {
        "dataset": "FashionMNIST",
        "official_train_size": len(targets),
        "split_seed": split_seed,
        "stratified": True,
        "train_indices": sorted(train_indices),
        "validation_indices": sorted(validation_indices),
    }


def validate_split_manifest(
    split_manifest: dict[str, Any],
    dataset_size: int,
    expected_train_size: int,
    expected_validation_size: int,
    expected_split_seed: int,
) -> None:
    train_indices = split_manifest.get("train_indices", [])
    validation_indices = split_manifest.get("validation_indices", [])

    if split_manifest.get("split_seed") != expected_split_seed:
        raise ValueError("Saved split seed does not match protocol.yaml")
    if len(train_indices) != expected_train_size:
        raise ValueError("Saved training split size does not match protocol.yaml")
    if len(validation_indices) != expected_validation_size:
        raise ValueError("Saved validation split size does not match protocol.yaml")
    if set(train_indices).intersection(validation_indices):
        raise ValueError("Training and validation splits overlap")

    all_indices = train_indices + validation_indices
    if len(set(all_indices)) != dataset_size:
        raise ValueError("Saved split does not contain every official training sample")
    if min(all_indices) < 0 or max(all_indices) >= dataset_size:
        raise ValueError("Saved split contains an out-of-range sample index")


def load_or_create_split_manifest(
    split_file: Path,
    targets: list[int],
    train_size: int,
    validation_size: int,
    split_seed: int,
) -> dict[str, Any]:
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as file:
            split_manifest = json.load(file)
    else:
        split_manifest = create_split_manifest(targets, validation_size, split_seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with split_file.open("w", encoding="utf-8") as file:
            json.dump(split_manifest, file, indent=2)

    validate_split_manifest(
        split_manifest,
        dataset_size=len(targets),
        expected_train_size=train_size,
        expected_validation_size=validation_size,
        expected_split_seed=split_seed,
    )
    return split_manifest


def calculate_normalization_statistics(
    dataset: datasets.FashionMNIST, train_indices: list[int]
) -> tuple[float, float]:
    train_images = dataset.data[train_indices].to(torch.float32).div(255.0)
    return float(train_images.mean()), float(train_images.std(unbiased=False))


def create_data_loaders(
    config: dict[str, Any],
    assignment_root: Path,
    seed: int,
    download: bool = False,
) -> FashionMnistDataLoaders:
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    normalization_config = preprocessing_config["normalization"]
    data_loader_config = config["dataloader"]

    data_root = assignment_root / data_config["root"]
    split_file = assignment_root / data_config["split_file"]

    train_transform = create_image_transform(
        normalization_enabled=normalization_config["enabled"],
        normalization_mean=normalization_config["mean"],
        normalization_std=normalization_config["std"],
        augmentation_enabled=preprocessing_config["augmentation"]["enabled"],
    )
    evaluation_transform = create_image_transform(
        normalization_enabled=normalization_config["enabled"],
        normalization_mean=normalization_config["mean"],
        normalization_std=normalization_config["std"],
        augmentation_enabled=False,
    )

    training_dataset = datasets.FashionMNIST(
        root=data_root, train=True, download=download, transform=train_transform
    )
    validation_dataset = datasets.FashionMNIST(
        root=data_root, train=True, download=download, transform=evaluation_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_root, train=False, download=download, transform=evaluation_transform
    )

    targets = [int(target) for target in training_dataset.targets]
    split_manifest = load_or_create_split_manifest(
        split_file=split_file,
        targets=targets,
        train_size=data_config["train_size"],
        validation_size=data_config["validation_size"],
        split_seed=data_config["split_seed"],
    )

    train_subset = Subset(training_dataset, split_manifest["train_indices"])
    validation_subset = Subset(
        validation_dataset, split_manifest["validation_indices"]
    )

    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": data_loader_config["num_workers"],
        "pin_memory": data_loader_config["pin_memory"],
    }

    from utils.seed import seed_data_loader_worker

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_data_loader_worker,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        worker_init_fn=seed_data_loader_worker,
        **loader_arguments,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        worker_init_fn=seed_data_loader_worker,
        **loader_arguments,
    )
    return FashionMnistDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        split_manifest=split_manifest,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Fashion-MNIST and create the Assignment 1 data split."
    )
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=6000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--calculate-normalization", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dataset = datasets.FashionMNIST(
        root=arguments.out,
        train=True,
        download=arguments.download,
        transform=transforms.ToTensor(),
    )
    split_file = arguments.split_file or Path(
        f"configs/splits/fashion_mnist_seed{arguments.split_seed}.json"
    )
    targets = [int(target) for target in dataset.targets]
    split_manifest = load_or_create_split_manifest(
        split_file=split_file,
        targets=targets,
        train_size=len(dataset) - arguments.validation_size,
        validation_size=arguments.validation_size,
        split_seed=arguments.split_seed,
    )

    print(f"Training samples: {len(split_manifest['train_indices'])}")
    print(f"Validation samples: {len(split_manifest['validation_indices'])}")
    print(f"Split manifest: {split_file.resolve()}")
    if arguments.calculate_normalization:
        mean, std = calculate_normalization_statistics(
            dataset, split_manifest["train_indices"]
        )
        print(f"Training subset mean: {mean:.8f}")
        print(f"Training subset std: {std:.8f}")


if __name__ == "__main__":
    main()
