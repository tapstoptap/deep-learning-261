from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as tv_datasets


class Food101Dataset(Dataset):
    def __init__(self, root: str | Path, samples: list[tuple[str, int]], transform=None):
        self.root = Path(root)
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_name, label = self.samples[index]
        image_path = self.root / "images" / f"{image_name}.jpg"

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_classes(root: str | Path) -> tuple[list[str], dict[str, int]]:
    root = Path(root)
    classes_file = root / "meta" / "classes.txt"

    if not classes_file.exists():
        raise FileNotFoundError(
            f"Classes file not found at {classes_file}. "
            "Please download Food-101 or check data root path."
        )

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    return classes, class_to_idx


def load_split(root: str | Path, split: str, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    root = Path(root)
    split_file = root / "meta" / f"{split}.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found at {split_file}.")

    samples: list[tuple[str, int]] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            image_name = line.strip()
            if not image_name:
                continue
            class_name = image_name.split("/")[0]
            if class_name in class_to_idx:
                label = class_to_idx[class_name]
                samples.append((image_name, label))
    return samples


def stratified_split(
    samples: list[tuple[str, int]], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Split samples into train and validation sets while preserving the class distribution.
    """
    rng = random.Random(seed)
    samples_by_class: dict[int, list[tuple[str, int]]] = {}

    for image_name, label in samples:
        if label not in samples_by_class:
            samples_by_class[label] = []
        samples_by_class[label].append((image_name, label))

    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []

    for label, class_samples in samples_by_class.items():
        class_samples = class_samples.copy()
        rng.shuffle(class_samples)
        val_size = (
            max(1, int(len(class_samples) * val_ratio))
            if len(class_samples) > 1 and val_ratio > 0
            else int(len(class_samples) * val_ratio)
        )

        val_samples.extend(class_samples[:val_size])
        train_samples.extend(class_samples[val_size:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def get_transforms(
    image_size: int = 224,
    normalize: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and evaluation data transformations.
    Uses standard ImageNet normalization parameters.
    """
    norm_step = (
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if normalize
        else []
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *norm_step,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        *norm_step,
    ])

    return train_transform, eval_transform


def create_datasets(
    root: str | Path = "data/food-101",
    image_size: int = 224,
    val_ratio: float = 0.1,
    seed: int = 42,
    normalize: bool = True,
    max_samples_per_class: int | None = None,
) -> tuple[Food101Dataset, Food101Dataset, Food101Dataset, list[str]]:
    """
    Create Food-101 train, validation, and test datasets.
    """
    root = Path(root)
    classes, class_to_idx = load_classes(root)

    all_train_samples = load_split(root, "train", class_to_idx)
    test_samples = load_split(root, "test", class_to_idx)

    # Optional subsetting for quick development / smoke tests
    if max_samples_per_class is not None:
        def filter_subset(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
            counts: dict[int, int] = {}
            filtered = []
            for item in samples:
                lbl = item[1]
                if counts.get(lbl, 0) < max_samples_per_class:
                    counts[lbl] = counts.get(lbl, 0) + 1
                    filtered.append(item)
            return filtered

        all_train_samples = filter_subset(all_train_samples)
        test_samples = filter_subset(test_samples)

    train_samples, val_samples = stratified_split(
        all_train_samples,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_transform, eval_transform = get_transforms(
        image_size=image_size, normalize=normalize
    )

    train_dataset = Food101Dataset(root=root, samples=train_samples, transform=train_transform)
    val_dataset = Food101Dataset(root=root, samples=val_samples, transform=eval_transform)
    test_dataset = Food101Dataset(root=root, samples=test_samples, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset, classes


def create_data_loaders(
    root: str | Path = "data/food-101",
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    val_ratio: float = 0.1,
    seed: int = 42,
    normalize: bool = True,
    max_samples_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Convenience wrapper to build DataLoaders for train, val, and test splits.
    """
    train_ds, val_ds, test_ds, classes = create_datasets(
        root=root,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        normalize=normalize,
        max_samples_per_class=max_samples_per_class,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, classes


def download_food101(destination: str | Path) -> None:
    """Download and extract the official Food-101 dataset (~5GB)."""
    import shutil
    from torchvision.datasets.utils import download_and_extract_archive

    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    target_dir = dest / "food-101"

    # If target directory exists but is empty/corrupt, clean it up first
    meta_classes = target_dir / "meta" / "classes.txt"
    if target_dir.exists():
        if not meta_classes.exists() or meta_classes.stat().st_size == 0:
            print(f"Found incomplete or empty dataset directory at {target_dir}. Cleaning up...")
            shutil.rmtree(target_dir, ignore_errors=True)

    print(f"Downloading Food-101 dataset (~5GB) to {dest}...")
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    md5 = "85eeb7614ecfb77eab51a12226b4ba85"
    download_and_extract_archive(
        url=url,
        download_root=str(dest),
        extract_root=str(dest),
        md5=md5,
    )
    print(f"Download and extraction completed successfully into {target_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food-101 Dataset preparation & verification")
    parser.add_argument("--download", action="store_true", help="Download official Food-101 dataset")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    parser.add_argument("--root", type=str, default="data/food-101", help="Path to Food-101 root")
    args = parser.parse_args()

    if args.download:
        download_food101(args.out)
    else:
        try:
            train_ds, val_ds, test_ds, classes = create_datasets(root=args.root)
            print(f"Classes: {len(classes)}")
            print(f"Train samples: {len(train_ds)}")
            print(f"Validation samples: {len(val_ds)}")
            print(f"Test samples: {len(test_ds)}")
        except Exception as e:
            print(f"Could not load dataset at '{args.root}': {e}")
            print("To download the dataset, run with --download --out data/")
