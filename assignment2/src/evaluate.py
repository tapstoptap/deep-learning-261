from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

# Add src to sys.path if running directly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from datasets.food101 import (
    Food101Dataset,
    get_transforms,
    load_classes,
    load_split,
    stratified_split,
)
from models import count_parameters, create_model
from utils.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    get_top_confused_pairs,
)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # For 101 classes, a full text annotation heatmap is dense, so plot matrix without text annotations
        plt.figure(figsize=(18, 16))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=False,
            yticklabels=False,
            cbar=True,
        )
        plt.title(f"Food-101 Confusion Matrix ({len(class_names)} classes)", fontsize=16)
        plt.xlabel("Predicted Class Index", fontsize=14)
        plt.ylabel("True Class Index", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Warning: could not generate confusion matrix plot ({e})")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    assignment_root = CURRENT_DIR.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = assignment_root / config_path

    config = load_yaml_config(config_path)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Data
    data_cfg = config.get("data", {})
    root_data = args.data_root if args.data_root else data_cfg.get("root", "data/food-101")
    root_data_path = Path(root_data)
    if not root_data_path.is_absolute():
        root_data_path = assignment_root / root_data_path

    classes, class_to_idx = load_classes(root_data_path)
    image_size = data_cfg.get("image_size", 224)
    _, eval_transform = get_transforms(image_size=image_size, normalize=True)

    if args.split == "test":
        samples = load_split(root_data_path, "test", class_to_idx)
    elif args.split == "val":
        all_train = load_split(root_data_path, "train", class_to_idx)
        val_ratio = data_cfg.get("val_ratio", 0.1)
        seed = config.get("seed", 42)
        _, samples = stratified_split(all_train, val_ratio=val_ratio, seed=seed)
    else:
        samples = load_split(root_data_path, args.split, class_to_idx)

    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 32)
    dataset = Food101Dataset(root=root_data_path, samples=samples, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model_cfg = config.get("model", {})
    model_cfg["num_classes"] = len(classes)
    model = create_model(model_cfg).to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = assignment_root / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading weights from checkpoint: {ckpt_path}")
    checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    model.eval()

    # Inference loop
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    print(f"Running evaluation on {args.split} set ({len(dataset)} samples)...")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            preds = logits.argmax(dim=1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)
    metrics = calculate_classification_metrics(all_targets, all_preds)
    per_class_report = calculate_per_class_metrics(all_targets, all_preds, target_names=classes)
    cm = calculate_confusion_matrix(all_targets, all_preds, labels=list(range(len(classes))))
    top_confusions = get_top_confused_pairs(
        all_targets, all_preds, class_names=classes, top_k=args.top_k_confusions
    )

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=" * 50)
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:  {metrics['macro_f1'] * 100:.2f}%")
    print("=" * 50)
    print(f"\nTop {len(top_confusions)} Confused Class Pairs:")
    for pair in top_confusions:
        print(f"  True: {pair['true_class']:<20} -> Pred: {pair['pred_class']:<20} (Count: {pair['count']})")

    # Save results
    out_dir = Path(args.output_dir or config.get("output", {}).get("result_dir", "results"))
    if not out_dir.is_absolute():
        out_dir = assignment_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "split": args.split,
        "samples": len(dataset),
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "checkpoint": str(ckpt_path),
        "top_confusions": top_confusions,
        "per_class": per_class_report,
    }

    metrics_file = out_dir / f"{args.split}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    cm_csv = out_dir / f"{args.split}_confusion_matrix.csv"
    np.savetxt(cm_csv, cm, delimiter=",", fmt="%d")

    plot_confusion_matrix(cm, classes, out_dir / f"{args.split}_confusion_matrix.png")

    print(f"\nEvaluation artifacts saved to {out_dir}")
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2 Model Evaluation (Food-101)")
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint .pt")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory of Food-101 data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--top-k-confusions", type=int, default=10, help="Number of top confused pairs to report")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
