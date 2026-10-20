from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import yaml

# Add src to sys.path if running directly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from datasets.food101 import create_data_loaders
from models import count_parameters, create_model
from utils.metrics import calculate_classification_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_optimizer(model: nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = training_cfg.get("learning_rate", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    opt_name = training_cfg.get("optimizer", "adamw").lower()

    # Filter trainable parameters only
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = training_cfg.get("momentum", 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer, training_cfg: dict[str, Any], total_epochs: int
):
    sched_name = training_cfg.get("scheduler", "cosine").lower()
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif sched_name == "step":
        step_size = training_cfg.get("step_size", max(1, total_epochs // 3))
        gamma = training_cfg.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool = True,
    smoke_test_batches: int | None = None,
) -> tuple[float, float, float]:
    """
    Run one epoch of training or evaluation.
    Returns: (average_loss, accuracy, macro_f1)
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, targets) in enumerate(dataloader):
            if smoke_test_batches and batch_idx >= smoke_test_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train and optimizer is not None:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    if total_samples == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / max(1, total_samples)
    metrics = calculate_classification_metrics(all_targets, all_preds)
    return avg_loss, metrics["accuracy"], metrics["macro_f1"]


def save_history_plots(history: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]
        val_f1 = [h["val_macro_f1"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
        ax1.plot(epochs, val_loss, label="Val Loss", marker="s")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(epochs, train_acc, label="Train Accuracy", marker="o")
        ax2.plot(epochs, val_acc, label="Val Accuracy", marker="s")
        ax2.plot(epochs, val_f1, label="Val Macro-F1", marker="^", linestyle="--")
        ax2.set_title("Training and Validation Performance")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: could not generate plots ({e})")


def train(args: argparse.Namespace) -> None:
    assignment_root = CURRENT_DIR.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = assignment_root / config_path

    config = load_yaml_config(config_path)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")

    # Data loaders
    data_cfg = config.get("data", {})
    root_data = args.data_root if args.data_root else data_cfg.get("root", "data/food-101")
    root_data_path = Path(root_data)
    if not root_data_path.is_absolute():
        root_data_path = assignment_root / root_data_path

    batch_size = args.batch_size if args.batch_size else config.get("training", {}).get("batch_size", 32)
    image_size = data_cfg.get("image_size", 224)
    val_ratio = data_cfg.get("val_ratio", 0.1)
    num_workers = args.num_workers if args.num_workers is not None else 2

    max_per_class = args.subset_per_class
    if args.smoke_test:
        max_per_class = max_per_class or 5

    print(f"Loading Food-101 from: {root_data_path}")
    train_loader, val_loader, _, classes = create_data_loaders(
        root=root_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        max_samples_per_class=max_per_class,
    )
    print(
        f"Dataset loaded: {len(classes)} classes | "
        f"Train: {len(train_loader.dataset)} samples | "
        f"Val: {len(val_loader.dataset)} samples"
    )

    # Model creation
    model_cfg = config.get("model", {})
    model_cfg["num_classes"] = len(classes)
    model = create_model(model_cfg).to(device)
    param_counts = count_parameters(model)
    print(
        f"Model: {model_cfg.get('name')} | "
        f"Total params: {param_counts['total']:,} | "
        f"Trainable: {param_counts['trainable']:,}"
    )

    # Output paths
    out_cfg = config.get("output", {})
    ckpt_dir = Path(args.checkpoint_dir or out_cfg.get("checkpoint_dir", "checkpoints"))
    res_dir = Path(args.result_dir or out_cfg.get("result_dir", "results"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = assignment_root / ckpt_dir
    if not res_dir.is_absolute():
        res_dir = assignment_root / res_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Training setup
    training_cfg = config.get("training", {})
    if args.lr:
        training_cfg["learning_rate"] = args.lr
    epochs = args.epochs if args.epochs else training_cfg.get("epochs", 20)
    if args.smoke_test:
        epochs = 1

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, training_cfg)
    scheduler = create_scheduler(optimizer, training_cfg, epochs)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    model_tag = model_cfg.get("name", "model")
    best_ckpt_path = ckpt_dir / f"{model_tag}_best.pt"
    latest_ckpt_path = ckpt_dir / f"{model_tag}_latest.pt"

    print(f"\n--- Starting training for {epochs} epochs ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        smoke_batches = 3 if args.smoke_test else None

        train_loss, train_acc, train_f1 = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
            smoke_test_batches=smoke_batches,
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            is_train=False,
            smoke_test_batches=smoke_batches,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}% | "
            f"Time: {epoch_duration:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_duration,
        }
        history.append(epoch_record)

        # Checkpoint saving
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "metrics": {"val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1},
            },
            latest_ckpt_path,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "metrics": {"val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1},
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path.name} (Val Macro-F1: {val_f1*100:.2f}%)")

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes.")

    # Save history artifacts
    history_json = res_dir / "history.json"
    history_csv = res_dir / "history.csv"
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_history_plots(history, res_dir / "training_curves.png")

    summary = {
        "model": model_tag,
        "seed": seed,
        "device": str(device),
        "total_epochs": epochs,
        "best_val_macro_f1": best_val_f1,
        "total_training_time_sec": total_training_time,
        "parameters": param_counts,
        "best_checkpoint": str(best_ckpt_path),
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Artifacts saved to {res_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2 Model Training (Food-101)")
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML file")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory of Food-101 data")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output dir")
    parser.add_argument("--result-dir", type=str, default=None, help="Result output dir")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test for 1 epoch")
    parser.add_argument(
        "--subset-per-class",
        type=int,
        default=None,
        help="Use a small number of samples per class for quick CPU verification",
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
