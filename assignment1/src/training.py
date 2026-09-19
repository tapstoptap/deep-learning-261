from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.metrics import calculate_classification_metrics


def create_optimizer(model: nn.Module, optimizer_config: dict[str, Any]) -> Optimizer:
    optimizer_name = optimizer_config["name"].lower()
    learning_rate = optimizer_config["learning_rate"]
    weight_decay = optimizer_config["weight_decay"]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    labels: list[int],
    zero_division: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_index, (images, targets) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError("The training data loader produced no samples")

    metrics = calculate_classification_metrics(
        all_targets, all_predictions, labels, zero_division
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    labels: list[int],
    zero_division: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())

    if total_samples == 0:
        raise ValueError("The evaluation data loader produced no samples")

    metrics = calculate_classification_metrics(
        all_targets, all_predictions, labels, zero_division
    )
    metrics["loss"] = total_loss / total_samples
    return metrics, all_targets, all_predictions


def validation_result_is_better(
    validation_metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
    min_delta: float,
) -> bool:
    if best_metrics is None:
        return True

    current_f1 = validation_metrics["macro_f1"]
    best_f1 = best_metrics["macro_f1"]
    if current_f1 > best_f1 + min_delta:
        return True
    if abs(current_f1 - best_f1) <= min_delta:
        return validation_metrics["loss"] < best_metrics["loss"]
    return False


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
            "config": config,
        },
        checkpoint_path,
    )


def load_model_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
