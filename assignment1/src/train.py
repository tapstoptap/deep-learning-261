from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from config import apply_training_overrides, load_experiment_config, save_config
from datasets.fashion_mnist import (
    calculate_split_manifest_hash,
    create_data_loaders,
)
from models import count_trainable_parameters, create_model
from training import (
    create_optimizer,
    evaluate_one_epoch,
    save_checkpoint,
    train_one_epoch,
    validation_result_is_better,
)
from utils.plots import save_training_curves, save_training_history
from utils.runtime import choose_device, get_device_description, get_git_commit
from utils.seed import set_random_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Assignment 1 model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def create_run_name(model_name: str, seed: int, requested_name: str | None) -> str:
    if requested_name:
        if Path(requested_name).name != requested_name or requested_name in {".", ".."}:
            raise ValueError("--run-name must be a single directory name")
        return requested_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_seed{seed}_{timestamp}"


def main() -> None:
    arguments = parse_arguments()
    assignment_root = Path(__file__).resolve().parent.parent
    repository_root = assignment_root.parent

    config = load_experiment_config(arguments.config)
    config = apply_training_overrides(
        config,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        max_epochs=arguments.max_epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        output_dir=arguments.output_dir,
    )
    if arguments.smoke_test:
        config["training"]["max_epochs"] = 1

    seed = config["reproducibility"]["development_seed"]
    set_random_seed(seed, config["reproducibility"]["deterministic"])
    device = choose_device(arguments.device)

    model_name = config["model"]["name"]
    run_name = create_run_name(model_name, seed, arguments.run_name)
    output_root = Path(config["paths"]["output_root"])
    checkpoint_root = Path(config["paths"]["checkpoint_root"])
    if not output_root.is_absolute():
        output_root = assignment_root / output_root
    if not checkpoint_root.is_absolute():
        checkpoint_root = assignment_root / checkpoint_root

    run_directory = output_root / model_name / run_name
    checkpoint_directory = checkpoint_root / model_name / run_name
    if run_directory.exists() or checkpoint_directory.exists():
        raise FileExistsError(f"Run already exists: {run_name}")
    run_directory.mkdir(parents=True)
    checkpoint_directory.mkdir(parents=True)

    config["run"] = {
        "name": run_name,
        "seed": seed,
        "device": str(device),
        "device_description": get_device_description(device),
        "git_commit": get_git_commit(repository_root),
        "smoke_test": arguments.smoke_test,
    }
    data_loaders = create_data_loaders(
        config=config,
        assignment_root=assignment_root,
        seed=seed,
        download=arguments.download,
    )
    config["run"]["split_manifest_sha256"] = calculate_split_manifest_hash(
        data_loaders.split_manifest
    )
    save_config(config, run_directory / "resolved_config.yaml")
    model = create_model(config["model"]).to(device)
    optimizer = create_optimizer(model, config["training"]["optimizer"])
    loss_function = nn.CrossEntropyLoss()

    labels = config["evaluation"]["labels"]
    zero_division = config["evaluation"]["zero_division"]
    early_stopping_config = config["training"]["early_stopping"]
    max_batches = 2 if arguments.smoke_test else None

    best_validation_metrics: dict[str, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    training_started_at = time.perf_counter()

    print(f"Model: {model_name}")
    print(f"Device: {get_device_description(device)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Run directory: {run_directory}")

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=data_loaders.train,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            labels=labels,
            zero_division=zero_division,
            max_batches=max_batches,
        )
        validation_metrics, _, _ = evaluate_one_epoch(
            model=model,
            data_loader=data_loaders.validation,
            loss_function=loss_function,
            device=device,
            labels=labels,
            zero_division=zero_division,
            max_batches=max_batches,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(history_row)

        if config["checkpoint"]["save_last"]:
            save_checkpoint(
                checkpoint_directory / "last.pt",
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
            )

        result_improved = validation_result_is_better(
            validation_metrics,
            best_validation_metrics,
            early_stopping_config["min_delta"],
        )
        if result_improved:
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            if config["checkpoint"]["save_best"]:
                save_checkpoint(
                    checkpoint_directory / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    validation_metrics,
                    config,
                )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"train F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f}, "
            f"val F1 {validation_metrics['macro_f1']:.4f}"
        )

        if (
            early_stopping_config["enabled"]
            and epochs_without_improvement >= early_stopping_config["patience"]
        ):
            print(f"Early stopping after epoch {epoch}")
            break

    training_time_seconds = time.perf_counter() - training_started_at
    save_training_history(history, run_directory / "history.csv")
    save_training_curves(history, run_directory / "training_curves.png")

    summary = {
        "run_name": run_name,
        "model": model_name,
        "seed": seed,
        "device": str(device),
        "device_description": get_device_description(device),
        "trainable_parameters": count_trainable_parameters(model),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "training_time_seconds": training_time_seconds,
        "best_checkpoint": str((checkpoint_directory / "best.pt").resolve()),
        "last_checkpoint": str((checkpoint_directory / "last.pt").resolve()),
        "git_commit": config["run"]["git_commit"],
    }
    with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
