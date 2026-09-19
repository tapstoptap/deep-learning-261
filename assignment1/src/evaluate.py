from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import apply_training_overrides, load_experiment_config, save_config
from datasets.fashion_mnist import (
    FASHION_MNIST_CLASS_NAMES,
    calculate_split_manifest_hash,
    create_data_loaders,
)
from models import count_trainable_parameters, create_model
from training import evaluate_one_epoch, load_model_checkpoint
from utils.metrics import calculate_confusion_matrix
from utils.plots import (
    collect_prediction_examples,
    save_confusion_matrix,
    save_prediction_examples,
)
from utils.runtime import choose_device, get_device_description, get_git_commit
from utils.seed import set_random_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Assignment 1 checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_time(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    measurement_repeats: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(data_loader):
            if batch_index >= warmup_batches:
                break
            model(images.to(device, non_blocking=True))
        synchronize_device(device)

        repeat_seconds: list[float] = []
        samples_per_repeat = 0
        for _ in range(measurement_repeats):
            total_forward_seconds = 0.0
            measured_samples = 0
            for batch_index, (images, _) in enumerate(data_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                synchronize_device(device)
                started_at = time.perf_counter()
                model(images)
                synchronize_device(device)
                total_forward_seconds += time.perf_counter() - started_at
                measured_samples += images.size(0)
            repeat_seconds.append(total_forward_seconds)
            samples_per_repeat = measured_samples

    mean_seconds = statistics.mean(repeat_seconds)
    standard_deviation_seconds = (
        statistics.stdev(repeat_seconds) if len(repeat_seconds) > 1 else 0.0
    )
    return {
        "scope": "forward_only",
        "measurement_repeats": measurement_repeats,
        "samples_per_repeat": samples_per_repeat,
        "mean_seconds": mean_seconds,
        "standard_deviation_seconds": standard_deviation_seconds,
        "milliseconds_per_sample": 1000.0 * mean_seconds / samples_per_repeat,
    }


def save_predictions(
    targets: list[int], predictions: list[int], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sample_index", "target", "target_name", "prediction", "prediction_name"]
        )
        for sample_index, (target, prediction) in enumerate(zip(targets, predictions)):
            writer.writerow(
                [
                    sample_index,
                    target,
                    FASHION_MNIST_CLASS_NAMES[target],
                    prediction,
                    FASHION_MNIST_CLASS_NAMES[prediction],
                ]
            )


def main() -> None:
    arguments = parse_arguments()
    assignment_root = Path(__file__).resolve().parent.parent
    repository_root = assignment_root.parent

    config = load_experiment_config(arguments.config)
    config = apply_training_overrides(
        config,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        output_dir=arguments.output_dir,
    )
    seed = config["reproducibility"]["development_seed"]
    set_random_seed(seed, config["reproducibility"]["deterministic"])
    device = choose_device(arguments.device)

    data_loaders = create_data_loaders(
        config=config,
        assignment_root=assignment_root,
        seed=seed,
        download=arguments.download,
    )
    model = create_model(config["model"]).to(device)
    checkpoint_path = arguments.checkpoint.resolve()
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config:
        if checkpoint_config.get("model") != config.get("model"):
            raise ValueError("The checkpoint model config does not match --config")
        if checkpoint_config.get("preprocessing") != config.get("preprocessing"):
            raise ValueError("The checkpoint preprocessing does not match --config")
        saved_split_hash = checkpoint_config.get("run", {}).get(
            "split_manifest_sha256"
        )
        current_split_hash = calculate_split_manifest_hash(
            data_loaders.split_manifest
        )
        if saved_split_hash and saved_split_hash != current_split_hash:
            raise ValueError("The current data split does not match the checkpoint")

    max_batches = 2 if arguments.smoke_test else None
    loss_function = nn.CrossEntropyLoss()
    labels = config["evaluation"]["labels"]
    test_metrics, targets, predictions = evaluate_one_epoch(
        model=model,
        data_loader=data_loaders.test,
        loss_function=loss_function,
        device=device,
        labels=labels,
        zero_division=config["evaluation"]["zero_division"],
        max_batches=max_batches,
    )

    timing_config = config["timing"]
    inference_timing = measure_inference_time(
        model=model,
        data_loader=data_loaders.test,
        device=device,
        warmup_batches=min(timing_config["warmup_batches"], 1)
        if arguments.smoke_test
        else timing_config["warmup_batches"],
        measurement_repeats=1
        if arguments.smoke_test
        else timing_config["measurement_repeats"],
        max_batches=max_batches,
    )

    model_name = config["model"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = assignment_root / output_root
    output_directory = output_root / model_name / f"evaluation_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)

    config["evaluation_run"] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "device_description": get_device_description(device),
        "git_commit": get_git_commit(repository_root),
        "smoke_test": arguments.smoke_test,
    }
    save_config(config, output_directory / "resolved_config.yaml")

    confusion_values = calculate_confusion_matrix(targets, predictions, labels)
    save_confusion_matrix(
        confusion_values,
        output_directory / "confusion_matrix.png",
        output_directory / "confusion_matrix.csv",
    )
    save_predictions(targets, predictions, output_directory / "predictions.csv")

    examples = collect_prediction_examples(model, data_loaders.test, device)
    normalization_config = config["preprocessing"]["normalization"]
    normalization_mean = (
        normalization_config["mean"] if normalization_config["enabled"] else None
    )
    normalization_std = (
        normalization_config["std"] if normalization_config["enabled"] else None
    )
    save_prediction_examples(
        examples["correct"],
        output_directory / "correct_predictions.png",
        "Correct predictions",
        normalization_mean,
        normalization_std,
    )
    save_prediction_examples(
        examples["incorrect"],
        output_directory / "incorrect_predictions.png",
        "Incorrect predictions",
        normalization_mean,
        normalization_std,
    )

    results = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
        "test_metrics": test_metrics,
        "trainable_parameters": count_trainable_parameters(model),
        "inference_timing": inference_timing,
        "device": str(device),
        "device_description": get_device_description(device),
        "seed": seed,
        "git_commit": config["evaluation_run"]["git_commit"],
        "smoke_test": arguments.smoke_test,
    }
    with (output_directory / "test_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Evaluation outputs: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
