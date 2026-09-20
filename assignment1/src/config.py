from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {file_path}")
    return content


def load_experiment_config(model_config_path: str | Path) -> dict[str, Any]:
    model_config_path = Path(model_config_path).resolve()
    model_config = load_yaml_file(model_config_path)

    protocol_file = model_config.get("protocol_file")
    if not protocol_file:
        raise ValueError(f"Missing 'protocol_file' in {model_config_path}")

    protocol_path = (model_config_path.parent / protocol_file).resolve()
    protocol_config = load_yaml_file(protocol_path)

    config = deepcopy(protocol_config)
    for key, value in model_config.items():
        if key != "protocol_file":
            config[key] = deepcopy(value)

    config["config_files"] = {
        "model": str(model_config_path),
        "protocol": str(protocol_path),
    }
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    required_sections = [
        "data",
        "preprocessing",
        "reproducibility",
        "dataloader",
        "evaluation",
        "checkpoint",
        "timing",
        "paths",
        "model",
        "training",
    ]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing config sections: {', '.join(missing_sections)}")

    if config["data"].get("dataset") != "fashion_mnist":
        raise ValueError("Assignment 1 currently supports only Fashion-MNIST")

    if config["training"].get("loss") != "cross_entropy":
        raise ValueError("Assignment 1 models must use cross-entropy loss")

    if config["training"].get("batch_size", 0) <= 0:
        raise ValueError("training.batch_size must be greater than zero")
    if config["training"].get("max_epochs", 0) <= 0:
        raise ValueError("training.max_epochs must be greater than zero")
    if config["dataloader"].get("num_workers", -1) < 0:
        raise ValueError("dataloader.num_workers cannot be negative")


def apply_training_overrides(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    batch_size: int | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    num_workers: int | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    resolved_config = deepcopy(config)

    if seed is not None:
        resolved_config["reproducibility"]["development_seed"] = seed
    if batch_size is not None:
        resolved_config["training"]["batch_size"] = batch_size
    if max_epochs is not None:
        resolved_config["training"]["max_epochs"] = max_epochs
    if learning_rate is not None:
        resolved_config["training"]["optimizer"]["learning_rate"] = learning_rate
    if weight_decay is not None:
        resolved_config["training"]["optimizer"]["weight_decay"] = weight_decay
    if num_workers is not None:
        resolved_config["dataloader"]["num_workers"] = num_workers
    if output_dir is not None:
        resolved_config["paths"]["output_root"] = output_dir

    validate_experiment_config(resolved_config)
    return resolved_config


def save_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
