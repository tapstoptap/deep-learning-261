from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ASSIGNMENT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ASSIGNMENT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from config import load_experiment_config  # noqa: E402
from datasets.fashion_mnist import (  # noqa: E402
    calculate_split_manifest_hash,
    create_split_manifest,
)
from models import create_model  # noqa: E402
from summarize_results import summarize_by_model  # noqa: E402
from utils.metrics import calculate_classification_metrics  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_every_model_config_loads(self) -> None:
        model_names = ["linear", "mlp", "cnn", "rnn", "transformer"]
        for model_name in model_names:
            config_path = ASSIGNMENT_ROOT / "configs" / "models" / f"{model_name}.yaml"
            config = load_experiment_config(config_path)
            self.assertEqual(config["model"]["name"], model_name)
            self.assertEqual(config["data"]["dataset"], "fashion_mnist")


class ModelTests(unittest.TestCase):
    def test_every_model_produces_ten_logits_and_backpropagates(self) -> None:
        images = torch.randn(2, 1, 28, 28)
        targets = torch.tensor([1, 4])
        loss_function = torch.nn.CrossEntropyLoss()

        for model_name in ["linear", "mlp", "cnn", "rnn", "transformer"]:
            config_path = ASSIGNMENT_ROOT / "configs" / "models" / f"{model_name}.yaml"
            config = load_experiment_config(config_path)
            model = create_model(config["model"])
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 10), msg=model_name)

            loss = loss_function(logits, targets)
            self.assertTrue(torch.isfinite(loss), msg=model_name)
            loss.backward()


class DataSplitTests(unittest.TestCase):
    def test_stratified_split_has_no_overlap(self) -> None:
        targets = [class_index for class_index in range(10) for _ in range(10)]
        split_manifest = create_split_manifest(
            targets=targets,
            validation_size=20,
            split_seed=42,
        )
        train_indices = split_manifest["train_indices"]
        validation_indices = split_manifest["validation_indices"]

        self.assertEqual(len(train_indices), 80)
        self.assertEqual(len(validation_indices), 20)
        self.assertFalse(set(train_indices).intersection(validation_indices))
        self.assertEqual(set(train_indices + validation_indices), set(range(100)))

    def test_split_hash_is_independent_of_dictionary_key_order(self) -> None:
        first_manifest = {"split_seed": 42, "train_indices": [0, 1]}
        second_manifest = {"train_indices": [0, 1], "split_seed": 42}
        self.assertEqual(
            calculate_split_manifest_hash(first_manifest),
            calculate_split_manifest_hash(second_manifest),
        )


class MetricTests(unittest.TestCase):
    def test_metrics_are_one_for_perfect_predictions(self) -> None:
        targets = list(range(10))
        metrics = calculate_classification_metrics(
            targets=targets,
            predictions=targets,
            labels=list(range(10)),
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)


class ResultSummaryTests(unittest.TestCase):
    def test_results_are_grouped_by_model(self) -> None:
        evaluations = [
            {
                "model": "linear",
                "seed": 42,
                "test_metrics": {"accuracy": 0.8, "macro_f1": 0.7},
                "trainable_parameters": 100,
                "training_time_seconds": 10.0,
                "inference_timing": {"milliseconds_per_sample": 0.2},
            },
            {
                "model": "linear",
                "seed": 123,
                "test_metrics": {"accuracy": 0.9, "macro_f1": 0.8},
                "trainable_parameters": 100,
                "training_time_seconds": 12.0,
                "inference_timing": {"milliseconds_per_sample": 0.3},
            },
        ]
        summaries = summarize_by_model(evaluations)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["runs"], 2)
        self.assertAlmostEqual(summaries[0]["accuracy_mean"], 0.85)


if __name__ == "__main__":
    unittest.main()
