from __future__ import annotations

import sys
import tempfile
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
from evaluate import (  # noqa: E402
    get_checkpoint_seed,
    validate_checkpoint_config,
    validate_test_evaluation_request,
)
from models import create_model  # noqa: E402
from summarize_results import summarize_by_model  # noqa: E402
from training import (  # noqa: E402
    load_model_checkpoint,
    save_checkpoint,
    validation_result_is_better,
)
from utils.metrics import calculate_classification_metrics  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_every_model_config_loads(self) -> None:
        model_names = ["linear", "mlp", "cnn", "rnn", "transformer"]
        for model_name in model_names:
            config_path = ASSIGNMENT_ROOT / "configs" / "models" / f"{model_name}.yaml"
            config = load_experiment_config(config_path)
            self.assertEqual(config["model"]["name"], model_name)
            self.assertEqual(config["data"]["dataset"], "fashion_mnist")
            self.assertEqual(config["data"]["train_size"], 50000)
            self.assertEqual(config["data"]["validation_size"], 10000)
            self.assertEqual(config["training"]["max_epochs"], 100)
            self.assertEqual(config["training"]["early_stopping"]["patience"], 15)


class ModelTests(unittest.TestCase):
    def test_every_model_handles_batch_sizes_one_and_seven(self) -> None:
        loss_function = torch.nn.CrossEntropyLoss()

        for model_name in ["linear", "mlp", "cnn", "rnn", "transformer"]:
            config_path = ASSIGNMENT_ROOT / "configs" / "models" / f"{model_name}.yaml"
            config = load_experiment_config(config_path)
            for batch_size in [1, 7]:
                model = create_model(config["model"])
                images = torch.randn(batch_size, 1, 28, 28)
                targets = torch.arange(batch_size) % 10
                logits = model(images)
                self.assertEqual(
                    tuple(logits.shape), (batch_size, 10), msg=model_name
                )

                loss = loss_function(logits, targets)
                self.assertTrue(torch.isfinite(loss), msg=model_name)
                loss.backward()


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_save_and_load_round_trip(self) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        expected_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=3,
                validation_metrics={"loss": 0.4, "macro_f1": 0.8},
                config={"run": {"seed": 42}},
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)

            checkpoint = load_model_checkpoint(
                model, checkpoint_path, torch.device("cpu")
            )
            self.assertEqual(checkpoint["epoch"], 3)
            for name, parameter in model.state_dict().items():
                self.assertTrue(torch.equal(parameter, expected_parameters[name]))
            self.assertEqual(list(Path(temporary_directory).glob("*.tmp")), [])

    def test_validation_tie_uses_lower_loss_then_keeps_earlier_epoch(self) -> None:
        best_metrics = {"macro_f1": 0.8, "loss": 0.4}
        self.assertTrue(
            validation_result_is_better(
                {"macro_f1": 0.8, "loss": 0.3}, best_metrics, min_delta=0.0001
            )
        )
        self.assertFalse(
            validation_result_is_better(
                {"macro_f1": 0.8, "loss": 0.4}, best_metrics, min_delta=0.0001
            )
        )


class EvaluationPolicyTests(unittest.TestCase):
    def test_checkpoint_seed_is_used_and_mismatch_is_rejected(self) -> None:
        checkpoint = {
            "config": {
                "run": {"seed": 123},
                "reproducibility": {"development_seed": 123},
            }
        }
        self.assertEqual(get_checkpoint_seed(checkpoint, None), 123)
        self.assertEqual(get_checkpoint_seed(checkpoint, 123), 123)
        with self.assertRaisesRegex(ValueError, "does not match checkpoint seed"):
            get_checkpoint_seed(checkpoint, 42)

    def test_checkpoint_config_mismatch_is_rejected(self) -> None:
        checkpoint = {
            "config": {
                "model": {"name": "cnn", "parameters": {}},
                "preprocessing": {"normalization": {"enabled": False}},
            }
        }
        current_config = {
            "model": {"name": "mlp", "parameters": {}},
            "preprocessing": {"normalization": {"enabled": False}},
        }
        with self.assertRaisesRegex(ValueError, "model config does not match"):
            validate_checkpoint_config(checkpoint, current_config)

    def test_disabled_normalization_statistics_do_not_cause_mismatch(self) -> None:
        checkpoint = {
            "config": {
                "model": {"name": "cnn", "parameters": {}},
                "preprocessing": {
                    "normalization": {"enabled": False, "mean": None, "std": None},
                    "augmentation": {"enabled": False},
                },
            }
        }
        current_config = {
            "model": {"name": "cnn", "parameters": {}},
            "preprocessing": {
                "normalization": {"enabled": False, "mean": 0.28, "std": 0.35},
                "augmentation": {"enabled": False},
            },
        }
        validate_checkpoint_config(checkpoint, current_config)

    def test_official_test_requires_explicit_confirmation(self) -> None:
        validate_test_evaluation_request(smoke_test=False, allow_test=True)
        with self.assertRaisesRegex(ValueError, "requires --allow-test"):
            validate_test_evaluation_request(smoke_test=False, allow_test=False)
        with self.assertRaisesRegex(ValueError, "Smoke tests must not use"):
            validate_test_evaluation_request(smoke_test=True, allow_test=True)


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
                "checkpoint": "checkpoints/linear/seed42/best.pt",
                "test_metrics": {"accuracy": 0.8, "macro_f1": 0.7},
                "trainable_parameters": 100,
                "training_time_seconds": 10.0,
                "inference_timing": {"milliseconds_per_sample": 0.2},
            },
            {
                "model": "linear",
                "seed": 123,
                "checkpoint": "checkpoints/linear/seed123/best.pt",
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

    def test_same_checkpoint_is_counted_once(self) -> None:
        evaluation = {
            "model": "cnn",
            "seed": 42,
            "checkpoint": "checkpoints/cnn/seed42/best.pt",
            "test_metrics": {"accuracy": 0.9, "macro_f1": 0.89},
            "trainable_parameters": 1000,
            "training_time_seconds": 20.0,
            "inference_timing": {"milliseconds_per_sample": 0.4},
        }
        repeated_evaluation = dict(evaluation)
        repeated_evaluation["result_file"] = "second/test_results.json"

        summaries = summarize_by_model([evaluation, repeated_evaluation])
        self.assertEqual(summaries[0]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
