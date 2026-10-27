from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml

ASSIGNMENT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ASSIGNMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from datasets.food101 import stratified_split
from models import count_parameters, create_model
from models.baseline import SimpleCNN
from models.pretrained import ResNet50Classifier
from utils.metrics import (
    calculate_accuracy,
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_macro_f1,
    get_top_confused_pairs,
)


class ConfigTests(unittest.TestCase):
    def test_configs_load_properly(self) -> None:
        configs = ["baseline.yaml", "pretrained.yaml", "pretrained_frozen.yaml"]
        for cfg_file in configs:
            cfg_path = ASSIGNMENT_ROOT / "configs" / cfg_file
            self.assertTrue(cfg_path.exists(), f"Config file missing: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.assertIn("seed", data)
            self.assertIn("data", data)
            self.assertIn("model", data)
            self.assertIn("training", data)
            self.assertEqual(data["model"]["num_classes"], 101)


class ModelTests(unittest.TestCase):
    def test_simple_cnn_forward(self) -> None:
        model = SimpleCNN(in_channels=3, num_classes=101, channels=[16, 32], dropout=0.1)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 101))

    def test_create_model_factory(self) -> None:
        model_baseline = create_model({"name": "simple_cnn", "num_classes": 101, "channels": [16, 32]})
        self.assertIsInstance(model_baseline, SimpleCNN)

        params = count_parameters(model_baseline)
        self.assertGreater(params["total"], 0)
        self.assertEqual(params["trainable"], params["total"])

    def test_pretrained_freeze_flag(self) -> None:
        # Test without downloading weights
        model_frozen = ResNet50Classifier(num_classes=101, pretrained=False, freeze_backbone=True)
        counts = count_parameters(model_frozen)
        self.assertGreater(counts["frozen"], 0)
        self.assertLess(counts["trainable"], counts["total"])


class MetricsTests(unittest.TestCase):
    def test_metrics(self) -> None:
        targets = [0, 1, 2, 1, 0]
        preds = [0, 1, 1, 1, 0]

        acc = calculate_accuracy(targets, preds)
        self.assertAlmostEqual(acc, 0.8)

        f1 = calculate_macro_f1(targets, preds)
        self.assertGreater(f1, 0.0)

        cm = calculate_confusion_matrix(targets, preds, labels=[0, 1, 2])
        self.assertEqual(cm.shape, (3, 3))

        top_confusions = get_top_confused_pairs(targets, preds, class_names=["A", "B", "C"])
        self.assertEqual(len(top_confusions), 1)
        self.assertEqual(top_confusions[0]["true_class"], "C")
        self.assertEqual(top_confusions[0]["pred_class"], "B")


class DatasetSplitTests(unittest.TestCase):
    def test_stratified_split(self) -> None:
        samples = []
        for c in range(10):
            for i in range(100):
                samples.append((f"class_{c}/img_{i}", c))

        train_s, val_s = stratified_split(samples, val_ratio=0.1, seed=42)
        self.assertEqual(len(train_s), 900)
        self.assertEqual(len(val_s), 100)


if __name__ == "__main__":
    unittest.main()
