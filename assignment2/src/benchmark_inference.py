"""Measure end-to-end inference cost: data loading, host-to-device copy, forward.

The timer spans the DataLoader iterator and the model forward, so the reported
throughput and latency describe the serving pipeline rather than the forward
pass alone. Increase ``--num-workers`` to overlay loading with compute.

Run in the training environment with a restored checkpoint, e.g.:

    python src/benchmark_inference.py \
        --config configs/pretrained.yaml \
        --checkpoint checkpoints/resnet50_finetune/resnet50_best.pt \
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from datasets.food101 import create_datasets
from models import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Food-101 inference cost")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 32)

    root = args.data_root or data_cfg.get("root", "data/food-101")
    train_ds, val_ds, test_ds, _ = create_datasets(
        root=root,
        image_size=data_cfg.get("image_size", 224),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=config.get("seed", 42),
    )
    dataset = val_ds if args.split == "val" else test_ds
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(args.warmup_batches):
            try:
                images, _ = next(iterator)
            except StopIteration:
                break
            model(images.to(device))

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    samples = 0
    batches = 0
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.measure_batches):
            try:
                images, _ = next(iterator)
            except StopIteration:
                break
            images = images.to(device, non_blocking=True)
            model(images)
            samples += images.size(0)
            batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = {
        "device": str(device),
        "batch_size": batch_size,
        "samples": samples,
        "batches": batches,
        "elapsed_sec": elapsed,
        "latency_ms_per_batch": (elapsed / max(batches, 1)) * 1000.0,
        "throughput_images_per_sec": samples / elapsed if elapsed > 0 else None,
        "peak_allocated_vram_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else None
        ),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
    }

    output_path = Path(args.output) if args.output else Path("results") / Path(args.checkpoint).parent.name / "inference_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
