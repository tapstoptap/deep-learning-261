# Assignment 2 — Checkpoints and Reconstruction

The checkpoint binaries are **not** stored in Git. This file is the reconstruction contract: it records the data split, commands, expected artifacts, and checkpoint-selection rule. The commands regenerate the configurations and splits, and the archived artifacts in `results/` are the source of the reported values; the provenance limits in Section 7 apply.

## 1. Environment

- Training: Kaggle GPU notebooks (Tesla T4 for E1/E2, Tesla P100 for E3) and Colab (A0 control, GPU model not logged). PyTorch 2.x + torchvision on the hosted images.
- Instrumented Kaggle session (inference benchmark, qualitative grid): `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, CUDA 12.8, cuDNN 91900, Python 3.13.15, Tesla T4.
- Local analysis/figures: Python 3.12.5, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu.
- Seed: `42` for data splitting, initialization, and training. No test-time augmentation.
- Data split: 90/10 stratified hold-out from the official training split; the official 25,250-image test split is untouched.

The training-session builds of E1--E3 were not logged. Capture the exact versions before a rerun:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

## 2. Data preparation

From `assignment2/`:

```bash
pip install -r ../requirements.txt
python src/datasets/food101.py --download --out data/
python src/datasets/food101.py --root data/food-101
```

The verification command must print:

```text
Classes: 101
Train samples: 68175
Validation samples: 7575
Test samples: 25250
```

## 3. Training commands

```bash
# E1 — Simple CNN from scratch
python src/train.py --config configs/baseline.yaml

# E2 — ResNet-50, frozen backbone
python src/train.py --config configs/pretrained_frozen.yaml

# E3 — ResNet-50, full fine-tuning
python src/train.py --config configs/pretrained.yaml

# Clean one-factor ablation: frozen backbone with the E3 optimization protocol
python src/train.py --config configs/ablation_frozen_e3proto.yaml
```

Each run writes `history.json`, `history.csv`, `summary.json`, `training_curves.png`, and the best checkpoint under the configured `checkpoint_dir`. Checkpoint selection is **best validation macro-F1**.

## 4. Evaluation commands

```bash
python src/evaluate.py --config configs/baseline.yaml \
  --checkpoint checkpoints/baseline/simple_cnn_best.pt --split test

python src/evaluate.py --config configs/pretrained_frozen.yaml \
  --checkpoint checkpoints/resnet50_frozen/resnet50_best.pt --split test

python src/evaluate.py --config configs/pretrained.yaml \
  --checkpoint checkpoints/resnet50_finetune/resnet50_best.pt --split test
```

## 5. Expected results

| Run | Checkpoint path | Val macro-F1 | Test accuracy | Test macro-F1 | Test loss |
|---|---|---:|---:|---:|---:|
| E1 Simple CNN | `checkpoints/baseline/simple_cnn_best.pt` | 46.38% | 51.96% | 51.21% | 1.9649 |
| E2 ResNet-50 frozen | `checkpoints/resnet50_frozen/resnet50_best.pt` | 61.55% | 66.70% | 66.36% | 1.2933 |
| A0 frozen, E3 protocol | `checkpoints/ablation_frozen_e3proto/resnet50_best.pt` | 56.61% | 61.71% | 60.88% | 1.6553 |
| E3 ResNet-50 fine-tuned | `checkpoints/resnet50_finetune/resnet50_best.pt` | 81.45% | 86.17% | 86.10% | 0.6494 |

These values match `results/*/summary.json` and `results/*/test_metrics.json`. Training times are wall-clock (T4 for E1/E2, P100 for E3, unlogged Colab GPU for A0) and are not hardware-matched. A0 and E3 share the optimisation protocol and differ in `freeze_backbone` (hardware also differs: unlogged Colab GPU vs P100; the E3 cosine schedule is confirmed by the exported Kaggle run, while A0's schedule is confirmed only by the committed config because the Colab session is not logged), so in the two recorded runs `E3 - A0` = +24.46 points test accuracy, +25.22 points macro-F1; the naive `E2 -> E3` gap of +19.47 points is smaller. Frozen parameters do not freeze BatchNorm statistics: the backbone runs in training mode and its running stats keep updating.

Measured end-to-end inference cost (Tesla T4, batch 32, 20 measured batches, fp32; timer covers data loading, host-to-device copy, and forward): E2 220.70 img/s / 144.99 ms per batch, E3 223.47 img/s / 143.19 ms, 530.15 MB peak allocated VRAM for both (`results/*/inference_benchmark.json`).

## 6. Additional checks

```bash
# Inference throughput/latency and peak allocated memory
python src/benchmark_inference.py --config configs/pretrained.yaml \
  --checkpoint checkpoints/resnet50_finetune/resnet50_best.pt --split test

# Correct/incorrect prediction grid
python src/qualitative_examples.py --config configs/pretrained_frozen.yaml \
  --checkpoint checkpoints/resnet50_frozen/resnet50_best.pt --split test
```

## 7. Known gaps

- The full E1 and E3 confusion matrices were not exported; only the saved top-10 pairs are available. E2's full matrix is in `results/resnet50_frozen/test_confusion_matrix.csv`.
- The A0 run's `history.json`/`history.csv` and the GPU model were not exported; `summary.json`, `test_metrics.json`, and `training_curves.png` are archived in `results/ablation_frozen_e3proto/`.
- The training-session builds of E1--E3 were not logged; only the benchmark session reports exact versions.
- Inference numbers come from a single fp32 session without TensorRT or mixed precision, and cover the end-to-end pipeline rather than the forward pass alone; E1 was not benchmarked.
- The E3 test metrics were saved by the original Kaggle run in `results/resnet50_finetune/test_metrics.json`. The full-precision accuracy there corresponds to 21,758.5 correct out of 25,250 and cannot be reproduced bit-exactly by `src/evaluate.py`, which counts integer predictions; the rounded values (86.17% / 86.10%) are the reported ones.
- The saved E1 history records a constant learning rate (E1 has no scheduler). The saved E3 history also records a constant 1e-4 rate although `configs/pretrained.yaml` specifies cosine annealing; the exported Kaggle run confirms the cosine schedule was applied in E3 (the history `lr` column is a curated artifact). A0 used the same committed config (`scheduler: cosine`), but its Colab session is not logged.
- The E1 history records `train_macro_f1` identical to `val_macro_f1`, and the E3 history leaves `train_macro_f1` empty, so both were written by an earlier version of `train.py`; the report figures use validation macro-F1 only.
- `--smoke-test` writes to `*_smoke` checkpoint/result directories unless `--checkpoint-dir` / `--result-dir` are given explicitly, so it cannot overwrite an official run.
