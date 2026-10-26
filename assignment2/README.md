# Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks

## Task track
- **Track:** Multi-class Image Classification
- **Dataset:** Food-101 (101 food categories, 101,000 images)

## Dataset proposal
See [`proposal/dataset_proposal.md`](proposal/dataset_proposal.md).  
- **Status:** Approved / In progress

## Setup
```bash
pip install -r ../requirements.txt
```

## Data preparation
To download and extract the official Food-101 dataset via torchvision:
```bash
python src/datasets/food101.py --download --out data/
```
The data will be organized in:
```text
data/
└── food-101/
    ├── images/
    │   ├── apple_pie/
    │   ├── baby_back_ribs/
    │   └── ...
    └── meta/
        ├── classes.txt
        ├── train.txt
        └── test.txt
```

## Training

### Baseline — Simple CNN trained from scratch (E1)
```bash
python src/train.py --config configs/baseline.yaml
```

### Pretrained ResNet-50 — Feature Extractor with Frozen Backbone (E2)
```bash
python src/train.py --config configs/pretrained_frozen.yaml
```

### Pretrained ResNet-50 — Full Fine-Tuning (E3)
```bash
python src/train.py --config configs/pretrained.yaml
```

### Quick Smoke Test / Verification (1 epoch, few batches)
```bash
python src/train.py --config configs/baseline.yaml --smoke-test
```

## Evaluation
Run evaluation on the test set:
```bash
python src/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/simple_cnn_best.pt --split test
python src/evaluate.py --config configs/pretrained.yaml --checkpoint checkpoints/resnet50_finetune/resnet50_best.pt --split test
```

## Checkpoints & Experiments Summary
| Model | Config | Status | Checkpoint | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---|---|---:|---:|---:|
| Simple CNN (Scratch) | `configs/baseline.yaml` | ✅ Completed (20 ep, 163m) | `checkpoints/baseline/simple_cnn_best.pt` | 46.38% | 51.96% | 51.21% |
| ResNet-50 (Frozen Backbone) | `configs/pretrained_frozen.yaml` | ✅ Completed (15 ep, 150m) | `checkpoints/resnet50_frozen/resnet50_best.pt` | 61.55% | 66.70% | 66.36% |
| ResNet-50 (Full Fine-tuned) | `configs/pretrained.yaml` | ✅ Completed (15 ep, 269m) | `checkpoints/resnet50_finetune/resnet50_best.pt` | **81.45%** | **86.17%** | **86.10%** |
