# Assignment 3 — Multimodal Deep Learning

## Task
[state chosen multimodal task]

## Dataset proposal
See `proposal/dataset_proposal.md`. Approval status: [Pending / Approved / Approved with conditions / Rejected]

## Setup
```bash
pip install -r ../requirements.txt
```

## Data preparation
```bash
python src/datasets/<dataset>.py --download --out data/
```

## Train
```bash
# unimodal baselines
python src/train.py --config configs/unimodal_a.yaml
python src/train.py --config configs/unimodal_b.yaml
# fusion models
python src/train.py --config configs/fusion_simple.yaml
python src/train.py --config configs/fusion_improved.yaml
```

## Evaluate
```bash
python src/evaluate.py --config configs/fusion_improved.yaml --checkpoint checkpoints/fusion_improved_best.pt
```

## Checkpoints
| Model | Path / link | Seed | Split | Commit |
|---|---|---|---|---|
| | | | | |
