# Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks

## Task track
[state chosen track]

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
python src/train.py --config configs/<model>.yaml
```

## Evaluate
```bash
python src/evaluate.py --config configs/<model>.yaml --checkpoint checkpoints/<model>_best.pt
```

## Checkpoints
| Model | Path / link | Seed | Split | Commit |
|---|---|---|---|---|
| | | | | |
