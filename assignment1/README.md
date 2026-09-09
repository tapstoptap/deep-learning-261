# Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

## Dataset
- Debugging: MNIST
- Main comparison: Fashion-MNIST
- Optional extension: CIFAR-10

## Setup
```bash
pip install -r ../requirements.txt
```

## Data preparation
```bash
python src/datasets/fashion_mnist.py --download --out data/
```

## Train
```bash
python src/train.py --config configs/models/<model>.yaml
```

Every model config references `configs/protocol.yaml`, which contains the shared
dataset split, seeds, evaluation, checkpoint, and timing rules. Model files contain
their own architecture and training settings so they can be adjusted independently. See the
[experiment protocol](../docs/assignment1-experiment-protocol.md) for the rationale
and current decision status.

## Evaluate
```bash
python src/evaluate.py --config configs/models/<model>.yaml --checkpoint checkpoints/<model>_best.pt
```

## Models implemented
- [ ] Linear / softmax
- [ ] MLP
- [ ] CNN
- [ ] LSTM/GRU
- [ ] Transformer

## Checkpoints
| Model | Path / link | Seed | Split | Commit |
|---|---|---|---|---|
| | | | | |
