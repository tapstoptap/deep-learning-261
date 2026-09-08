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
python src/train.py --config configs/<model>.yaml
```

## Evaluate
```bash
python src/evaluate.py --config configs/<model>.yaml --checkpoint checkpoints/<model>_best.pt
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
