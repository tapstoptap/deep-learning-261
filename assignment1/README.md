# Assignment 1 — Fashion-MNIST Benchmarks

This directory contains the Assignment 1 training and evaluation pipeline and
the five-family Fashion-MNIST benchmark: Linear, MLP, CNN, GRU, and Transformer.

## Dataset and protocol

- Official Fashion-MNIST training set: 60,000 images.
- Saved stratified split: 50,000 training and 10,000 validation images.
- Official test set: 10,000 images.
- Split seed: `42`.
- Training seeds: `42`, `123`, and `2026`.
- Primary model-selection metric: validation macro-F1.
- Reported test metrics: accuracy and macro-F1.

The committed split manifest is
`configs/splits/fashion_mnist_train50000_val10000_seed42.json`. Training seeds
change initialization, dropout, and batch order; they do not change the data
split.

## Training configuration

- Batch size: `128`.
- Maximum epochs: `100`.
- Optimizer: Adam, learning rate `0.001`, weight decay `0.0001`.
- Early stopping: validation macro-F1, patience `15`, `min_delta=0.0001`.
- Mean/std normalization is enabled using statistics from the saved training
  subset; augmentation and learning-rate scheduling are disabled.
- Best-checkpoint tie-break: lower validation loss, then earlier epoch.

Shared settings are in `configs/protocol.yaml`; model settings are in
`configs/models/<model>.yaml`.

## Reproduction on Windows PowerShell

Run every command from `assignment1/`. Activate the environment containing
PyTorch first, then save its Python executable:

```powershell
conda activate uav
$pythonUav = (Get-Command python).Source
```

Prepare the dataset and verify the saved split:

```powershell
& $pythonUav src/datasets/fashion_mnist.py `
    --out data `
    --split-file configs/splits/fashion_mnist_train50000_val10000_seed42.json `
    --split-seed 42 `
    --validation-size 10000 `
    --download `
    --calculate-normalization
```

Run the test suite:

```powershell
& $pythonUav -m unittest discover -s tests -v
```

Train all five models for all three seeds:

```powershell
$seeds = @(42, 123, 2026)
$models = @("linear", "mlp", "cnn", "rnn", "transformer")

foreach ($seed in $seeds) {
    foreach ($model in $models) {
        & $pythonUav src/train.py `
            --config "configs/models/$model.yaml" `
            --seed $seed `
            --device cuda `
            --num-workers 0
    }
}
```

Evaluate each selected checkpoint. Evaluation reads the seed from the
checkpoint; `--allow-test` explicitly confirms that model selection is complete.

```powershell
$seeds = @(42, 123, 2026)
$models = @("linear", "mlp", "cnn", "rnn", "transformer")

foreach ($seed in $seeds) {
    foreach ($model in $models) {
        $run = Get-ChildItem -LiteralPath "checkpoints/$model" -Directory |
            Where-Object { $_.Name -like "${model}_seed${seed}_*" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($null -eq $run) {
            throw "Cannot find checkpoint: model=$model, seed=$seed"
        }

        & $pythonUav src/evaluate.py `
            --config "configs/models/$model.yaml" `
            --checkpoint (Join-Path $run.FullName "best.pt") `
            --device cuda `
            --num-workers 0 `
            --allow-test
    }
}
```

Create the aggregate table:

```powershell
& $pythonUav src/summarize_results.py `
    --results-dir results `
    --output-file results/model_comparison.csv
```

Raw run outputs and checkpoint binaries remain local. Portable artifacts are
tracked under `results/benchmark/`.

## Milestone results

The table reports sample mean ± sample standard deviation over three seeds.

| Model | Test accuracy | Test macro-F1 | Parameters |
|---|---:|---:|---:|
| Linear | 84.31 ± 0.09% | 84.23 ± 0.06% | 7,850 |
| MLP | 89.37 ± 0.11% | 89.31 ± 0.12% | 235,146 |
| CNN | **92.22 ± 0.39%** | **92.20 ± 0.41%** | 421,642 |
| GRU (configured as `rnn`) | 90.29 ± 0.19% | 90.26 ± 0.21% | 61,962 |
| Transformer | 89.40 ± 0.09% | 89.37 ± 0.13% | 71,242 |

CNN has the highest mean test accuracy and macro-F1 in this comparison.
Detailed per-seed metrics, histories, curves, confusion matrices, and prediction
examples are in `results/benchmark/`. The Transformer seed 2026 run took much
longer than its other seeds; inspect `run_metrics.csv` before comparing runtime.

## Output layout

```text
results/
├── eda/
├── linear/                 # ignored raw runs and evaluations
├── mlp/                    # ignored raw runs and evaluations
├── cnn/                    # ignored raw runs and evaluations
├── rnn/                    # ignored raw runs and evaluations
├── transformer/            # ignored raw runs and evaluations
└── benchmark/              # tracked portable artifacts

checkpoints/                # ignored checkpoint binaries, grouped by model
```

See the [experiment protocol](../docs/assignment1-experiment-protocol.md) and
the [milestone report](../docs/assignment1.md) for methodology and discussion.
