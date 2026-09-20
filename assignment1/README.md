# Assignment 1 — Fashion-MNIST Benchmarks

This directory contains the Assignment 1 training and evaluation pipeline. The
23 September milestone reports the Linear and MLP baselines. CNN, RNN, and
Transformer modules and configs remain available for later Assignment 1 work,
but they are not included in the current result table.

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
- Augmentation, additional normalization, and learning-rate scheduling: disabled.
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

Train Linear and MLP for all three seeds:

```powershell
$seeds = @(42, 123, 2026)
$models = @("linear", "mlp")

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
$models = @("linear", "mlp")

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
| Linear | 84.58 ± 0.10% | 84.48 ± 0.14% | 7,850 |
| MLP | **89.18 ± 0.21%** | **89.15 ± 0.22%** | 235,146 |

The MLP improves macro-F1 by approximately 4.67 percentage points over the
linear baseline. Detailed per-seed metrics, histories, curves, confusion
matrices, and prediction examples are in `results/benchmark/`.

## Output layout

```text
results/
├── eda/
├── linear/                 # ignored raw runs and evaluations
├── mlp/                    # ignored raw runs and evaluations
└── benchmark/              # tracked portable artifacts

checkpoints/
├── linear/                 # ignored checkpoint binaries
└── mlp/
```

See the [experiment protocol](../docs/assignment1-experiment-protocol.md) and
the [milestone report](../docs/assignment1.md) for methodology and discussion.
