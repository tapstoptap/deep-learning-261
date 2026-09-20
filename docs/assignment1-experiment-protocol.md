---
title: Assignment 1 Experiment Protocol
---

# Assignment 1 Experiment Protocol

This document is the human-readable contract for
`assignment1/configs/protocol.yaml`. The 23 September milestone covers the
Linear and MLP baselines; later Assignment 1 models must follow the same shared
rules before their results can be added to the comparison.

## Status

- Protocol ID: `a1`.
- Status: executed draft.
- Approval: pending group review.
- Completed main runs: six (two models × three training seeds).

## Dataset and split

Fashion-MNIST provides 60,000 official training images and 10,000 official test
images. The official training set is divided into a saved, stratified split:

- 50,000 training images;
- 10,000 validation images;
- split seed `42`;
- exactly 5,000 training and 1,000 validation images per class.

The test set remains unchanged. The committed manifest is
`assignment1/configs/splits/fashion_mnist_train50000_val10000_seed42.json`.
Changing a training seed must not recreate this partition.

## Reproducibility

The main training seeds are `42`, `123`, and `2026`. They control model
initialization, dropout, and training-loader order. Mean and sample standard
deviation are reported across the three runs.

Inputs have shape `[batch, 1, 28, 28]` and are scaled to `[0, 1]`. The saved
50,000-image training subset has mean `0.2862758040` and standard deviation
`0.3531853259`. Additional normalization and data augmentation are disabled.

## Training policy

Every current model uses:

- batch size `128`;
- at most `100` epochs;
- cross-entropy loss;
- Adam with learning rate `0.001` and weight decay `0.0001`;
- no learning-rate scheduler;
- early stopping on validation macro-F1 with patience `15` and
  `min_delta=0.0001`.

The best checkpoint is selected by validation macro-F1. Values within
`min_delta` use lower validation loss as the tie-break; an exact remaining tie
keeps the earlier epoch.

## Test-set policy

The official test set is not used for optimization, early stopping, or
hyperparameter selection. `evaluate.py` requires `--allow-test`, rejects smoke
tests on the official test set, reads the training seed from the checkpoint, and
rejects seed, model, preprocessing, or split mismatches.

Accuracy and macro-F1 are the required predictive metrics. Parameter count,
training time, inference time, learning curves, confusion matrices, and
representative predictions are also retained. Repeated evaluation of the same
`(model, seed, checkpoint)` is counted only once in aggregate statistics.

## Checkpoint and timing policy

Best and last checkpoints are written atomically through a temporary file and
replacement. Checkpoint binaries stay local; portable metrics and figures are
stored in `assignment1/results/benchmark/`.

Inference timing uses float32 forward passes, five warm-up batches, three
measurement repetitions, and CUDA synchronization around each timed pass. All
reported milestone runs use an NVIDIA GeForce RTX 4050 Laptop GPU and batch
size 128. Runtime comparisons apply only to this environment and procedure.

## Configuration ownership

```text
assignment1/configs/protocol.yaml
    Shared data, reproducibility, evaluation, checkpoint, timing and paths

assignment1/configs/models/*.yaml
    Architecture and training settings

assignment1/configs/splits/
    Saved train/validation split manifest
```

## Verification checklist

- [x] Save and validate the 50,000/10,000 stratified split.
- [x] Measure statistics only on the training subset.
- [x] Train Linear and MLP with all three training seeds.
- [x] Evaluate six selected checkpoints on the official test set.
- [x] Run the 13-test unit suite.
- [x] Record the benchmark hardware and package versions.
- [ ] Obtain group review and populate `approved_by` before changing protocol
  status to `frozen`.

## Change control

After the protocol is frozen, a change to the split, preprocessing, optimizer,
early stopping, or evaluation procedure requires rerunning every affected model.
Test results must not be used to choose a favorable configuration.
