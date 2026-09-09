---
title: Assignment 1 Experiment Protocol
---

# Assignment 1 Experiment Protocol

This document is the human-readable contract for the shared configuration in
`assignment1/configs/protocol.yaml`. It records what every model must share, why
the choices were made, which values are still provisional, and how changes are
approved. Architecture and training settings are kept in each model YAML because
different model families may require different training choices. The YAML files
are consumed by code; this page is used for group review, the report, and
reproducibility evidence.

## Status

- Protocol ID: `a1`
- Status: Draft
- Approval: Pending group review
- Freeze point: Before the main five-model comparison

Values marked as pending must be resolved after a smoke test and before the
protocol status changes to `frozen`.

## Shared experiment rules

All five required models use Fashion-MNIST, one saved stratified split, the same
evaluation implementation, the same checkpoint rule, and the same timing method.
MNIST may be used for debugging only and must not appear in the main comparison.

The official 60,000-sample training set is split into 54,000 training and 6,000
validation samples with split seed `42`. The official 10,000-sample test set is
kept unchanged. Changing the training seed must not recreate the split.

The development run uses seed `42`. Main experiments are planned for seeds
`42`, `123`, and `2026`, with mean and standard deviation reported across runs.
If compute limitations force a smaller plan, the change and its limitation must
be documented before inspecting final test results.

## Preprocessing

Inputs have shape `[batch, 1, 28, 28]`. Normalization statistics must be measured
using only the final training subset; validation and test samples must not
contribute to them. Data augmentation is disabled in the initial controlled
comparison. Any later augmentation experiment must be reported separately.

## Training policy

Architecture and training parameters belong in `assignment1/configs/models/`.
Each model currently starts with batch size 128, at most 50 epochs, cross-entropy
loss, and Adam with learning rate `0.001` and weight decay `0.0001`. Early stopping
monitors validation macro-F1 with patience 7.

These are starting values rather than claims that one training recipe is optimal
for every architecture. They may be adjusted independently during development,
but every final value must remain in the corresponding model YAML and be reported
with that model's results. Dataset split, run seeds, evaluation, checkpoint
selection, and timing rules remain shared.

## Model selection and test policy

The best checkpoint is selected by maximum validation macro-F1. Ties are broken
by lower validation loss and then by the earlier epoch. The official test set is
evaluated only after model and checkpoint selection; it must not be used for
hyperparameter tuning or early stopping.

Required metrics are accuracy and macro-F1. The final submission must also report
parameter count, training time, inference time, learning curves, confusion
matrices, and representative correct and incorrect predictions.

## Timing policy

Inference timing uses float32 forward passes, five warm-up batches, and three
measurement repetitions. Before main runs, the group must record the benchmark
machine, accelerator, framework versions, inference batch size, and device
synchronization method. Runtime comparisons are valid only on the same hardware
and with the same measurement procedure.

## Configuration ownership

```text
assignment1/configs/protocol.yaml
    Shared data, reproducibility, evaluation, checkpoint and timing rules

assignment1/configs/models/*.yaml
    Architecture-specific and training parameters

assignment1/configs/splits/
    Saved train/validation split manifest
```

The training command loads a model YAML file and then resolves its
`protocol_file`. The resolved configuration and Git commit should be copied into
each main-run result directory.

## Decisions still required before freeze

- Measure and record normalization mean and standard deviation from the training split.
- Confirm a feasible batch size for each model and record it in its YAML file.
- Record the exact benchmark CPU/GPU and software versions.
- Confirm that deterministic execution is supported by the selected operations.
- Estimate whether all 15 planned main runs fit the available compute budget.
- Add member names to `approved_by` and change the protocol status to `frozen`.

## Change control

Before the freeze, update the relevant YAML and record the reason in the Pull
Request. After the freeze, any change that affects comparability requires rerunning
the affected model and documenting the change. Never change a value only because a
test-set result looks unfavorable.
