# Assignment 3 — Multimodal Deep Learning

## Task
Visual question answering on a VQA v2 subset, framed as classification over the 771 most frequent train answers.

## Documents
- Assignment page: https://tapstoptap.github.io/deep-learning-261/assignment3
- M1 proposal: [PROPOSAL.md](PROPOSAL.md). Approval status: no formal decision received.
- M2 draft: [report/M2_DRAFT.md](report/M2_DRAFT.md)

## Setup
Python 3.14 with the packages in `eda/requirements-eda.txt`, plus torch, torchvision, transformers and pyyaml (versions in the M2 draft, Sec. 7).

## Data preparation
Download VQA v2 and the subset images into `eda/data/vqa_v2/` as described in `eda/PHASE3_GUIDE.md`. Split id lists are in `splits/`.

```bash
python src/extract_features.py
```

## Train
```bash
python src/train.py --config configs/unimodal_a.yaml     # image only
python src/train.py --config configs/unimodal_b.yaml     # question only
python src/train.py --config configs/fusion_simple.yaml  # concat fusion
```
The improved fusion config is a placeholder until M3.

## Evaluate
```bash
python src/evaluate.py --split val
python src/evaluate.py --split test   # once per final checkpoint
python src/report_tables.py
python src/plot_curves.py
```

## Checkpoints
Not stored in git. They are reproduced by the training commands above.

| Model | Path | Seed | Best score (VQA acc) | Commit | sha256 (first 16) |
|---|---|---|---|---|---|
| c: image only | `checkpoints/unimodal_a_s261.pt` (local) | 261 | val 25.40 | `a0a3a9c` | `89d564e2f41c8f1d` |
| c: image only | `checkpoints/unimodal_a_s262.pt` (local) | 262 | val 25.77 | `a0a3a9c` | `e9891498a2595e78` |
| c: image only | `checkpoints/unimodal_a_s263.pt` (local) | 263 | val 25.47 | `a0a3a9c` | `39bfa15ac130d607` |
| b: text only | `checkpoints/unimodal_b_s261.pt` (local) | 261 | val 39.30 | `a0a3a9c` | `251f80e5fc24ccc9` |
| b: text only | `checkpoints/unimodal_b_s262.pt` (local) | 262 | val 39.31 | `a0a3a9c` | `d252259e48f1c57f` |
| b: text only | `checkpoints/unimodal_b_s263.pt` (local) | 263 | val 38.95 | `a0a3a9c` | `8cb79528a9c06ce1` |
| d: concat fusion | `checkpoints/fusion_simple_s261.pt` (local) | 261 | val 44.15 | `a0a3a9c` | `10893dcd249b60ee` |
| d: concat fusion | `checkpoints/fusion_simple_s262.pt` (local) | 262 | val 43.80 | `a0a3a9c` | `ee6bb51d9bad8d61` |
| d: concat fusion | `checkpoints/fusion_simple_s263.pt` (local) | 263 | val 43.70 | `a0a3a9c` | `31575342ae1c7a98` |
