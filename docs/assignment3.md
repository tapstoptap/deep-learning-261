---
title: Assignment 3
---

# Assignment 3 — Multimodal Deep Learning: VQA on a VQA v2 subset
### Task: Visual Question Answering (image + question → answer)

**Status:** M1 Proposal submitted (18 Nov 2026). M2 Draft submitted (25 Nov 2026). M3 Final: coming.

**Group 99 - PTQ members:**
- Nguyen Minh Phuc (2453005)
- Nguyen Thanh The (2453191)
- Nguyen Minh Quang (2453052)
**Instructor:** Lê Thành Sách

## Milestones

| Milestone | Due | Status |
|---|---|---|
| M1 Proposal | 18 Nov 2026 | Submitted 18 Nov 2026 |
| M2 Draft | 25 Nov 2026 | Submitted 25 Nov 2026 |
| M3 Final | 02 Dec 2026 | Coming |

---

## Problem Statement
Visual question answering: given a COCO photo and an English question about it, predict the answer. We frame it as **classification over the K = 771 most frequent training answers**.

## Dataset Description & EDA
- **Dataset:** VQA v2 (balanced real images) on MS COCO 2014 images.
- **Subset:** 15,000 images and 79,731 question–image pairs (after removing 12 train/val questions during cleaning).
- **Split:** by image, seed 261 — train 48,245 / val 5,265 / test 26,221 questions.
- **Proposal:** [assignment3/PROPOSAL.md](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/PROPOSAL.md) has the data source and license, pair-quality audit, split and leakage checks, the choice of K, planned models and metrics, and the measured compute estimate. Approval status: no formal decision received.

## Methodology
Frozen ResNet-50 (pooled 2048-d) and frozen BERT-base ([CLS], 768-d) features, each L2-normalised, followed by an MLP classifier over 771 answers trained with soft-score BCE. Compared: a question-type prior, image only, question only, and concatenation fusion.

## Experimental Setup
20 epochs of AdamW (lr 1e-3, weight decay 1e-4, batch 256), seeds 261, 262 and 263, best epoch chosen on val. Test was run once on the chosen checkpoints.

## Results
Preliminary (M2). Test split, 26,221 questions, official VQA accuracy in %, mean ± std over 3 seeds; oracle ceiling 87.10.

| model | VQA acc | exact match | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| a: prior (no training) | 31.85 | 25.46 | 63.42 | 26.64 | 9.32 | 24.38 |
| c: image only (unimodal A) | 24.29 ± 0.47 | 19.14 ± 0.50 | 62.84 ± 1.30 | 0.41 ± 0.11 | 1.46 ± 0.16 | 0.46 ± 0.29 |
| b: text only (unimodal B) | 38.59 ± 0.10 | 30.79 ± 0.23 | 65.43 ± 0.15 | 28.70 ± 0.84 | 20.91 ± 0.36 | 32.07 ± 1.08 |
| d: concat fusion | 42.98 ± 0.22 | 34.79 ± 0.19 | 66.21 ± 0.34 | 28.04 ± 0.24 | 29.37 ± 0.40 | 38.12 ± 0.89 |

Validation results, hypotheses and example cases are in the [M2 draft](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/report/M2_DRAFT.md).

## Comparison and Discussion
Coming (M3).

## Error Analysis
Coming (M3).

## Ablation Study
Coming (M3).

## Compute-Cost Analysis
Measured costs for the M2 runs are in the [M2 draft](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/report/M2_DRAFT.md), Sec. 7. Full analysis: M3.

## Limitations and Conclusion
Coming (M3).

## Links
- Dataset proposal: [GitHub](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/PROPOSAL.md)
- Source code: [GitHub](https://github.com/tapstoptap/deep-learning-261/tree/main/assignment3)
- M2 draft report: [GitHub](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/report/M2_DRAFT.md)
- Reproduction instructions: [GitHub](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/README.md)
- Checkpoint downloads: coming
- Final report / slides: coming
- YouTube presentation video: coming

## AI Usage Disclosure (Assignment 3)
AI assistance (Claude) was used for planning, code generation and review. See [AI_USAGE.md](https://github.com/tapstoptap/deep-learning-261/blob/main/AI_USAGE.md) for the full log (entries #10–#15).
