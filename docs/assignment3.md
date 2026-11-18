---
title: Assignment 3
---

# Assignment 3 — Multimodal Deep Learning: VQA on a VQA v2 subset
### Task: Visual Question Answering (image + question → answer)

**Status:** M1 Proposal submitted (18 Nov 2026). M2 draft and M3 final: coming.

**Group 99 - PTQ members:**
- Nguyen Minh Phuc (2453005)
- Nguyen Thanh The (2453191)
- Nguyen Minh Quang (2453052)
**Instructor:** Lê Thành Sách

## Milestones

| Milestone | Due | Status |
|---|---|---|
| M1 Proposal | 18 Nov 2026 | Submitted 18 Nov 2026 |
| M2 Draft | 25 Nov 2026 | Coming |
| M3 Final | 02 Dec 2026 | Coming |

---

## Problem Statement
Visual question answering: given a COCO photo and an English question about it, predict the answer. We frame it as **classification over the K = 771 most frequent training answers**.

## Dataset Description & EDA
- **Dataset:** VQA v2 (balanced real images) on MS COCO 2014 images.
- **Subset:** 15,000 images and 79,731 question–image pairs (after removing 12 train/val questions during cleaning).
- **Split:** by image, seed 261 — train 48,245 / val 5,265 / test 26,221 questions.
- **Proposal:** [assignment3/PROPOSAL.md](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/PROPOSAL.md) has the data source and license, pair-quality audit, split and leakage checks, the choice of K, planned models and metrics, and the measured compute estimate. Approval status: submitted for M1, awaiting review.

## Methodology
Coming (M2).

## Experimental Setup
Coming (M2).

## Results
Coming (M3).

## Comparison and Discussion
Coming (M3).

## Error Analysis
Coming (M3).

## Ablation Study
Coming (M3).

## Compute-Cost Analysis
Coming (M3). The M1 compute estimate is in the proposal.

## Limitations and Conclusion
Coming (M3).

## Links
- Dataset proposal: [GitHub](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment3/PROPOSAL.md)
- Source code: [GitHub](https://github.com/tapstoptap/deep-learning-261/tree/main/assignment3)
- Checkpoints / reproduction instructions: coming
- Report / slides: coming
- YouTube presentation video: coming

## AI Usage Disclosure (Assignment 3)
AI assistance (Claude) was used for planning, code generation and review. See [AI_USAGE.md](https://github.com/tapstoptap/deep-learning-261/blob/main/AI_USAGE.md) for the full log.
