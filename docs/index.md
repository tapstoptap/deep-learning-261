---
title: Home
---

# Deep Learning and Its Applications — Course Project

## Course Project · Semester 261

Comparative experiments on deep learning architectures by **Group 99 — PTQ (choco)**.

| Course | Instructor | Faculty |
|---|---|---|
| CO3133 | Lê Thành Sách | Computer Science and Engineering, HCMUT |

## Team

| Member | Student ID | GitHub |
|---|---:|---|
| Nguyen Minh Phuc | 2453005 | [tapstoptap](https://github.com/tapstoptap) |
| Nguyen Thanh The | 2453191 | [Themenli1](https://github.com/Themenli1) |
| Nguyen Minh Quang | 2453052 | [quangnguyencomeng](https://github.com/quangnguyencomeng) |

| Member | Main contribution |
|---|---|
| Nguyen Minh Phuc | Repository skeleton and Food-101 reference check; Assignment 1 not contributed. |
| Nguyen Thanh The | Assignment 2 proposal, implementation, experiments, report, figures; Assignment 1 not contributed. |
| Nguyen Minh Quang | Assignment 1 sole author. |

[View the source repository on GitHub](https://github.com/tapstoptap/deep-learning-261)

---
# Assignment 1

## Foundations of Deep Learning Pipelines and Architectures

**Sole author:** Nguyen Minh Quang · 2453052

Nguyen Minh Phuc and Nguyen Thanh The are members of the course project, but
they did not contribute to Assignment 1.

This assignment compares Linear, MLP, CNN, GRU, and Transformer classifiers on
Fashion-MNIST under a shared split and evaluation protocol.

[Open the complete Assignment 1 page](assignment1)

[Download the Assignment 1 report PDF](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment1/report/DL_A1_report.pdf)

[Watch the presentation](https://youtu.be/nHcVq0U0Zc4)

---

# Assignment 2

## Deep Learning on Large-Scale Data and Specialized Tasks

**Task track:** image classification on Food-101 (101 classes, 101,000 images).  
**Status:** M3 Final complete; report, slides, and presentation video published.

Three adaptation regimes plus a matched frozen control were trained on the same 68,175/7,575/25,250 split and evaluated once on the official 25,250-image test set:

| Run | Model | Adaptation | Test accuracy | Test macro-F1 |
|---|---|---|---:|---:|
| E1 | Simple CNN | Scratch | 51.96% | 51.21% |
| E2 | ResNet-50 | Frozen backbone | 66.70% | 66.36% |
| A0 | ResNet-50 | Frozen, E3 protocol (control) | 61.71% | 60.88% |
| E3 | ResNet-50 | Full fine-tuning | **86.17%** | **86.10%** |

[Open the complete Assignment 2 page](assignment2)

[Download the Assignment 2 report PDF](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment2/report/CHOCO_A2_Report.pdf)

[Download the Assignment 2 slides](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment2/report/CHOCO_A2_Slides.pdf)

[Watch the presentation](https://youtu.be/xfPd0ZnpQYE)

[Source code and reproduction instructions](https://github.com/tapstoptap/deep-learning-261/tree/main/assignment2)

[Checkpoint reconstruction](https://github.com/tapstoptap/deep-learning-261/blob/main/assignment2/CHECKPOINTS.md)

The Assignment 2 dataset proposal is approved; the M2 draft and M3 final artifacts are recorded in the repository. The [presentation video](https://youtu.be/xfPd0ZnpQYE) is available on YouTube (unlisted).

---

# Assignment 3

**Task:** visual question answering on a VQA v2 subset.  
**Status:** M1 proposal submitted (18 Nov 2026); M2 draft due 25 Nov, M3 final due 02 Dec.

[Open the Assignment 3 page](assignment3)

---

## Assignment index

- [Assignment 1](assignment1)
- [Assignment 2](assignment2)
- [Assignment 3](assignment3)

## AI Usage Disclosure

Generative-AI use is disclosed on the landing page, each assignment page, each report, and in the repository log. The models and reported metrics were trained and verified by the group on Kaggle GPUs; AI assistance was limited to scaffolding, documentation, visualization scripts, and consistency checks. See [AI_USAGE.md](https://github.com/tapstoptap/deep-learning-261/blob/main/AI_USAGE.md) for per-entry prompts, affected files, verification, and the responsible member.

AI was also used in Assignment 3 M1 (proposal, EDA, split plan, compute estimate and Pages stub): Claude Opus for planning and drafting, Claude Sonnet via Claude Code for code, run under the direction and review of Nguyen Minh Phuc; see entries #10–#15 in [AI_USAGE.md](https://github.com/tapstoptap/deep-learning-261/blob/main/AI_USAGE.md).
