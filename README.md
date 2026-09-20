# CO3133 — Deep Learning and Its Applications

Course project for Semester 261 at Ho Chi Minh City University of Technology,
VNU-HCM, Faculty of Computer Science and Engineering.

## Group 99 — PTQ

| Name | Student ID | GitHub |
|---|---|---|
| Nguyen Minh Phuc | 2453005 | [tapstoptap](https://github.com/tapstoptap) |
| Nguyen Thanh The | 2453191 | [Themenli1](https://github.com/Themenli1) |
| Nguyen Minh Quang | 2453052 | [quangnguyencomeng](https://github.com/quangnguyencomeng) |

## Assignment status

| Assignment | Status | Milestone |
|---|---|---|
| Assignment 1 | Linear and MLP baseline complete | M1 — 23 Sep 2026 |
| Assignment 2 | Planned | Proposal — 7 Oct 2026 |
| Assignment 3 | Planned | Proposal — 18 Nov 2026 |

- [Assignment 1 report](docs/assignment1.md)
- [Assignment 1 experiment protocol](docs/assignment1-experiment-protocol.md)
- [Assignment 1 reproduction commands](assignment1/README.md)
- [AI usage log](AI_USAGE.md)
- [GitHub Pages](https://tapstoptap.github.io/deep-learning-261/)

## Assignment 1 milestone

The current benchmark uses Fashion-MNIST with a saved stratified split of
50,000 training and 10,000 validation images. Linear and MLP models were trained
with seeds `42`, `123`, and `2026`; the official 10,000-image test set was used
only after checkpoint selection.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Linear | 84.58 ± 0.10% | 84.48 ± 0.14% |
| MLP | **89.18 ± 0.21%** | **89.15 ± 0.22%** |

## Repository layout

```text
assignment1/
├── configs/          # shared protocol, model configs, saved split
├── src/              # data, models, training, evaluation and summaries
├── tests/            # unit tests
├── results/          # EDA and portable benchmark artifacts
└── README.md         # exact reproduction commands

docs/                 # GitHub Pages and reports
environment/          # benchmark environment
AI_USAGE.md           # AI-use disclosure
```

## Setup

```powershell
conda activate uav
python -m pip install -r requirements.txt
cd assignment1
python -m unittest discover -s tests -v
```

Dataset preparation, training, evaluation, and aggregation commands are kept in
[`assignment1/README.md`](assignment1/README.md).

## Reproducibility

- Fixed split manifest: `assignment1/configs/splits/fashion_mnist_train50000_val10000_seed42.json`.
- Shared protocol: `assignment1/configs/protocol.yaml`.
- Model configs: `assignment1/configs/models/`.
- Portable per-seed artifacts: `assignment1/results/benchmark/`.
- Hardware and software versions: `environment/README.md`.

Raw datasets, checkpoint binaries, and full local run directories are excluded
from Git.
