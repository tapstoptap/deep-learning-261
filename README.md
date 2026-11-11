# CO3133 — Deep Learning and Its Applications

Course project for Semester 261 at Ho Chi Minh City University of Technology,
VNU-HCM, Faculty of Computer Science and Engineering.

## Group 99 — PTQ (choco)

| Name | Student ID | GitHub |
|---|---|---|
| Nguyen Minh Phuc | 2453005 | [tapstoptap](https://github.com/tapstoptap) |
| Nguyen Thanh The | 2453191 | [Themenli1](https://github.com/Themenli1) |
| Nguyen Minh Quang | 2453052 | [quangnguyencomeng](https://github.com/quangnguyencomeng) |

For Assignment 1, Nguyen Minh Quang (2453052) is the sole author and
contributor. Nguyen Minh Phuc and Nguyen Thanh The did not contribute to
Assignment 1.

## Assignment status

| Assignment | Status | Milestone |
|---|---|---|
| Assignment 1 | Five model families benchmarked | 20 Oct 2026 |
| Assignment 2 | Food-101: three adaptation regimes plus a matched frozen control | M3 Final — 11 Nov 2026 |
| Assignment 3 | Planned | Proposal — 18 Nov 2026 |

- [Assignment 1 report](docs/assignment1.md)
- [Assignment 1 experiment protocol](docs/assignment1-experiment-protocol.md)
- [Assignment 1 reproduction commands](assignment1/README.md)
- [Assignment 2 report](docs/assignment2.md)
- [Assignment 2 reproduction and checkpoints](assignment2/CHECKPOINTS.md)
- [AI usage log](AI_USAGE.md)
- [GitHub Pages](https://tapstoptap.github.io/deep-learning-261/)

## Assignment 1 milestone

The benchmark uses Fashion-MNIST with a saved stratified split of 50,000
training and 10,000 validation images. Linear, MLP, CNN, GRU, and Transformer
models were trained with seeds `42`, `123`, and `2026`; the official
10,000-image test set was used after checkpoint selection.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Linear | 84.31 ± 0.09% | 84.23 ± 0.06% |
| MLP | 89.37 ± 0.11% | 89.31 ± 0.12% |
| CNN | **92.22 ± 0.39%** | **92.20 ± 0.41%** |
| GRU | 90.29 ± 0.19% | 90.26 ± 0.21% |
| Transformer | 89.40 ± 0.09% | 89.37 ± 0.13% |

## Assignment 2 milestone

The benchmark uses Food-101 (101 classes, 101,000 images) with a stratified
90/10 hold-out of the official training split (68,175 training / 7,575
validation images) and the untouched 25,250-image official test set. Three
adaptation regimes plus a matched frozen control were trained with seed `42`;
checkpoints were selected by best validation macro-F1 and evaluated once on the
test set.

| Run | Model | Adaptation | Test accuracy | Test macro-F1 |
|---|---|---|---:|---:|
| E1 | Simple CNN | Scratch | 51.96% | 51.21% |
| E2 | ResNet-50 | Frozen backbone | 66.70% | 66.36% |
| A0 | ResNet-50 | Frozen, E3 protocol (control) | 61.71% | 60.88% |
| E3 | ResNet-50 | Full fine-tuning | **86.17%** | **86.10%** |

Under the matched protocol (A0 vs E3), the observed gap is 24.46 points; the
protocol switch coincides with a 4.99-point drop for the frozen model. Reproduction commands
and expected metrics are in [`assignment2/README.md`](assignment2/README.md)
and [`assignment2/CHECKPOINTS.md`](assignment2/CHECKPOINTS.md); the
[presentation video](https://youtu.be/xfPd0ZnpQYE) is on YouTube (unlisted).

## Repository layout

```text
assignment1/
├── configs/          # shared protocol, model configs, saved split
├── src/              # data, models, training, evaluation and summaries
├── tests/            # unit tests
├── results/          # EDA and portable benchmark artifacts
└── README.md         # exact reproduction commands

assignment2/
├── configs/          # baseline, frozen, fine-tune and ablation configs
├── proposal/         # approved Food-101 dataset proposal
├── src/              # data, models, training, evaluation, EDA, benchmarks
├── results/          # histories, metrics, curves, EDA artifacts
├── CHECKPOINTS.md    # reconstruction contract and expected metrics
└── README.md         # A2 overview and commands

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
