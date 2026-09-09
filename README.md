# CO3133 — Deep Learning and Its Applications (Semester 261)
Repository status: initial course-project scaffold. The GitHub Pages structure is available, while training, evaluation, and model implementations are still under development. No experimental results are reported yet.
## Group 99 — PTQ

Repository for the course project (Assignments 1–3), HCMUT – VNU-HCM,
Faculty of Computer Science and Engineering. Instructor: Lê Thành Sách.

| Assignment | Current Status | Next Milestone |
|---|---|---|
| Assignment 1 | In progress | M1 Draft - 23 Sep 2026 |
| Assignment 2 | Planned; proposal not yet approved | Proposal — 7 Oct 2026 |
| Assignment 3 | Planned; proposal not yet approved | Proposal — 18 Nov 2026 |

- **Landing page (GitHub Pages):** https://tapstoptap.github.io/deep-learning-261/
- **Assignment 1:** [docs/assignment1.md](docs/assignment1.md) — Foundations of DL Pipelines and Architectures
- **Assignment 2:** [docs/assignment2.md](docs/assignment2.md) — Large-Scale Data / Specialized Task
- **Assignment 3:** [docs/assignment3.md](docs/assignment3.md) — Multimodal Deep Learning

## Repository Layout

```
.
├── README.md                 <- this file
├── AI_USAGE.md                <- detailed AI-use log (all assignments)
├── requirements.txt
├── docs/                      <- GitHub Pages source (landing + assignment pages)
│   ├── index.md
│   ├── assignment1.md
│   ├── assignment2.md
│   └── assignment3.md
├── assignment1/
│   ├── README.md
│   ├── data/                  <- raw/processed data (or download scripts only; do not commit large files)
│   ├── notebooks/             <- exploratory notebooks (EDA, debugging, still planning)
│   ├── src/
│   │   ├── datasets/          <- Dataset / DataLoader code
│   │   ├── models/             <- linear, mlp, cnn, rnn, transformer
│   │   ├── utils/              <- metrics, seed, misc
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── configs/                <- one YAML per model/experiment
│   ├── checkpoints/             <- saved weights or download instructions
│   ├── results/                 <- logs, curves, confusion matrices
│   ├── report/                  <- report source (md/tex) + exported PDF , added later
│   └── slides/                  <- presentation slides , added later
├── assignment2/
│   ├── proposal/dataset_proposal.md
│   └── ... (same layout as assignment1, plus proposal/)
└── assignment3/
    ├── proposal/dataset_proposal.md
    ├── src/fusion/               <- early/late/intermediate fusion modules
    └── ... (same layout as assignment2)
```

## Prerequisites

- Git
- Python `<tested version>`
- PyTorch-compatible CPU or GPU environment

## Clone
```bash
git clone https://github.com/tapstoptap/deep-learning-261.git
cd deep-learning-261
```

## Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Compatibility Table
| Environment | Status |
| --- | --- |
| Windows 11 + PowerShell + Python X.Y | - |
| Ubuntu + Python X.Y | - |
| CUDA X.Y + GPU model | - |
| CPU-only | - |

## Supported environments

- Primary benchmark environment: Ubuntu `22.04.5 LTS`, NVIDIA RTX 4090 24 GB
- Local development: Windows 11 with PowerShell
- Main reported comparisons are run on the same Ubuntu RTX 4090 machine.

## Dataset Preparation

See each assignment's `README.md` (e.g. `assignment1/README.md`) for
dataset download/preparation commands.

## Train / Evaluate

Each assignment folder is self-contained:

```bash
cd assignment1
python src/train.py --config configs/cnn.yaml
python src/evaluate.py --config configs/cnn.yaml --checkpoint checkpoints/cnn_best.pt
```

## Reproducibility

| Item | Location |
|---|---|
| Seeds | set in each `configs/*.yaml` |
| Dependency versions | `requirements.txt` (pin exact versions before final submission) |
| Hardware used | documented in each assignment's report, Training Setup section |
| Checkpoints | `assignmentX/checkpoints/` or linked download + SHA256 in `assignmentX/README.md` |
| Config → result mapping | each result in the report cites its config file name + commit hash |

Main reported results must be traceable to: model configuration, dataset split,
checkpoint, log/experiment ID, and the corresponding commit/tag.

## Training Logging
WandB...

## AI Usage Disclosure

Summary on the [landing page](docs/index.md#ai-usage-disclosure) and per-assignment
pages; full log in [AI_USAGE.md](AI_USAGE.md).

## Group Members

| Name | Student ID | Role | GitHub |
|---|---|---|---|
| Nguyen Minh Phuc | 2453005 | Ideas & Techies  | [tapstoptap](https://github.com/tapstoptap) |
| Nguyen Thanh The | 2453191 | Ideas & Techies | [Themenli1](https://github.com/Themenli1) |
| Nguyen Minh Quang | 2453052 | Ideas & Techies | [quangnguyencomeng](https://github.com/quangnguyencomeng) |
