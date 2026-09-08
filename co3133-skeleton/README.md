# CO3133 — Deep Learning and Its Applications (Semester 261)
## Group [ID] — [Group Name]

Repository for the course project (Assignments 1–3), HCMUT – VNU-HCM,
Faculty of Computer Science and Engineering. Instructor: Lê Thành Sách.

- **Landing page (GitHub Pages):** https://<github-username>.github.io/<repo-name>/
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
│   ├── notebooks/             <- exploratory notebooks (EDA, debugging)
│   ├── src/
│   │   ├── datasets/          <- Dataset / DataLoader code
│   │   ├── models/             <- linear, mlp, cnn, rnn, transformer
│   │   ├── utils/              <- metrics, seed, misc
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── configs/                <- one YAML per model/experiment
│   ├── checkpoints/             <- saved weights or download instructions
│   ├── results/                 <- logs, curves, confusion matrices
│   ├── report/                  <- report source (md/tex) + exported PDF
│   └── slides/                  <- presentation slides
├── assignment2/
│   ├── proposal/dataset_proposal.md
│   └── ... (same layout as assignment1, plus proposal/)
└── assignment3/
    ├── proposal/dataset_proposal.md
    ├── src/fusion/               <- early/late/intermediate fusion modules
    └── ... (same layout as assignment2)
```

## Installation

```bash
git clone <repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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

## AI Usage Disclosure

Summary on the [landing page](docs/index.md#ai-usage-disclosure) and per-assignment
pages; full log in [AI_USAGE.md](AI_USAGE.md).

## Group Members

| Name | Student ID | Role | GitHub |
|---|---|---|---|
| ... | ... | ... | ... |
| ... | ... | ... | ... |
| ... | ... | ... | ... |
