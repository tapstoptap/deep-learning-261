# Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks

**Status:** Completed — M3 Final (11 Nov 2026).

## Task track
- **Track:** Multi-class Image Classification
- **Dataset:** Food-101 (101 food categories, 101,000 images)

## Dataset proposal
See [`proposal/dataset_proposal.md`](proposal/dataset_proposal.md).  
- **Approval status:** Approved (course dataset-proposal workflow; group record).
- **Approval record:** maintained with the group's course-workflow records.

## Deliverables
- [Report PDF](report/CHOCO_A2_Report.pdf)
- [Slides PDF](report/CHOCO_A2_Slides.pdf)
- [Checkpoint and reconstruction instructions](CHECKPOINTS.md)
- [Interactive web demo (Streamlit)](demo/README.md)
- [Assignment page](../docs/assignment2.md)

The presentation video is recorded from the slides and published on YouTube (unlisted): <https://youtu.be/xfPd0ZnpQYE>.

### Report sources
Only the exported PDFs are tracked, as in Assignment 1. The LaTeX sources are kept in the local working copy and are not part of the repository, so there is nothing to rebuild from a fresh clone.

## Interactive demo
[`demo/`](demo/README.md) is a Streamlit web app that loads the E3 (full fine-tune) and E2 (frozen backbone) checkpoints side by side: upload a food photo or pick a built-in Food-101 example, and each model shows its top-6 class probabilities on CPU. It reuses the exact evaluation transform and the same `.pt` checkpoints, so its outputs match the recorded test behaviour. Live at <https://deep-learning-261-ahricdx9pw4yfwaqmrmx7e.streamlit.app/>. Run locally with `streamlit run demo/streamlit_app.py`; deployment notes (Streamlit Cloud + a free Hugging Face checkpoint host) are in [`demo/README.md`](demo/README.md).

## Setup
```bash
pip install -r ../requirements.txt
```

## Data preparation
To download and extract the official Food-101 dataset via torchvision:
```bash
python src/datasets/food101.py --download --out data/
```
The data will be organized in:
```text
data/
└── food-101/
    ├── images/
    │   ├── apple_pie/
    │   ├── baby_back_ribs/
    │   └── ...
    └── meta/
        ├── classes.txt
        ├── train.txt
        └── test.txt
```

## Training

### Baseline — Simple CNN trained from scratch (E1)
```bash
python src/train.py --config configs/baseline.yaml
```

### Pretrained ResNet-50 — Feature Extractor with Frozen Backbone (E2)
```bash
python src/train.py --config configs/pretrained_frozen.yaml
```

### Pretrained ResNet-50 — Full Fine-Tuning (E3)
```bash
python src/train.py --config configs/pretrained.yaml
```

### Quick Smoke Test / Verification (1 epoch, few batches)
```bash
python src/train.py --config configs/baseline.yaml --smoke-test
```
Smoke runs write to `checkpoints/*_smoke/` and `results/*_smoke/` (unless `--checkpoint-dir` / `--result-dir` are passed), so they cannot overwrite the official checkpoints or artifacts.

## Evaluation
Run evaluation on the test set:
```bash
python src/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/simple_cnn_best.pt --split test
python src/evaluate.py --config configs/pretrained.yaml --checkpoint checkpoints/resnet50_finetune/resnet50_best.pt --split test
```

## Checkpoints & Experiments Summary
| Model | Config | Status | Checkpoint | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---|---|---:|---:|---:|
| Simple CNN (Scratch) | `configs/baseline.yaml` | Completed (20 ep, 163m) | `checkpoints/baseline/simple_cnn_best.pt` | 46.38% | 51.96% | 51.21% |
| ResNet-50 (Frozen Backbone) | `configs/pretrained_frozen.yaml` | Completed (15 ep, 150m) | `checkpoints/resnet50_frozen/resnet50_best.pt` | 61.55% | 66.70% | 66.36% |
| ResNet-50 (Frozen, E3 protocol) | `configs/ablation_frozen_e3proto.yaml` | Completed (15 ep, 108m) | `checkpoints/ablation_frozen_e3proto/resnet50_best.pt` | 56.61% | 61.71% | 60.88% |
| ResNet-50 (Full Fine-tuned) | `configs/pretrained.yaml` | Completed (15 ep, 269m) | `checkpoints/resnet50_finetune/resnet50_best.pt` | **81.45%** | **86.17%** | **86.10%** |

Checkpoint binaries are excluded from Git. The table records the paths expected by the evaluation commands; [`CHECKPOINTS.md`](CHECKPOINTS.md) gives the exact reconstruction procedure and the expected metrics for each run. Checkpoint selection is by **best validation macro-F1**.

### Strict frozen-vs-unfrozen ablation (A0)
The reported E2/E3 comparison also changes optimizer, learning rate, batch size, and GPU. A0 is the matched control: the frozen backbone trained with the E3 protocol, so A0 and E3 differ only in `freeze_backbone`.

```bash
python src/train.py --config configs/ablation_frozen_e3proto.yaml
python src/evaluate.py --config configs/ablation_frozen_e3proto.yaml   --checkpoint checkpoints/ablation_frozen_e3proto/resnet50_best.pt --split test
```

Under the matched protocol, E3 exceeds A0 by +24.46 accuracy points (86.17% vs. 61.71%) in the two recorded runs; the protocol switch coincides with a 4.99-point drop for the frozen model.

## Environment
- Training: Kaggle Tesla T4 (E1/E2), Tesla P100 (E3), and Colab for A0 (GPU model not logged), PyTorch 2.x + torchvision.
- Analysis/plots: Python 3.12.5, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu.
- Seeds: 42 for data splitting, model initialization, and training.
- The instrumented Kaggle session (benchmark + qualitative grid) reported torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / CUDA 12.8 / cuDNN 91900 / Python 3.13.15 on a Tesla T4. The builds used for E1–E3 were not logged; capture them with `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"` when reconstructing the runs.
