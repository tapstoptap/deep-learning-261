# Assignment 3 — M2 Draft: unimodal baselines and simple fusion for VQA on a VQA v2 subset

CO3133 Deep Learning and Its Applications · Semester 261 · Milestone M2 (due 25 Nov 2026)

Data, splits, answer vocabulary and metrics are as in the [M1 proposal](../PROPOSAL.md). Work in progress: improved fusion and ablations belong to M3.

## 1. Task and data

Visual question answering as classification over the K = 771 most frequent train answers. Split by image, seed 261.

| split | images | questions |
|---|---|---|
| train | 9,000 | 48,245 |
| val | 1,000 | 5,265 |
| test | 5,000 | 26,221 |

EDA: [M1 proposal Sec. 4](../PROPOSAL.md) and [figures](../eda/eda_outputs/figures/). Missing or unreadable images: 0; empty questions: 0.

## 2. Method

Pipeline: raw (image, question) pair → preprocessing → frozen encoders → cached features → fusion → MLP → soft-score BCE → AdamW → argmax answer → VQA accuracy.

| item | choice |
|---|---|
| Image encoder | ResNet-50, IMAGENET1K_V2, frozen; resize 256, center crop 224; pooled 2048-d feature |
| Text encoder | bert-base-uncased, frozen; max length 32; [CLS] 768-d feature |
| Alignment | each question is joined to its image by `image_id` from the official annotations |
| Normalisation | each modality feature is L2-normalised |
| Head (all trained models) | LayerNorm → Linear(→1024) → ReLU → Dropout 0.5 → Linear(→771) |
| Fusion stage and operation | intermediate (feature level); concatenation, 2816-d (model d) |
| Loss | binary cross-entropy against soft targets, target = official VQA accuracy of each vocabulary answer; summed over answers, averaged over the batch. One output head per model, so there are no separate per-modality losses |
| Training | AdamW, lr 1e-3, weight decay 1e-4, batch 256, 20 epochs, bf16 autocast; seeds 261, 262, 263 |
| Checkpoint rule | epoch with the highest val VQA accuracy |
| Pretraining / fine-tuning | pretrained encoders, no fine-tuning in M2 |
| Missing modality | none in this data, so no handling is needed |

The same hyperparameters are used for b, c and d. They were fixed before the first run and were not tuned.

| id | model | input |
|---|---|---|
| a | prior: most common train answer per official question_type (no training) | question type label |
| b | unimodal B | question only |
| c | unimodal A | image only |
| d | simple fusion (concat) | image and question |

## 3. Experimental design (fixed before training)

**Main metric:** official VQA accuracy (mean over the 10 leave-one-annotator-out subsets). Also exact match with the majority answer, and VQA accuracy per answer type. The oracle ceiling for K = 771 is reported beside the results.

| | H1 | H2 |
|---|---|---|
| Hypothesis | Text-only (b) is far above image-only (c), because of language priors, most on yes/no | Fusion (d) gains over text-only mostly on "other" (especially colour) and "number" |
| Changed factor | input modality | adding the image to the question |
| Fixed factors | splits, vocabulary, head, loss, optimiser, epochs, seeds | same |
| Metric | val VQA accuracy, overall and per answer type | val gain d − b per answer type and on "what color" questions |
| Supported if | b − c > 0 on every seed, and the yes/no gap is the largest of the three type gaps | on every seed the gains on "other" and "number" are positive and both larger than the yes/no gain |

A hypothesis with only one of its two conditions met is reported as partly supported. Val selects epochs; test is run once per final checkpoint and changes no choice.

## 4. Results

Full tables, per-seed gaps and raw values: [m2_results.md](../results/m2_results.md), [m2_results.json](../results/m2_results.json). Trained models: mean ± sample std over 3 seeds. All values in %.

### Validation (n = 5,265; oracle ceiling 88.54)

| model | VQA acc | exact match | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| a: prior (no training) | 31.88 | 25.36 | 62.90 | 26.55 | 9.22 | 22.56 |
| c: image only (unimodal A) | 25.55 ± 0.20 | 20.70 ± 0.32 | 65.22 ± 0.68 | 0.60 ± 0.07 | 1.52 ± 0.19 | 0.11 ± 0.15 |
| b: text only (unimodal B) | 39.19 ± 0.20 | 30.90 ± 0.31 | 66.06 ± 0.15 | 28.62 ± 1.48 | 21.19 ± 0.20 | 27.62 ± 0.79 |
| d: concat fusion | 43.88 ± 0.24 | 35.11 ± 0.38 | 66.71 ± 0.18 | 27.72 ± 0.53 | 30.56 ± 0.46 | 34.61 ± 0.66 |

### Test (n = 26,221; oracle ceiling 87.10)

| model | VQA acc | exact match | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| a: prior (no training) | 31.85 | 25.46 | 63.42 | 26.64 | 9.32 | 24.38 |
| c: image only (unimodal A) | 24.29 ± 0.47 | 19.14 ± 0.50 | 62.84 ± 1.30 | 0.41 ± 0.11 | 1.46 ± 0.16 | 0.46 ± 0.29 |
| b: text only (unimodal B) | 38.59 ± 0.10 | 30.79 ± 0.23 | 65.43 ± 0.15 | 28.70 ± 0.84 | 20.91 ± 0.36 | 32.07 ± 1.08 |
| d: concat fusion | 42.98 ± 0.22 | 34.79 ± 0.19 | 66.21 ± 0.34 | 28.04 ± 0.24 | 29.37 ± 0.40 | 38.12 ± 0.89 |

Test was run once, after the val results were in, on the best-val checkpoints. Nothing was changed afterwards. The test split is unfiltered, as set in M1.

## 5. Hypotheses

Judged on val with the criteria of Sec. 3; test is shown as confirmation.

**H1 (unimodal comparison): partly supported.**
- First condition met: text only beats image only on every seed (val +13.48 to +13.90 points; test +13.70 to +14.68).
- Second condition not met: the gap is not largest on yes/no (val -0.04 to +1.52). It is largest on number (val +26.46 to +29.54), followed by other (+19.63 to +19.72).
- The image-only model answers "yes" or "no" for 96.30% to 98.06% of val questions (three seeds). It does not see the question, so it cannot tell which kind of answer is asked for. This also explains why it scores below the prior, which is given the question type.

**H2 (adding the image to the question): partly supported.**
- Holds for "other": fusion gains on every seed (val +9.12 to +9.65; test +7.74 to +9.19), including "what color" questions (val +6.75 to +7.18).
- Does not hold for number: the change has no consistent sign (val -2.64 to +1.18; test -1.77 to +0.33).
- Yes/no barely moves (val +0.37 to +0.81).

## 6. Preliminary support and conflict cases

Validation, seed 261. "Right" means VQA accuracy 1, "wrong" means 0. Five questions of each kind were drawn at random (seed 261) from all that qualify; they are examples, not rates. Image ids are the question id without its last three digits. Images are not redistributed.

**Image supports the question: fusion right, text only wrong** (621 of 5,265 val questions)

| question_id | question | majority answer | text only | fusion |
|---|---|---|---|---|
| 193824000 | What sport is this? | soccer | tennis | soccer |
| 216757001 | What is the ground covered with? | snow | grass | snow |
| 270198003 | What kind of vehicle? | truck | motorcycle | truck |
| 330256003 | Are spectators present? | no | yes | no |
| 334259002 | How many cranes? | 3 | 1 | 3 |

**Conflict: text only right, fusion wrong** (405 of 5,265 val questions)

| question_id | question | majority answer | text only | fusion |
|---|---|---|---|---|
| 220903005 | How many panels of fencing is the player covering? | 2 | 2 | 1 |
| 242541000 | What color is the water? | blue | blue | brown |
| 290911002 | How many wheels are visible? | 3 | 3 | 4 |
| 348573002 | How many vehicles are in the background? | 1 | 1 | 4 |
| 355668002 | Are there any tall skyscrapers in this picture? | no | no | yes |

Fusion adds correct answers more often than it removes them (621 against 405). Why it fails on particular questions has not been analysed yet.

## 7. Compute cost

Hardware: NVIDIA RTX 4050 Laptop (6 GB), Intel i5-12450HX. Software: Python 3.14.4, torch 2.15.0 + CUDA 13.2, torchvision 0.29.2, transformers 5.23.0.

- Feature caching, once: 15,000 images and 79,731 questions (question pass 31 s in this run; images took 21 s in the M1 benchmark). Feature files take about 177 MB.
- Head training:

| model | parameters | s/epoch | train s, 3 seeds | best epoch (261, 262, 263) |
|---|---|---|---|---|
| c: image only (unimodal A) | 2,892,547 | 0.36 | 21.8 | 20, 4, 16 |
| b: text only (unimodal B) | 1,579,267 | 0.27 | 16.5 | 19, 18, 20 |
| d: concat fusion | 3,680,515 | 0.45 | 27.2 | 20, 17, 9 |

Training times cover the optimisation steps only, without validation passes or checkpoint writing.

## 8. Limitations

- Encoders are frozen, and the image is reduced to one pooled vector with no spatial information.
- One fixed set of hyperparameters; nothing was tuned. Some runs reached their best val score at the last epoch, so 20 epochs may be short for them.
- The prior uses the official question_type annotation, which the other models are not given.
- Answers outside the 771-answer vocabulary cannot be predicted (see the oracle ceilings).
- Small std over 3 seeds measures training randomness only, not sampling error of the split.

## 9. Reproduce

Run from `assignment3/`, with the VQA v2 files and images in `eda/data/vqa_v2/` (see `eda/PHASE3_GUIDE.md`):

```bash
python src/extract_features.py
python src/models/prior.py
python src/train.py --config configs/unimodal_a.yaml
python src/train.py --config configs/unimodal_b.yaml
python src/train.py --config configs/fusion_simple.yaml
python src/evaluate.py --split val
python src/evaluate.py --split test
python src/report_tables.py
```

Checks made: the val oracle (88.54) and train coverage (86.59) reproduce the M1 values; the prior's val accuracy (31.88) agrees between two separate computations; retraining model d with seed 261 gave identical weights.

## 10. Remaining work for M3

- Improved fusion (model e): question-guided attention over the ResNet 7×7 grid.
- Ablations: concat against attention; soft-score BCE against hard cross-entropy (H4); K = 771 against K = 1000; fine-tuning the top 2 BERT layers; zeroing one modality at test time.
- Error analysis and support/conflict cases in depth, final report, slides, video and checkpoint links.
