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

