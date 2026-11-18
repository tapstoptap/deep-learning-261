# Assignment 3 — M1 Dataset Proposal: Visual Question Answering on a VQA v2 subset

CO3133 Deep Learning and Its Applications · Semester 261 · Milestone M1 (due 18 Nov 2026)

All numbers below come from scripts in this repo: `assignment3/eda/` (Phase 3), `assignment3/splits/` (Phase 4), `assignment3/compute/` (Phase 6). Seed 261 everywhere.

## 1. Dataset name, source, version, license

| field | value |
|---|---|
| Dataset | VQA v2 (balanced real images), questions and annotations |
| Images | MS COCO 2014 (train2014, val2014) |
| Source | https://visualqa.org/download.html, https://cocodataset.org/ |
| Version | VQA v2.0 (`v2_OpenEnded_mscoco_*2014_questions.json`, `v2_mscoco_*2014_annotations.json`) |
| License | VQA annotations: CC BY 4.0. COCO images: Flickr Terms of Use (research use, no redistribution of images; we redistribute only ids and derived statistics) |

Fallback dataset (Phase 2): VizWiz-VQA.

## 2. Task, modalities, input, output

- **Task:** visual question answering, framed as **answer classification** over a fixed vocabulary of the K = 771 most frequent train answers.
- **Modality A (image):** an RGB COCO photo.
- **Modality B (text):** an English question about that photo.
- **Input:** one (image, question) pair. **Output:** one answer class out of 771.
- **Pair validity:** every question was written by annotators *for that specific image*, so the two modalities are linked by the same scene (handbook Sec. 28.1). No pairs are constructed by us.
- Questions whose answer is outside the vocabulary stay in train/val/test and simply count as unreachable (this is reflected in the oracle accuracy below).

## 3. Number of paired samples and annotation types

- Full VQA v2: 443,757 questions on 82,783 train2014 images; 214,354 questions on 40,504 val2014 images (our loader reproduces these counts exactly).
- **Our subset (subset rules in Sec. 7):** 15,000 images, 79,743 question–image pairs as sampled, 79,731 after removing 12 train/val questions (Sec. 5). This is ~8× the handbook's VQA threshold of ~10,000 pairs.
- **Annotations per question:** 10 free-form human answers, each with a confidence flag; one official `answer_type` (yes/no, number, other); one official `question_type` (65 categories by question prefix); a precomputed majority answer.

## 4. Preliminary pair-quality and distribution analysis (Phase 3)

- **Missing modality:** 0 missing or unreadable images; 0 empty questions.
- **Manual pair audit:** 100 random pairs checked by hand: 98/100 the question is answerable from the image and the majority answer is correct (95% Wilson CI 93.0–99.4%). The 2 failures are genuine VQA annotation errors (e.g. qid 82157001, "checkers"), not loading bugs.
- **Duplicates:** 2 real near-duplicate image pairs (perceptual hash, confirmed by eye; both in the train pool) and 6 exact duplicate (image, question) pairs per subset. One flagged pair (406749/412172) was a false alarm on inspection.
- **Questions per image:** mean ≈ 5.3 in every split; max 176 (train image 265622, 0.36% of train). No cap applied.
- **Question length:** with bert-base-uncased incl. [CLS]/[SEP], train p95 = 13, p99 = 16, max = 27 tokens → max_len = 32.
- **Answer vocabulary:** 5,335 distinct majority answers in train; heavy-tailed (see K table in Sec. 5).
- Answer-type and question-type distributions, image-size statistics and example pairs: figures F1–F9 in `assignment3/eda/` (see `PHASE3_LOG.md`).

## 5. Split plan and leakage-prevention unit (Phase 4)

- **Split unit: the image.** All questions about one image stay in one split, so a model cannot see the same scene in train and test. The 2 near-duplicate image pairs are grouped and kept in the same split.
- **Test:** 5,000 images sampled from val2014 (the official test answers are not public). Test is never used for any decision, including K.
- **Train/val:** 10,000 images sampled from train2014, split by image into 9,000 train / 1,000 val (seed 261).
- **Cleaning (train/val only):** removed 6 questions with an empty majority answer and 6 exact duplicate (image, question) pairs with the same majority answer (kept the lowest question_id). Test is left as sampled; its 6 empty-majority questions and 6 duplicate pairs are reported, not removed.

| split | images | questions (raw) | removed | questions (final) |
|---|---|---|---|---|
| train | 9,000 | 48,255 | 10 | 48,245 |
| val | 1,000 | 5,267 | 2 | 5,265 |
| test | 5,000 | 26,221 | 0 | 26,221 |

- **Leakage checks (all PASS):** image ids disjoint across splits; question ids disjoint; no near-duplicate group crosses splits; answer vocabulary built from train rows only.
- **Balance check:** total variation distance (TVD) between train and val, compared with 1,000 random image-level splits of the same pool:

| measure | seed 261 | null mean | null p95 | seed-261 percentile |
|---|---|---|---|---|
| top-20 question_type (+other) | 0.0229 | 0.0241 | 0.0320 | 41.8 |
| answer_type | 0.0064 | 0.0083 | 0.0169 | 37.6 |

  *Deviation from plan:* the question_type check was first a fixed TVD ≤ 0.02, an untested guess. Seed 261 gave 0.0229, which is below the random-split average, so the limit was replaced by "≤ p95 of random splits". The split itself was not changed and no other seed was tried.

- **Choosing K (train vocab, val only).** Rule fixed before seeing results: the largest K whose K-th answer has ≥ 5 train examples.

| K | val oracle VQA acc % | train coverage % | val coverage % | K-th answer (train count) |
|---|---|---|---|---|
| 500 | 86.26 | 83.33 | 83.36 | checkered (7) |
| **771 (chosen)** | **88.54** | **86.59** | **85.93** | yellow and blue (5) |
| 1000 | 89.54 | 88.41 | 87.24 | animals (3) |
| 2000 | 91.65 | 93.09 | 90.50 | "1 6" (1) |
| 3000 | 92.17 | 95.16 | 91.23 | document (1) |

  Oracle accuracy = VQA accuracy of a model that always picks the best in-vocabulary answer, i.e. the ceiling for any classifier with that K.

## 6. Metrics, baselines, planned models (Phase 5)

**Metrics.**
- Main: official VQA accuracy, min(#annotators giving the answer / 3, 1), averaged over the 10 choose-9 annotator subsets.
- Also: top-1 exact match against the majority answer, and VQA accuracy per answer type (yes/no, number, other).
- Val selects models and epochs; test is evaluated once per final configuration.

**Models** (all heads output 771 classes):

| id | model | modalities | role |
|---|---|---|---|
| a | Prior: most common train answer per question_type | question type only | sanity floor |
| b | Frozen BERT-base [CLS] → MLP | text only | unimodal B |
| c | Frozen ResNet-50 pooled feature → MLP | image only | unimodal A |
| d | Concat(image, text) → MLP | both | simple fusion (intermediate) |
| e | Question-guided attention over the ResNet 7×7 grid, element-wise product with the question vector → MLP | both | improved fusion |

Encoders are frozen and their features cached once; heads are trained with soft-score BCE (targets = per-answer VQA scores), AdamW, 20 epochs, 3 seeds for a–e.

**Ablations.**
1. Fusion: concat (d) vs attention (e).
2. Loss: soft-score BCE vs hard cross-entropy on the majority answer.
3. Vocabulary: K = 771 vs K = 1000.
4. Encoder: fully frozen vs fine-tuning the top 2 BERT layers (on model e).
5. Modality contribution: zero out one modality at test time for d and e.

**Hypotheses.**
- H1: text-only (b) is far above image-only (c), because of language priors, most on yes/no.
- H2: fusion gains over text-only mostly on "other" (especially colour) and "number" questions.
- H3: attention fusion (e) beats concat (d).
- H4: soft-score BCE beats hard cross-entropy.

Each experiment will report hypothesis, changed factor, fixed factors, metric and conclusion criterion (handbook Sec. 30). If fusion does not beat text-only, we will explain why with per-type results and examples.

## 7. Subset selection rules

- Images, not questions, are sampled: 10,000 train2014 images and 5,000 val2014 images, uniformly at random with seed 261. Every question of a sampled image is kept.
- Reason: a laptop GPU (6 GB) and a 3-week window. The subset is still ~8× the size threshold.
- Cleaning and K rules as in Sec. 5. All id lists are committed with sha256 hashes in `assignment3/splits/split_manifest.json`, so the subset is exactly reproducible.

## 8. Compute estimate (Phase 6, measured)

**Hardware/software:** NVIDIA RTX 4050 Laptop (6 GB), Intel i5-12450HX, torch 2.15.0 + CUDA 13.2, torchvision 0.29.2, transformers 5.23.0, Python 3.14.

| item | measured | for our data |
|---|---|---|
| ResNet-50 features (pooled + 7×7 grid, fp16) | 713 img/s | 15,000 images in 21 s |
| BERT-base features, max_len 32 | 2,576 q/s | 79,731 questions in 31 s |
| Concat MLP head | 0.33 s/epoch | 20 epochs in 7 s |
| Attention head over 7×7 grid | 3.91 s/epoch | 20 epochs in 78 s |
| Fine-tune BERT top 2 layers | 0.038 s/step, 1.2 GB VRAM | 28.6 s/epoch |

**Storage:** pooled image features 0.06 GB; 7×7 grid features 3.0 GB for all 15,000 images (stored once per image and indexed by image id); [CLS] features 0.12 GB.

**Total:** 5 models × 3 seeds × 20 epochs plus ablations ≈ 0.06–0.56 GPU-hours, ≈ 1 GPU-hour if fine-tuning is included; under 2 GPU-hours with a 2× margin. Everything runs on our own laptop; no cloud GPU is needed.

## 9. Risks

- **Language priors:** VQA is known to be answerable from the question alone in many cases; H1 and ablation 5 measure this directly instead of hiding it.
- **Annotation noise:** ~2% of pairs have a wrong answer (audit); the soft-score target already softens this.
- **Unreachable answers:** 14% of val answers are outside K = 771; the oracle ceiling of 88.54% is reported next to every result.

## 10. AI use

AI assistance (Claude) was used for planning, code generation and review; details in `AI_USAGE.md`.
