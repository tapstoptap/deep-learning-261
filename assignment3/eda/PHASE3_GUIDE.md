# Phase 3: Download and preliminary EDA (VQA v2)

**Goal:** produce every number and figure that the A3 Dataset Proposal and the report's
Part 1 need about the data, from a script anyone in the group can re-run.

Handbook items this phase covers:
- Sec. 27: "number of paired samples and annotation types" and "preliminary pair-quality and distribution analysis".
- Sec. 31: multimodal EDA must report pair counts, missing-modality rates, text length and pair quality.
- Sec. 28.1: check missing, noisy and mismatched pairs.
- Sec. 1.5: no vague claims; every statement needs a number.

The folder contains:

| File | What it is |
|---|---|
| `vqa_eda.py` | The EDA script: statistics, subset, image checks, figures, proposal tables |
| `audit.py` | The manual pair-quality audit: an HTML viewer, plus a summary with 95% confidence intervals |
| `requirements-eda.txt` | Package versions the scripts were tested with |
| `PHASE3_GUIDE.md` | This file |

---

## 3.1 Set up

1. Put the folder in the group repo at `a3/eda/`.
2. Create a separate environment:
   `python -m venv .venv`, activate it, then `pip install -r requirements-eda.txt`.
3. Optional extras (tokenizer lengths and CLIP similarity): `pip install torch transformers`.
4. Add `data/` and `eda_outputs/` to `.gitignore`. The images and JSON files are large and must not be committed.
   Commit only the scripts, `subset/*.txt`, `subset/*_question_ids.csv`, `summary.json`,
   `proposal_tables.md`, `figures/` and `audit_*.{csv,json}`.

## 3.2 Download the annotations

From https://visualqa.org/download.html, section **Balanced Real Images**, download:
- Training questions 2017 v2.0 and Validation questions 2017 v2.0
- Training annotations 2017 v2.0 and Validation annotations 2017 v2.0

Unzip them into `data/vqa_v2/` so you have:

```
data/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json
data/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json
data/vqa_v2/v2_mscoco_train2014_annotations.json
data/vqa_v2/v2_mscoco_val2014_annotations.json
```

**Record for the proposal:**
- the download date
- the page's version labels ("2017 v2.0")
- the `info` and `license` blocks the script copies into `summary.json → meta.json_headers`
- the sha256 checksums in `summary.json → meta.input_files`. These prove exactly which files you used.

You do **not** need the test2015 questions. Test answers are hidden, so the official val2014 set becomes your test set.

## 3.3 Run 1: annotations only (minutes)

```
python vqa_eda.py --data-dir data/vqa_v2 --out-dir eda_outputs
```

Before going further, sanity-check the full-split counts in `proposal_tables.md`, Table 1, against the
official numbers:
- train2014: 82,783 images and 443,757 questions
- val2014: 40,504 images and 214,354 questions

If they differ, you have the wrong or an old file. Also confirm `full_image_overlap_train_val` = 0.

## 3.4 Run 2: images, tokenizer and CLIP (the full EDA)

```
python vqa_eda.py --data-dir data/vqa_v2 --out-dir eda_outputs --images --workers 16 ^
    --tokenizer bert-base-uncased --clip openai/clip-vit-base-patch32
```

- The `^` continues the line in PowerShell/cmd. Use `\` on Linux.
- `--images` downloads only the subset's images, roughly 15k files of about 160 kB each, from images.cocodataset.org.
  Re-running resumes: images that already exist are skipped. If you already have the COCO `train2014/` and
  `val2014/` folders, put them under `data/vqa_v2/images/` and add `--no-download`.
- `--tokenizer`: use the text encoder you expect to use in Phase 5. `bert-base-uncased` is a safe placeholder.
  This gives the question length in tokens, which sets your max sequence length.
- `--clip` runs on a GPU if one is available, otherwise on the CPU (slower, but fine for about 80k pairs if you wait).
- Keep `--seed 261`, `--n-train-images 10000` and `--n-test-images 5000` unless you deliberately change the
  subset. If you change them, re-run everything, because all numbers depend on them.

## 3.5 Manual pair-quality audit (do-er judgment)

1. Build the viewer:
   `python audit.py html --audit eda_outputs/audit_sample.csv --data-dir data/vqa_v2 --out eda_outputs/audit_sheet.html`
2. Open `audit_sheet.html` in a browser. For each of the 100 cards, fill the `verdict` column of
   `audit_sample.csv` in Excel or LibreOffice. Row *i* is card *#i*.
   - **matched**: the question is clearly about this image and the majority answer is correct.
   - **weak**: the question is related to the image but ambiguous or subjective ("Is this a nice day?"), or the majority answer is debatable.
   - **mismatched**: the question doesn't fit the image, or the majority answer is wrong.

   Write the reason in `note` (e.g. "subjective", "answer wrong: 3 not 2", "object not visible").
3. Summarize:
   `python audit.py summary --audit eda_outputs/audit_sample.csv --out eda_outputs/audit_summary.json`

   This gives counts, %, and a Wilson 95% interval. With n = 100, the interval is roughly ±8–10 points, so report it.
4. Rules for the audit:
   - Don't change verdicts after seeing the summary.
   - Don't re-sample to get a better number.
   - If a teammate audits too, report both people's counts and how often you agreed.

## 3.6 Look at the outputs critically

Open each figure and check it before using it. In particular:
- `near_dups_*.csv`: open the flagged image pairs. Perceptual hashes can produce false alarms (e.g. two
  plain snowy scenes). For each pair, mark it as a real duplicate or a false alarm, and report how many are real.
- `F11_lowest_clip_pairs.png`: low CLIP similarity often just means an abstract question ("Is it daytime?").
  Use it to find examples, not as a mismatch rate.
- If an image is missing or corrupt, re-run with `--images` once to retry the download, then report what's left
  as the missing-modality rate.

---

## 3.7 What each output means and where it goes in the proposal

### Counts and structure
| Output | Meaning | Use in proposal |
|---|---|---|
| Table 1 (full splits) | Official size of VQA v2 train/val | Dataset description, "number of paired samples" |
| Table 2 (subset) | Your actual data: images, questions, answer-type % | Number of paired samples, subset rule |
| `questions_per_image` | Each image has several questions, which is why the split unit is the image | Phase 4 leakage argument |
| `answer_type_pct`, F1 | yes/no vs number vs other mix | Distribution analysis; later, per-type results |
| `question_types`, F4 | Which question templates dominate | Distribution analysis; language-prior discussion |

### Pair quality (handbook: "pair quality", "noisy/mismatched pairs")
| Output | Meaning | Use |
|---|---|---|
| `pct_top_answer_>=3_votes` | Share of questions where at least 3 annotators agree. These are the only questions a model can score 100% VQA accuracy on. | Main quantitative pair-quality number |
| `pct_all_10_agree`, mean votes, by answer type, F6 | How subjective or ambiguous questions are, per type | Pair-quality analysis; explains why VQA accuracy uses soft scores |
| `answer_confidence_pct` | Annotators' own "yes / maybe / no" confidence | Supporting pair-quality evidence |
| `audit_summary.json` | Your human check of 100 random pairs, with a 95% CI | The "checked mismatched pairs" requirement (Sec. 28.1) |
| F9 low-agreement examples | What noisy pairs look like | Qualitative figure for the proposal and the report |
| CLIP stats, F10/F11 (optional) | Weak automatic mismatch signal | Supporting evidence only |

### Missing and broken data (handbook: "missing modality rates")
| Output | Meaning | Use |
|---|---|---|
| `questions_missing_annotation`, `annotations_missing_question`, `question_annotation_image_id_mismatch` | Broken links between question, answers and image | Should be 0. Report it. |
| `text_hygiene` | Empty or very short questions, missing "?", non-ASCII text, missing answers | Cleaning rules for Phase 4 |
| `subset_images`: missing, corrupt, `missing_modality_rate_pct` | Questions whose image is unusable | **The missing-modality rate.** Decide in Phase 4: drop these, or handle them as missing |

### Text modality (handbook: "text length")
| Output | Meaning | Use |
|---|---|---|
| `question_length_words`, F5 | Question length in words | Text-length EDA |
| `token_lengths` (with `--tokenizer`) | Length in model tokens, % over 16/24/32 | Pick the max sequence length so that at least 99% of questions fit (Phase 5) |
| `majority_answer_length_words`, `pct_multi_word_majority_answers` | Most answers are 1–3 words | Justifies answer classification over generation |

### Image modality
| Output | Meaning | Use |
|---|---|---|
| width/height/aspect ratio, F7 | COCO images are mostly about 640 × 480 | Preprocessing: resize and crop to the encoder's input size (e.g. 224) |
| `formats`, `modes` | Grayscale (`L`) images exist in COCO | Preprocessing: convert to RGB |
| `images_smaller_than_224px_side` | Images that need upsampling | Preprocessing note |
| `total_mb`, `file_kb` | Disk size of the subset | Phase 6 compute and storage estimate |

### Labels, i.e. answers as classes (the classification framing)
| Output | Meaning | Use |
|---|---|---|
| `answer_distribution`: top-1 share, top-10 share, normalized entropy, answers seen once | How imbalanced and long-tailed the classes are ("yes" and "no" dominate) | Distribution analysis; explains why macro measures and per-type accuracy matter |
| F2, F3 coverage curve | How many questions K answer classes cover | Choosing K |
| Table 5: vocab options with **test oracle VQA accuracy** | The best score any classifier limited to that vocabulary could reach | **Choose K here** (Phase 5), e.g. the smallest K whose oracle is within about 1 point of the largest. The vocabulary is built from the train pool only, never the test set. |

### Bias and leakage
| Output | Meaning | Use |
|---|---|---|
| Table 6 language prior | Accuracy from the question type alone, with no image and no model | The floor your models must clearly beat. The question-only baseline should land above it; the fusion model should beat both. |
| `leakage_in_text` | % where the answer is literally in the question, and "A or B?" questions | Known shortcut; analyze these separately later |
| `question_text_overlap` | The same question text appears in train and test on different images, which is normal | Explains why the split unit is the **image**, not the question |
| `full_image_overlap_train_val`, `near_dups_*` | Same or near-identical images across train pool and test | Leakage control. Report the real duplicates and remove them in Phase 4. |
| Table 7 representativeness (TVD) | How close the random subset is to the full split (0 = identical; under about 0.02 is very close) | **Justifies the subset rule**: the subset is a faithful sample |

### Reproducibility
- `summary.json → meta`: run time, Python and package versions, input checksums and arguments.
- `subset/*.txt` and `subset/*_question_ids.csv`: the exact subset. Commit these. Phase 4 builds the splits from them.

---
