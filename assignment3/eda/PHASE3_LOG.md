# Phase 3 log: VQA v2 download and EDA (CO3133 Assignment 3)

Self-contained record of Phase 3. Read this first in a new session. Task: VQA v2 framed as answer
classification over the top-K answers. Kit: `vqa_eda.py`, `audit.py`, `PHASE3_GUIDE.md`,
`requirements-eda.txt`. Date of work: 2026-11-18. Phase 3 PR #13 is merged into `main`.

## Standing rules (from the user)
- Never add a `Co-Authored-By` trailer or any Claude/AI attribution to commits, PRs or repo files.
- Don't push without asking; don't merge PRs (the user merges; `main` needs 1 approving review).
- Don't change the subset rules, seed (261), 10k train / 5k test images, or the metrics.
- Never fill in audit labels without the user's explicit verdicts.
- Ask before any download over 1 GB. Commit only code and small outputs; raw data and images are gitignored.
- A normal rerun of `vqa_eda.py` into `eda_outputs/` overwrites `audit_sample.csv` (the user's 100 verdicts).
  To regenerate figures, run into a scratch `--out-dir` and copy the PNGs you need.

## 1. Setup and data
- Venv `assignment3/eda/.venv` (Python 3.14.4) already had the `requirements-eda.txt` packages plus torch 2.15 and
  transformers 5.23 (the user installed them in guide step 3.1). Nothing else was installed.
- Annotations downloaded 2026-11-18 from `s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/` (about 43 MB): VQA v2 train and val
  questions and annotations, unzipped to `data/vqa_v2/` (date in `data/vqa_v2/DOWNLOAD_DATE.txt`).
- Images: about 2.4 GB (15,000 COCO images) downloaded by the script after the user approved. Plus the BERT tokenizer
  and CLIP weights.
- `assignment3/eda/.gitignore` ignores `data/`, `run2.log`, and most of `eda_outputs/`. It re-includes
  `proposal_tables.md`, `summary.json`, `figures/`, `subset/*.txt`, `subset/*_question_ids.csv`,
  `subset/near_dups_*.csv`, `audit_sample.csv` and `audit_summary.json`.

## 2. Commands run (in `assignment3/eda/`)
- Run 1 (annotations only): `.venv/bin/python -I vqa_eda.py --data-dir data/vqa_v2 --out-dir eda_outputs` (26 s).
- Run 2 (full): same plus `--images --workers 16 --tokenizer bert-base-uncased --clip openai/clip-vit-base-patch32`
  (about 25 min, seed 261, 10k train images, 5k test images).
- `audit.py html --audit eda_outputs/audit_sample.csv --data-dir data/vqa_v2 --out eda_outputs/audit_sheet.html`
- `audit.py summary --audit eda_outputs/audit_sample.csv --out eda_outputs/audit_summary.json`
- No script broke on real data, so there were no bug fixes. The only code changes are cosmetic plot edits (section 7).

## 3. Findings
- Full splits match the official counts: train2014 82,783 images / 443,757 questions; val2014 40,504 / 214,354.
  Train/val image overlap is 0.
- Subset: train pool 10,000 images / 53,522 questions; test 5,000 images / 26,221 questions (val2014).
- 0 missing or corrupt images (missing-modality rate 0%). Sizes: median 640x480, min 120x130, max 640x640.
- 0 broken question/annotation/image links.
- Near-duplicates: 3 flagged pairs inside the train pool, none in test or across splits. User review: 9878/141785
  real duplicate, 167028/224112 real duplicate, 406749/412172 false alarm (violet vs pink flowers). Phase 4: drop
  one image from each real pair. Review file: `eda_outputs/subset/near_dups_within_trainpool_review.csv`.
- 6 duplicate (image, question) pairs per subset (cause not investigated). 6 empty majority answers per subset.
  1 test question under 3 words. No non-ASCII text.
- Agreement: top answer has >=3 votes in 93.82% (train pool) / 93.33% (test); all 10 agree 33.35% / 33.2%.
- Language prior (test VQA accuracy): always "yes" 23.92%; best per-question-type rule 31.8%
  (63.47% yes/no, 26.64% number, 9.19% other).
- Representativeness (TVD): at most 0.016 against the full splits.
- BERT tokens: mean 9.4, p95 13, max 27; 0.88% over 16 tokens; a max length of 24 covers 99.99%.
- CLIP similarity: mean 0.241 (train) / 0.243 (test) over 79,743 pairs.
- Manual audit (n = 100, the user's own verdicts, transcribed by Claude): 98 matched, 1 weak, 1 mismatched
  (Wilson 95% CI: matched 93.0-99.4%, weak 0.2-5.4%, mismatched 0.2-5.4%). Card 47 mismatched ("How much is a cronut?",
  no price shown); card 57 weak (three arrows, question denotes one). File: `eda_outputs/audit_summary.json`.

## 4. Outputs
Committed: `proposal_tables.md` (Tables 1-8), `summary.json` (with input sha256s and JSON headers), figures F1-F11,
`subset/*.txt` and `*_question_ids.csv`, `near_dups_*.csv` (+ review), `audit_sample.csv`, `audit_summary.json`.
Not committed (gitignored): raw data and images, `audit_sheet.html`, `clip_pair_similarity.csv`,
`image_checks.csv`, `run2.log`.

## 5. Git history
PR #13 (`feat/a3-eda`, merged 2026-11-18): EDA kit and outputs; audit and summary; near-duplicate review; plot-label
fixes for F2-F5; `.gitignore` tidy. PR #13 needed an approving review before it merged. The figure fixes below are on
branch `feat/a3-eda-figfix`.

## 6. Open items
- Phase 4: drop one image from each real near-duplicate pair; decide what to do with duplicate (image, question) pairs.
- Phase 5: choose K and the max sequence length (see the Table 5 note below).
- Report: caption F1, F8-F11 as examples/subsets, as set out in the notes below.

---

## Figure fixes and checks (2026-11-18)

### F1-F11 fixes (all plotting code in `vqa_eda.py`; metrics, seed and subset rules untouched)
1. **F4:** title is now "Top-20 VQA question types (official question_type, mutually exclusive)". An earlier interim title
   called these overlapping prefixes; that was wrong. The data is VQA's official `question_type` field (read from the
   annotations at `vqa_eda.py` line 183), so the types are mutually exclusive. Names such as "what is the" and "is this a"
   are the official type names.
2. **Colours:** one `ANSWER_TYPE_COLORS` map (yes/no blue, number orange, other green) shared by F1, F2, F6.
3. **Subset and n in every aggregate title.** Data actually used: F1 train pool (53,522) and test (26,221);
   F2, F3, F4 train pool only; F5, F6, F10 train pool + test (79,743 questions; F5 also 15,000 images);
   F7 15,000 images; F8 and F9 train pool; F11 train pool + test.
   F8, F9, F11 carry a suptitle with subset, n and seed.
4. **F5:** bins centred on integers; overflow bins labelled "20+" (question words), "15+" (questions per image),
   "5+" (answer words); y-axis "count".
5. **F3:** markers at K = 500, 1000, 2000, 3000 (coverage 83.4%, 88.4%, 92.9%, 95.0%), matching Table 5; title wrapped
   so it is not cut off.
6. **F9:** question_id 183071006 ("What type of hat is the dog wearing?", answers include an offensive word) excluded in
   code, then resampled with seed 261. The nine new panels were checked and are clean.
7. **F2:** added an answer-type legend. **F7:** n moved to a single title over both panels.
8. Earlier (also on `main`): F2 legend, F3 label overlap, F4 title, F5 y-labels.
- Regeneration method: full script run into a scratch folder (one run included CLIP, 10.5 min) and PNGs copied in.
  Subset id files were byte-identical, and all numeric entries in `summary.json` matched the committed ones.
  `audit_sample.csv` was never touched.
- Figures are not individually re-verified by anyone else: Haiku reviewed the first versions and Claude opened all
  eleven after the final fixes. The user should still look at them before they go in the report.

### F11 "checkers" lookup (genuine VQA v2 annotation mismatch)
Caption/note for F11: *"Includes a genuine VQA v2 annotation mismatch (qid 82157001: checkerboard-floor question
attached to image 82157, which has no floor)."* F11 shows examples only, never a mismatch rate.
- Raw `v2_OpenEnded_mscoco_val2014_questions.json` and `v2_mscoco_val2014_annotations.json`: question_id 82157001,
  "What game is played on a board with a similar pattern to this floor?", answer "checkers", image_id **82157**.
- The F11 panel loads `COCO_val2014_000000082157.jpg` (image_id 82157, val2014): a bearded man holding a pink bitten donut
  against a black background, no board and no floor. `clip_pair_similarity.csv` has one row for this question, also
  image_id 82157 / val2014, so the panel and ids agree. It is not a pipeline id mix-up.
- Image 82157 has **64 questions** in the raw val file; the other 63 are about the man, the donut, his shirt, etc.
  (for example "What is he eating?", "What flavor donut is this?"). The checkers question is the odd one out.
- Coincidence noted: F9's bathroom image (train2014 image_id **490596**, question 490596077 "What is the date when the
  photo was taken?") has a checkered floor, which the checkers question would fit. The ids do not link the two,
  so do not claim that it is the source image.

### Table 5 vocabulary: train pool only (verified in code)
The answer vocabulary for Table 5 (oracle VQA accuracy by K) is built from the train pool only; the test set is used
only to evaluate coverage and the oracle ceiling.
- `vqa_eda.py` line 825: `s["answer_vocab_from_train_pool"] = vocab_options(sub_tr, sub_te)`
- `vocab_options(train, test)` (line 337): `counts = Counter(train["mc_answer"])`, then `ranked = [a for a, _ in counts.most_common()]`;
  the min-count sets come from `counts`; the top-K sets are `set(ranked[:k])` for k in (500, 1000, 2000, 3000).
- Test is touched only at lines 347-349: `test["mc_answer"].isin(v)` for coverage and `oracle_accuracy(g, v)` over `test["answers"]`.
- Because the test subset only measures a ceiling, Table 5 is now labelled: **"Test-set ceiling (vocab from train pool),
  descriptive only, not used to choose K."** (changed both in `proposal_tables.md` and in the generator string at
  `vqa_eda.py` line 738). The earlier guide text said to choose K from this table; per the user's decision it is not used for that.
  So K must be chosen another way in Phase 5 (for example a validation split built from the train pool).
- Table 5 values for reference (test oracle VQA acc %): min_count_3 K=1,393 89.38; min_count_5 K=830 87.51;
  min_count_9 K=476 84.69; top_500 84.97; top_1000 88.23; top_2000 90.35; top_3000 90.89.

### Cleanups and caveats
- `int_hist` in `vqa_eda.py` returns an overflow count that nothing uses (harmless).
- `summary.json` records Run 2's timestamp; the figures were redrawn later from the updated script.
- Two background wait-loops I started earlier matched their own command line in `pgrep -f` and never exited;
  they were killed. If you wait for a script, poll its PID or log instead of `pgrep -f` on the script name.
