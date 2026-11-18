# Phase 4 split report

Seed 261. Test set is never used for any choice below.

## Counts

| split | images | questions (raw) | removed | questions (final) |
|---|---|---|---|---|
| train | 9,000 | 48,255 | 10 | 48,245 |
| val | 1,000 | 5,267 | 2 | 5,265 |
| test | 5,000 | 26,221 | 0 | 26,221 |

Test not filtered: 6 empty-majority questions, 6 exact duplicate (image, question) pairs (3 with identical majority answer). Report-only.

Removed (train/val): duplicate_image_question_same_answer=6, empty_majority_answer=6.

## Balance (train vs val, after cleaning)

| measure | TVD | limit |
|---|---|---|
| answer_type | 0.0064 | 0.02 |
| top-20 question_type (+other) | 0.0229 | 0.02 |

## Choosing K (train vocab, val only)

Train vocab size: 5,335 distinct answers.

| K | val oracle VQA acc % | train coverage % | val coverage % | K-th answer | its train count |
|---|---|---|---|---|---|
| 500 | 86.26 | 83.33 | 83.36 | checkered | 7 |
| 1000 | 89.54 | 88.41 | 87.24 | animals | 3 |
| 2000 | 91.65 | 93.09 | 90.5 | 1 6 | 1 |
| 3000 | 92.17 | 95.16 | 91.23 | document | 1 |

BERT question length (train only, bert-base-uncased, incl. [CLS]/[SEP]): p95=13.0, p99=16.0, max=27

## Questions per image

| split | max | mean | top-5 (image: n) |
|---|---|---|---|
| train | 176 | 5.36 | 265622: 176, 334098: 125, 490596: 99, 349907: 80, 279182: 73 |
| val | 92 | 5.26 | 61842: 92, 474955: 78, 218334: 64, 25872: 53, 189617: 30 |
| test | 88 | 5.24 | 223241: 88, 229427: 69, 82157: 64, 479448: 50, 175785: 42 |

## Checklist

- [x] **PASS** Image ids disjoint across train/val/test
- [x] **PASS** Question ids disjoint across train/val/test
- [x] **PASS** No near-dup group (real pairs) crosses splits
- [x] **PASS** Answer vocab built from TRAIN rows only (no val/test question ids used)
- [x] **PASS** Train/val image counts are 9000/1000 (got 9000/1000)
- [x] **PASS** TVD answer_type <= 0.02 (0.0064)
- [ ] **FAIL** TVD top-20 question_type <= 0.02 (0.0229)

## SHA-256

- `removed.csv`: `dab53da36d4c5e4844fe8819222adac90c7d1efcad59e3e104717cb067560912`
- `test_image_ids.txt`: `8053ead1439801bff7aa81b77f489ee7fb1a34b429152b09c7139e147765325d`
- `test_question_ids.csv`: `d3082d007e7bd2b451baa42c680695a4706d3f1d3ab802448f44f999e8552843`
- `train_image_ids.txt`: `7dae12473f369c4ff0bab06e596abb7e7ae83e9c15d8380e340ed880ea6e94d9`
- `train_question_ids.csv`: `e2749e7d930a396f27352420cc860914150d346f34b653459d362e287fddb5f2`
- `val_image_ids.txt`: `679c18595d94a3f17c079b0b5251f5625f9712de1b16b93cc4deb5cdcac432ac`
- `val_question_ids.csv`: `70f48fb45f7878bd1910c3a018d9be3f9ae8bb0bc309090493d314c094a75298`
