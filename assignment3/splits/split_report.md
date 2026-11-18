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

| measure | seed-261 TVD | null mean | null p50 | null p95 | null p99 | seed-261 percentile | check |
|---|---|---|---|---|---|---|---|
| answer_type | 0.0064 | 0.0083 | 0.0077 | 0.0169 | 0.0205 | 37.6 | <= 0.02 |
| top-20 question_type (+other) | 0.0229 | 0.0241 | 0.0238 | 0.0320 | 0.0357 | 41.8 | <= null p95 |

Null: 1000 random image-level 9000/1000 splits (seeds 0..999, same near-dup grouping, same cleaning; top-20 recomputed per split). Values in `tvd_null.csv`.

**Deviation from plan:** the fixed 0.02 limit for question_type TVD was an untested guess. The first run gave 0.0229 and failed it, so the check was replaced by seed-261 TVD <= p95 of the random-split null. The split itself is unchanged (seed 261, not reseeded). answer_type keeps the fixed 0.02 limit.

## Choosing K (train vocab, val only)

Train vocab size: 5,335 distinct answers.

| K | val oracle VQA acc % | train coverage % | val coverage % | K-th answer | its train count |
|---|---|---|---|---|---|
| 500 | 86.26 | 83.33 | 83.36 | checkered | 7 |
| 771 (K_rule) | 88.54 | 86.59 | 85.93 | yellow and blue | 5 |
| 1000 | 89.54 | 88.41 | 87.24 | animals | 3 |
| 2000 | 91.65 | 93.09 | 90.5 | 1 6 | 1 |
| 3000 | 92.17 | 95.16 | 91.23 | document | 1 |

**K choice (pre-registered rule):** largest K whose K-th most frequent train answer has >= 5 train examples gives **K_rule = 771** (answer 'yellow and blue', train count 5). `answer_vocab.json` maps index -> answer (train only, sorted by -count then answer ascending), size K_rule.

**Text encoder:** bert-base-uncased, max_len = 32 (train p99 = 16.0, max = 27).

BERT question length (train only, bert-base-uncased, incl. [CLS]/[SEP]): p95=13.0, p99=16.0, max=27

## Questions per image

| split | max | mean | top-5 (image: n) |
|---|---|---|---|
| train | 176 | 5.36 | 265622: 176, 334098: 125, 490596: 99, 349907: 80, 279182: 73 |
| val | 92 | 5.26 | 61842: 92, 474955: 78, 218334: 64, 25872: 53, 189617: 30 |
| test | 88 | 5.24 | 223241: 88, 229427: 69, 82157: 64, 479448: 50, 175785: 42 |

## Notes

- No per-image question cap: the largest image (265622) has 176 questions = 0.36% of train.
- The train answer "1 6" is likely a normalization artifact of "1/6"; left as is.

## Checklist

- [x] **PASS** Image ids disjoint across train/val/test
- [x] **PASS** Question ids disjoint across train/val/test
- [x] **PASS** No near-dup group (real pairs) crosses splits
- [x] **PASS** Answer vocab built from TRAIN rows only (no val/test question ids used)
- [x] **PASS** Train/val image counts are 9000/1000 (got 9000/1000)
- [x] **PASS** TVD answer_type <= 0.02 (0.0064)
- [x] **PASS** TVD top-20 question_type (0.0229) <= p95 of 1000 random splits (0.0320)

## SHA-256

- `answer_vocab.json`: `06670f55d59bccac9dba6c45cad8e54f4c58425c8d84751856e53288c3e20576`
- `removed.csv`: `dab53da36d4c5e4844fe8819222adac90c7d1efcad59e3e104717cb067560912`
- `test_image_ids.txt`: `8053ead1439801bff7aa81b77f489ee7fb1a34b429152b09c7139e147765325d`
- `test_question_ids.csv`: `d3082d007e7bd2b451baa42c680695a4706d3f1d3ab802448f44f999e8552843`
- `train_image_ids.txt`: `7dae12473f369c4ff0bab06e596abb7e7ae83e9c15d8380e340ed880ea6e94d9`
- `train_question_ids.csv`: `e2749e7d930a396f27352420cc860914150d346f34b653459d362e287fddb5f2`
- `tvd_null.csv`: `c6765113ebf91fd140c62bb44ff6a1c891d0af39034f7210c0d4fd1af090e014`
- `val_image_ids.txt`: `679c18595d94a3f17c079b0b5251f5625f9712de1b16b93cc4deb5cdcac432ac`
- `val_question_ids.csv`: `70f48fb45f7878bd1910c3a018d9be3f9ae8bb0bc309090493d314c094a75298`
