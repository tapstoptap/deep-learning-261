# M2 results

All accuracies in %. Trained models: mean ± sample std over seeds 261, 262, 263.

## val (n = 5,265; yes/no 1,996, number 701, other 2,568; oracle ceiling 88.54)

| model | VQA acc | exact match | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| a: prior (no training) | 31.88 | 25.36 | 62.90 | 26.55 | 9.22 | 22.56 |
| c: image only (unimodal A) | 25.55 ± 0.20 | 20.70 ± 0.32 | 65.22 ± 0.68 | 0.60 ± 0.07 | 1.52 ± 0.19 | 0.11 ± 0.15 |
| b: text only (unimodal B) | 39.19 ± 0.20 | 30.90 ± 0.31 | 66.06 ± 0.15 | 28.62 ± 1.48 | 21.19 ± 0.20 | 27.62 ± 0.79 |
| d: concat fusion | 43.88 ± 0.24 | 35.11 ± 0.38 | 66.71 ± 0.18 | 27.72 ± 0.53 | 30.56 ± 0.46 | 34.61 ± 0.66 |

## test (n = 26,221; yes/no 9,811, number 3,459, other 12,951; oracle ceiling 87.10)

| model | VQA acc | exact match | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| a: prior (no training) | 31.85 | 25.46 | 63.42 | 26.64 | 9.32 | 24.38 |
| c: image only (unimodal A) | 24.29 ± 0.47 | 19.14 ± 0.50 | 62.84 ± 1.30 | 0.41 ± 0.11 | 1.46 ± 0.16 | 0.46 ± 0.29 |
| b: text only (unimodal B) | 38.59 ± 0.10 | 30.79 ± 0.23 | 65.43 ± 0.15 | 28.70 ± 0.84 | 20.91 ± 0.36 | 32.07 ± 1.08 |
| d: concat fusion | 42.98 ± 0.22 | 34.79 ± 0.19 | 66.21 ± 0.34 | 28.04 ± 0.24 | 29.37 ± 0.40 | 38.12 ± 0.89 |

## Per-seed gaps (percentage points)

### b − c (text only minus image only)

| split | seed | overall | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| val | 261 | +13.90 | +1.52 | +28.05 | +19.67 | +27.89 |
| val | 262 | +13.54 | -0.04 | +29.54 | +19.72 | +26.65 |
| val | 263 | +13.48 | +1.01 | +26.46 | +19.63 | +27.98 |
| test | 261 | +14.68 | +3.63 | +28.18 | +19.43 | +32.64 |
| test | 262 | +13.70 | +1.17 | +29.22 | +19.04 | +31.08 |
| test | 263 | +14.55 | +2.97 | +27.46 | +19.88 | +31.12 |

### d − b (fusion minus text only)

| split | seed | overall | yes/no | number | other | what color |
|---|---|---|---|---|---|---|
| val | 261 | +4.85 | +0.81 | -1.24 | +9.65 | +7.03 |
| val | 262 | +4.49 | +0.77 | -2.64 | +9.33 | +7.18 |
| val | 263 | +4.75 | +0.37 | +1.18 | +9.12 | +6.75 |
| test | 261 | +4.48 | +1.03 | -0.54 | +8.44 | +5.86 |
| test | 262 | +4.46 | +0.41 | -1.77 | +9.19 | +6.30 |
| test | 263 | +4.20 | +0.90 | +0.33 | +7.74 | +5.98 |

## Training cost (heads only, cached features)

| model | parameters | s/epoch | train s, 3 seeds | best epoch (261, 262, 263) |
|---|---|---|---|---|
| c: image only (unimodal A) | 2,892,547 | 0.36 | 21.8 | 20, 4, 16 |
| b: text only (unimodal B) | 1,579,267 | 0.27 | 16.5 | 19, 18, 20 |
| d: concat fusion | 3,680,515 | 0.45 | 27.2 | 20, 17, 9 |
