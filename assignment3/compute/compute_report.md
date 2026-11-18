# Phase 6 compute report (measured on this machine)

Script: `bench.py` (raw numbers in `compute_results.json`); this table: `make_report.py`.

## Environment
NVIDIA GeForce RTX 4050 Laptop GPU, 5772 MiB VRAM (laptop GPU), CPU 12th Gen Intel i5-12450HX (12 threads), torch 2.15.0+cu132 (CUDA 13.2, cuDNN 92600),
torchvision 0.29.2+cu132, transformers 5.23.0, Python 3.14.4.
torchvision was not in the venv; installed with `pip install --no-deps` (7.6 MB), plus the 98 MB ResNet-50 IMAGENET1K_V2 weights.

## Measurements
| item | setup | measured | extrapolated |
|---|---|---|---|
| ResNet-50 features (pooled 2048 + 7x7x2048 grid) | 512 train images, batch 64, 224x224, fp16 autocast, 1 warm-up batch, includes copy to host | 713.5 img/s | 15,000 images = 21 s |
| Image decode + resize (8 workers, warm page cache) | same 512 images | 892.0 img/s | not the bottleneck |
| Disk, pooled fp16 | | 4,096 B/img | 0.061 GB for 15k |
| Disk, grid fp16 | | 200,704 B/img | 3.01 GB for 15k (train 9,000 imgs: 1.81 GB) |
| BERT-base frozen, [CLS] + tokens | 2,048 train questions, max_len 32, batch 256, fp16 autocast, 1 warm-up batch | 2575.6 q/s (tokenizing 57,140 q/s, excluded) | 79,731 questions = 31 s |
| Disk, [CLS] only fp16 | | 1,536 B/q | 122.5 MB for 79,731 (all 32 tokens: 3.92 GB) |
| Concat MLP head (2048+768 -> 1024 -> 771, soft-score BCE) | 48,245 random cached tensors on GPU, batch 256, bf16 autocast, mean of 3 epochs after 1 warm-up | **0.33 s/epoch** (3.67 M params) | 20 epochs = 7 s |
| Single-glimpse attention head over 7x7 grid | same, grid cycled from 8,192 GPU-resident samples (compute only) | 2.75 s/epoch (6.56 M params) | |
| ... plus streaming grid from pinned CPU memory | 0.0061 s/batch x 189 batches | +1.15 s/epoch | **3.91 s/epoch**, 20 epochs = 78 s |
| Fine-tune step, BERT top 2 layers unfrozen | batch 64, max_len 32, bf16 autocast, median of 30 steps after 5 warm-up | **0.0379 s/step**, peak VRAM 1172 MiB allocated / 1238 MiB reserved | 754 steps = 28.6 s/epoch (image features cached, not included) |

## Caveats
- The full train grid cache is 9.68 GB, which does not fit in 5772 MiB of VRAM and is tight in 14 GB of RAM.
  The attention-head time therefore assumes streaming from RAM/pinned memory; reading from disk (memmap) could be slower. Not measured.
- Pooled-feature extraction time is GPU-bound (decode ran at 892.0 img/s with a warm page cache; a cold read of 2.4 GB of JPEGs is not included).
- Inputs are random tensors, so no accuracy information. The timings exclude validation passes and checkpointing (add roughly 10%).

## GPU-hours (planning assumptions are mine; adjust if the model list differs)
Assumptions: 5 models x 3 seeds x 20 epochs = 300 epochs; ablations = 10 extra single-seed runs of 20 epochs; every model costs
the MLP rate (lower bound) or the attention-head rate (upper bound). Feature caching = ResNet (15k images) + BERT (79,731 questions) = 52 s.

| component | lower bound (all MLP-rate) | upper bound (all attention-rate) |
|---|---|---|
| feature caching | 0.014 h | 0.014 h |
| 5 models x 3 seeds x 20 epochs | 0.028 h | 0.326 h |
| ablations (10 runs x 20 epochs) | 0.018 h | 0.217 h |
| **total** | **0.06 GPU-h** | **0.56 GPU-h** |

If one of the 5 models also fine-tunes BERT's top 2 layers (3 seeds x 20 epochs x 28.6 s = 0.48 h), the upper bound becomes **0.97 GPU-h**.
Even with a 2x safety margin the whole plan is under 1.9 GPU-hours.
