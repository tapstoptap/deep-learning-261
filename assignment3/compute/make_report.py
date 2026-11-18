"""Builds compute_report.md from compute_results.json (measured numbers) plus stated planning assumptions."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.load(open(H / "compute_results.json"))
e, rn, b, m, g, f = R["env"], R["resnet"], R["bert"], R["mlp_head"], R["glimpse_head"], R["finetune_top2"]
EP = 20
cache_s = rn["extrap_15k_s_gpu_only"] + b["extrap_79731_s"]
mlp, gl, ft = m["s_per_epoch"], g["s_per_epoch_with_pinned_h2d"], f["s_per_epoch"]
runs_main, runs_abl = 5 * 3, 10  # 5 models x 3 seeds; ablations: assumed 10 extra single-seed runs
h = lambda s: s / 3600  # noqa: E731
lo = cache_s + runs_main * EP * mlp + runs_abl * EP * mlp
hi = cache_s + runs_main * EP * gl + runs_abl * EP * gl
ft_main = 3 * EP * ft  # one of the 5 models fine-tunes BERT top-2, 3 seeds
hi_ft = hi - 3 * EP * gl + ft_main
L = f"""# Phase 6 compute report (measured on this machine)

Script: `bench.py` (raw numbers in `compute_results.json`); this table: `make_report.py`.

## Environment
{e['gpu']}, {e['vram_mib']} MiB VRAM (laptop GPU), CPU {e['cpu']}, torch {e['torch']} (CUDA {e['cuda']}, cuDNN {e['cudnn']}),
torchvision {e['torchvision']}, transformers {e['transformers']}, Python {e['python']}.
torchvision was not in the venv; installed with `pip install --no-deps` (7.6 MB), plus the 98 MB ResNet-50 IMAGENET1K_V2 weights.

## Measurements
| item | setup | measured | extrapolated |
|---|---|---|---|
| ResNet-50 features (pooled 2048 + 7x7x2048 grid) | 512 train images, batch 64, 224x224, fp16 autocast, 1 warm-up batch, includes copy to host | {rn['gpu_only_img_per_s']} img/s | 15,000 images = {rn['extrap_15k_s_gpu_only']} s |
| Image decode + resize (8 workers, warm page cache) | same 512 images | {rn['decode_preprocess_img_per_s_8workers']} img/s | not the bottleneck |
| Disk, pooled fp16 | | {rn['pooled_fp16_bytes_per_img']:,} B/img | {rn['pooled_15k_GB']} GB for 15k |
| Disk, grid fp16 | | {rn['grid_fp16_bytes_per_img']:,} B/img | {rn['grid_15k_GB']} GB for 15k (train 9,000 imgs: {rn['grid_fp16_bytes_per_img']*9000/1e9:.2f} GB) |
| BERT-base frozen, [CLS] + tokens | 2,048 train questions, max_len 32, batch 256, fp16 autocast, 1 warm-up batch | {b['questions_per_s']} q/s (tokenizing {b['tokenize_questions_per_s']:,} q/s, excluded) | 79,731 questions = {b['extrap_79731_s']} s |
| Disk, [CLS] only fp16 | | {b['cls_fp16_bytes_per_q']:,} B/q | {b['cls_79731_MB']} MB for 79,731 (all 32 tokens: {b['tokens_79731_GB']} GB) |
| Concat MLP head (2048+768 -> 1024 -> 771, soft-score BCE) | 48,245 random cached tensors on GPU, batch 256, bf16 autocast, mean of 3 epochs after 1 warm-up | **{mlp} s/epoch** ({m['params_M']} M params) | 20 epochs = {mlp*EP:.0f} s |
| Single-glimpse attention head over 7x7 grid | same, grid cycled from 8,192 GPU-resident samples (compute only) | {g['s_per_epoch_compute_only']} s/epoch ({g['params_M']} M params) | |
| ... plus streaming grid from pinned CPU memory | {g['h2d_s_per_batch']} s/batch x 189 batches | +{g['h2d_s_per_epoch']} s/epoch | **{gl} s/epoch**, 20 epochs = {gl*EP:.0f} s |
| Fine-tune step, BERT top 2 layers unfrozen | batch 64, max_len 32, bf16 autocast, median of 30 steps after 5 warm-up | **{f['s_per_step_median']} s/step**, peak VRAM {f['peak_vram_alloc_MiB']} MiB allocated / {f['peak_vram_reserved_MiB']} MiB reserved | {f['steps_per_epoch']} steps = {ft} s/epoch (image features cached, not included) |

## Caveats
- The full train grid cache is {g['train_grid_fp16_GB']} GB, which does not fit in {e['vram_mib']} MiB of VRAM and is tight in 14 GB of RAM.
  The attention-head time therefore assumes streaming from RAM/pinned memory; reading from disk (memmap) could be slower. Not measured.
- Pooled-feature extraction time is GPU-bound (decode ran at {rn['decode_preprocess_img_per_s_8workers']} img/s with a warm page cache; a cold read of 2.4 GB of JPEGs is not included).
- Inputs are random tensors, so no accuracy information. The timings exclude validation passes and checkpointing (add roughly 10%).

## GPU-hours (planning assumptions are mine; adjust if the model list differs)
Assumptions: 5 models x 3 seeds x {EP} epochs = {runs_main*EP} epochs; ablations = {runs_abl} extra single-seed runs of {EP} epochs; every model costs
the MLP rate (lower bound) or the attention-head rate (upper bound). Feature caching = ResNet (15k images) + BERT (79,731 questions) = {cache_s} s.

| component | lower bound (all MLP-rate) | upper bound (all attention-rate) |
|---|---|---|
| feature caching | {h(cache_s):.3f} h | {h(cache_s):.3f} h |
| 5 models x 3 seeds x {EP} epochs | {h(runs_main*EP*mlp):.3f} h | {h(runs_main*EP*gl):.3f} h |
| ablations ({runs_abl} runs x {EP} epochs) | {h(runs_abl*EP*mlp):.3f} h | {h(runs_abl*EP*gl):.3f} h |
| **total** | **{h(lo):.2f} GPU-h** | **{h(hi):.2f} GPU-h** |

If one of the 5 models also fine-tunes BERT's top 2 layers (3 seeds x {EP} epochs x {ft} s = {h(ft_main):.2f} h), the upper bound becomes **{h(hi_ft):.2f} GPU-h**.
Even with a 2x safety margin the whole plan is under {2*h(hi_ft):.1f} GPU-hours.
"""
(H / "compute_report.md").write_text(L)
print(L)
