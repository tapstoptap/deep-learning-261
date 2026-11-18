"""Phase 6 timing benchmarks (measured, not estimated). Writes compute_results.json; see compute_report.md.
Run from assignment3/compute:  ../eda/.venv/bin/python -I bench.py
"""
import json, os, platform, statistics as st, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EDA = HERE.parent / "eda"
SPL = HERE.parent / "splits"
IMG_DIR = EDA / "data" / "vqa_v2" / "images" / "train2014"
dev = "cuda"
R = {}
sync = torch.cuda.synchronize


def timeit(fn):
    sync(); t = time.perf_counter(); out = fn(); sync()
    return time.perf_counter() - t, out


# ---- 1. environment
import torchvision, transformers  # noqa: E402
p = torch.cuda.get_device_properties(0)
R["env"] = {"gpu": p.name, "vram_mib": p.total_memory // 2**20, "cpu": "12th Gen Intel i5-12450HX (12 threads)",
            "torch": torch.__version__, "cuda": torch.version.cuda, "torchvision": torchvision.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "cudnn": torch.backends.cudnn.version()}
torch.manual_seed(0)

# ---- 2. ResNet-50 feature extraction
from torchvision.models import resnet50, ResNet50_Weights  # noqa: E402
from PIL import Image  # noqa: E402
tf = ResNet50_Weights.IMAGENET1K_V2.transforms(resize_size=256, crop_size=224)
net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(dev)
trunk = nn.Sequential(*list(net.children())[:-2])  # -> B x 2048 x 7 x 7
ids = sorted(int(x) for x in (SPL / "train_image_ids.txt").read_text().split())[:512]


class DS(torch.utils.data.Dataset):
    def __len__(self): return len(ids)
    def __getitem__(self, i):
        return tf(Image.open(IMG_DIR / f"COCO_train2014_{ids[i]:012d}.jpg").convert("RGB"))


# pre-decode the 512 images once so GPU time and loader time are reported separately
t_load0 = time.perf_counter()
dl = torch.utils.data.DataLoader(DS(), batch_size=64, num_workers=8, multiprocessing_context="fork")
batches = [b for b in dl]
t_load = time.perf_counter() - t_load0
pooled, grid = [], []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    trunk(batches[0].to(dev))  # warm-up batch
    sync()
    t0 = time.perf_counter()
    for b in batches:
        g = trunk(b.to(dev, non_blocking=True))
        pooled.append(g.mean((2, 3)).half().cpu()); grid.append(g.flatten(2).transpose(1, 2).half().cpu())
    sync(); t_gpu = time.perf_counter() - t0
pooled = torch.cat(pooled).numpy(); grid = torch.cat(grid).numpy()
np.save(HERE / "_p.npy", pooled); np.save(HERE / "_g.npy", grid)
sz_p, sz_g = os.path.getsize(HERE / "_p.npy") / 512, os.path.getsize(HERE / "_g.npy") / 512
os.remove(HERE / "_p.npy"); os.remove(HERE / "_g.npy")
R["resnet"] = {"n_images": 512, "batch": 64, "precision": "fp16 autocast",
               "gpu_only_img_per_s": round(512 / t_gpu, 1), "gpu_incl_d2h_img_per_s_note": "timing includes copy to host",
               "decode_preprocess_img_per_s_8workers": round(512 / t_load, 1),
               "pooled_fp16_bytes_per_img": int(sz_p), "grid_fp16_bytes_per_img": int(sz_g)}
rn = R["resnet"]
rn["end_to_end_img_per_s_est"] = round(min(rn["gpu_only_img_per_s"], rn["decode_preprocess_img_per_s_8workers"]), 1)
rn["extrap_15k_s_gpu_only"] = round(15000 / rn["gpu_only_img_per_s"])
rn["extrap_15k_s_bottleneck"] = round(15000 / rn["end_to_end_img_per_s_est"])
rn["pooled_15k_GB"] = round(sz_p * 15000 / 1e9, 3); rn["grid_15k_GB"] = round(sz_g * 15000 / 1e9, 2)
del net, trunk, batches; torch.cuda.empty_cache()

# ---- 3. BERT frozen features
from transformers import AutoModel, AutoTokenizer  # noqa: E402
qs = json.load(open(EDA / "data/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"))["questions"]
qtext = {q["question_id"]: q["question"] for q in qs}
qids = pd.read_csv(SPL / "train_question_ids.csv").question_id.to_numpy()[:2048]
texts = [qtext[int(q)] for q in qids]
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
t_tok, enc = timeit(lambda: tok(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt"))
bert = AutoModel.from_pretrained("bert-base-uncased").eval().to(dev)
cls_out, tok_out = [], []


def run_bert(i0, i1, keep):
    ii, am = enc["input_ids"][i0:i1].to(dev), enc["attention_mask"][i0:i1].to(dev)
    h = bert(input_ids=ii, attention_mask=am).last_hidden_state
    if keep:
        cls_out.append(h[:, 0].half().cpu()); tok_out.append(h.half().cpu())


with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    run_bert(0, 256, False)  # warm-up
    sync(); t0 = time.perf_counter()
    for i in range(0, 2048, 256):
        run_bert(i, i + 256, True)
    sync(); t_b = time.perf_counter() - t0
cls_np = torch.cat(cls_out).numpy(); tok_np = torch.cat(tok_out).numpy()
R["bert"] = {"n_questions": 2048, "batch": 256, "max_len": 32, "precision": "fp16 autocast",
             "questions_per_s": round(2048 / t_b, 1), "tokenize_questions_per_s": round(2048 / t_tok),
             "extrap_79731_s": round(79731 / (2048 / t_b)),
             "cls_fp16_bytes_per_q": 768 * 2, "cls_79731_MB": round(768 * 2 * 79731 / 1e6, 1),
             "tokens_fp16_bytes_per_q": 32 * 768 * 2, "tokens_79731_GB": round(32 * 768 * 2 * 79731 / 1e9, 2)}
del bert; torch.cuda.empty_cache()

# ---- 4. head epochs on random cached tensors
N, K, BS = 48245, 771, 256


def epoch_times(model, make_batch, n_epochs=4):
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    ts = []
    for e in range(n_epochs):
        perm = torch.randperm(N, device=dev)
        sync(); t0 = time.perf_counter()
        for i in range(0, N, BS):
            idx = perm[i:i + BS]
            inputs, y = make_batch(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*inputs)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(); ts.append(time.perf_counter() - t0)
    return ts[1:]  # drop first (warm-up) epoch


img_p = torch.randn(N, 2048, device=dev, dtype=torch.float16)
txt = torch.randn(N, 768, device=dev, dtype=torch.float16)
tgt = (torch.rand(N, K, device=dev) ** 8).half()  # soft scores in [0,1]


class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(2816, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, K))
    def forward(s, i, t): return s.f(torch.cat([i, t], 1))


m = MLP().to(dev)
ts = epoch_times(m, lambda idx: ((img_p[idx], txt[idx]), tgt[idx].float()))
R["mlp_head"] = {"params_M": round(sum(x.numel() for x in m.parameters()) / 1e6, 2), "batch": BS,
                 "s_per_epoch": round(st.mean(ts), 2), "epochs_timed": len(ts)}

GN = 8192  # grid for 48,245 x 49 x 2048 fp16 = 9.7 GB does not fit in 6 GB VRAM: keep 8,192 on GPU and cycle (compute-only)
grid_gpu = torch.randn(GN, 49, 2048, device=dev, dtype=torch.float16)


class Glimpse(nn.Module):
    def __init__(s):
        super().__init__()
        s.v = nn.Linear(2048, 1024); s.q = nn.Linear(768, 1024); s.a = nn.Linear(1024, 1)
        s.f = nn.Sequential(nn.Linear(2048 + 768, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, K))
    def forward(s, g, t):
        att = s.a(torch.tanh(s.v(g) * s.q(t).unsqueeze(1))).softmax(1)
        return s.f(torch.cat([(att * g).sum(1), t], 1))


m = Glimpse().to(dev)
ts = epoch_times(m, lambda idx: ((grid_gpu[idx % GN], txt[idx]), tgt[idx].float()))
# host->device cost of streaming grids from pinned CPU memory (what an on-disk/RAM-cached grid would add)
pin = torch.randn(BS, 49, 2048, dtype=torch.float16).pin_memory()
sync(); t0 = time.perf_counter()
for _ in range(20): pin.to(dev, non_blocking=True)
sync(); h2d = (time.perf_counter() - t0) / 20
R["glimpse_head"] = {"params_M": round(sum(x.numel() for x in m.parameters()) / 1e6, 2), "batch": BS,
                     "s_per_epoch_compute_only": round(st.mean(ts), 2),
                     "h2d_s_per_batch": round(h2d, 4), "h2d_s_per_epoch": round(h2d * (N / BS), 2),
                     "s_per_epoch_with_pinned_h2d": round(st.mean(ts) + h2d * (N / BS), 2),
                     "train_grid_fp16_GB": round(N * 49 * 2048 * 2 / 1e9, 2),
                     "note": "grid data cycled from 8,192 GPU-resident samples; full train grid needs 9.7 GB"}
del grid_gpu, m; torch.cuda.empty_cache()

# ---- 5. fine-tune step: BERT top 2 layers
bert = AutoModel.from_pretrained("bert-base-uncased").to(dev)
for q in bert.parameters(): q.requires_grad = False
train_p = []
for layer in bert.encoder.layer[-2:]:
    for q in layer.parameters(): q.requires_grad = True; train_p.append(q)
head = MLP().to(dev)
opt = torch.optim.AdamW([{"params": train_p, "lr": 2e-5}, {"params": head.parameters(), "lr": 1e-3}])
B = 64
ii = enc["input_ids"][:B].to(dev); am = enc["attention_mask"][:B].to(dev)
imf = torch.randn(B, 2048, device=dev); y = torch.rand(B, K, device=dev) ** 8
torch.cuda.reset_peak_memory_stats()


def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cls = bert(input_ids=ii, attention_mask=am).last_hidden_state[:, 0]
        logits = head(imf, cls)
    loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


for _ in range(5): step()
ts = [timeit(step)[0] for _ in range(30)]
R["finetune_top2"] = {"batch": B, "max_len": 32, "precision": "bf16 autocast, fp32 master weights",
                      "s_per_step_median": round(st.median(ts), 4), "s_per_step_mean": round(st.mean(ts), 4),
                      "peak_vram_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
                      "peak_vram_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
                      "steps_per_epoch": -(-N // B), "s_per_epoch": round(st.median(ts) * -(-N // B), 1)}

json.dump(R, open(HERE / "compute_results.json", "w"), indent=2)
print(json.dumps(R, indent=2))
