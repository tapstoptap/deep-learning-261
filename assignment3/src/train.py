"""Train one model configuration on cached features, for each seed. Uses train and val only.

Usage: python src/train.py --config configs/unimodal_a.yaml
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from datasets.vqa import A3, load_features, load_rows, load_vocab, soft_targets  # noqa: E402
from models import MODELS  # noqa: E402

DEV = "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def tensors(rows, vocab):
    image, text = load_features(rows)
    return [torch.from_numpy(x).to(DEV) for x in (image, text, soft_targets(rows, vocab))]


@torch.no_grad()
def predict(model, image, text, batch=1024):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([model(image[i:i + batch], text[i:i + batch]).argmax(1)
                          for i in range(0, len(image), batch)])


def run_seed(cfg, seed, train, val, n_classes):
    set_seed(seed)
    model = MODELS[cfg["model"]](n_classes, cfg["hidden"], cfg["dropout"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    gen = torch.Generator().manual_seed(seed)
    image, text, target = train
    n, bs = len(image), cfg["batch_size"]
    ckpt = A3 / "checkpoints" / f"{cfg['name']}_s{seed}.pt"
    history, best = [], None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        torch.cuda.synchronize()
        t0, total = time.perf_counter(), 0.0
        order = torch.randperm(n, generator=gen).to(DEV)
        for i in range(0, n, bs):
            j = order[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(image[j], text[j])
            # per-sample loss: summed over answers, averaged over the batch
            loss = loss_fn(logits.float(), target[j]) / len(j)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        pred = predict(model, val[0], val[1])
        val_acc = 100 * val[2][torch.arange(len(pred), device=DEV), pred].mean().item()
        history.append({"epoch": epoch, "train_loss": total / n, "val_vqa_acc": val_acc, "seconds": seconds})
        if best is None or val_acc > best["val_vqa_acc"]:
            best = history[-1]
            torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": seed,
                        "epoch": epoch, "val_vqa_acc": val_acc}, ckpt)
    log = {"config": cfg, "seed": seed, "params": sum(p.numel() for p in model.parameters()),
           "best_epoch": best["epoch"], "best_val_vqa_acc": best["val_vqa_acc"],
           "mean_epoch_seconds": float(np.mean([h["seconds"] for h in history[1:]])),
           "train_seconds_total": sum(h["seconds"] for h in history), "history": history}
    (A3 / "results" / "logs" / f"{cfg['name']}_s{seed}.json").write_text(json.dumps(log, indent=1))
    print(f"{cfg['name']} seed {seed}: best val VQA acc {best['val_vqa_acc']:.2f}% at epoch {best['epoch']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    (A3 / "checkpoints").mkdir(exist_ok=True)
    (A3 / "results" / "logs").mkdir(parents=True, exist_ok=True)
    vocab = load_vocab()
    data = load_rows(["train", "val"])
    train, val = tensors(data["train"], vocab), tensors(data["val"], vocab)
    for seed in cfg["seeds"]:
        run_seed(cfg, seed, train, val, len(vocab))


if __name__ == "__main__":
    main()
