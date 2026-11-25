"""Evaluate the prior and every best-val checkpoint on one split.

Usage: python src/evaluate.py --split val
       python src/evaluate.py --split test     (once per final checkpoint)
"""
import argparse
import json
import platform
import sys

import numpy as np
import torch
import yaml

from datasets.vqa import A3, load_rows, load_vocab
from models import MODELS, prior
from train import DEV, predict, tensors
from utils.metrics import per_question, summarise

CONFIGS = ["unimodal_b", "unimodal_a", "fusion_simple"]


def examples(rows, preds, n=5):
    """Val questions (seed 261) where one of text-only / fusion scores 1 and the other 0."""
    (a_b, v_b), (a_d, v_d) = preds["unimodal_b"], preds["fusion_simple"]
    out = {}
    for name, mask in (("fusion_right_text_wrong", (v_d >= 1) & (v_b == 0)),
                       ("text_right_fusion_wrong", (v_b >= 1) & (v_d == 0))):
        idx = np.flatnonzero(mask)
        pick = np.sort(np.random.default_rng(261).choice(idx, min(n, len(idx)), replace=False))
        out[name] = {"count": int(len(idx)), "examples": [
            {"question_id": int(rows.question_id[i]), "image_id": int(rows.image_id[i]),
             "question": rows.question[i], "majority_answer": rows.mc_answer[i],
             "text_only": a_b[i], "fusion": a_d[i]} for i in pick]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--force", action="store_true", help="overwrite existing test results")
    args = ap.parse_args()
    out_path = A3 / "results" / f"eval_{args.split}.json"
    if args.split == "test" and out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists: test is evaluated once per final checkpoint")

    vocab = load_vocab()
    data = load_rows(["train", args.split])
    rows = data[args.split]
    image, text, target_t = tensors(rows, vocab)
    targets = target_t.cpu().numpy()

    res = {"split": args.split, "n_questions": len(rows),
           "oracle_vqa_acc": 100 * float(targets.max(1).mean()),
           "train_coverage_pct": 100 * float(data["train"].mc_answer.isin(set(vocab)).mean()),
           "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "python": platform.python_version()},
           "models": {}}

    index = {a: i for i, a in enumerate(vocab)}
    p = prior.predict(prior.fit(data["train"]), rows)
    _, v, e = per_question(np.array([index[a] for a in p]), rows, vocab, targets)
    res["models"]["prior"] = {"seeds": {"-": summarise(v, e, rows)}}

    preds261 = {}
    for name in CONFIGS:
        cfg = yaml.safe_load(open(A3 / "configs" / f"{name}.yaml"))
        entry = {"seeds": {}}
        for seed in cfg["seeds"]:
            ck = torch.load(A3 / "checkpoints" / f"{name}_s{seed}.pt", map_location=DEV, weights_only=True)
            model = MODELS[cfg["model"]](len(vocab), cfg["hidden"], cfg["dropout"]).to(DEV)
            model.load_state_dict(ck["state_dict"])
            idx = predict(model, image, text).cpu().numpy()
            ans, v, e = per_question(idx, rows, vocab, targets)
            entry["seeds"][str(seed)] = {**summarise(v, e, rows), "epoch": ck["epoch"]}
            if seed == 261:
                preds261[name] = (ans, v)
        res["models"][name] = entry

    for entry in res["models"].values():
        runs = list(entry["seeds"].values())
        keys = [k for k in runs[0] if k != "epoch"]
        entry["mean"] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        entry["std"] = {k: float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0 for k in keys}

    if args.split == "val":
        res["examples_seed261"] = examples(rows, preds261)
    out_path.write_text(json.dumps(res, indent=1))

    print(f"{args.split}: n={len(rows)} oracle={res['oracle_vqa_acc']:.2f}% "
          f"train coverage={res['train_coverage_pct']:.2f}%")
    cols = ["vqa_acc", "exact_match", "vqa_acc_yes/no", "vqa_acc_number", "vqa_acc_other", "vqa_acc_what_color"]
    print("model".ljust(15) + "".join(c.replace("vqa_acc_", "").rjust(16) for c in cols))
    for name, entry in res["models"].items():
        print(name.ljust(15) + "".join(f"{entry['mean'][c]:.2f}±{entry['std'][c]:.2f}".rjust(16) for c in cols))


if __name__ == "__main__":
    main()
