"""Phase 4: split plan and leakage control (CO3133 A3, VQA v2).

Reads the Phase 3 subset id lists (does not re-sample), splits the 10k train2014 pool BY IMAGE into
train 9,000 / val 1,000 (seed 261, real near-duplicate pairs kept together), cleans train/val, runs
balance and leakage checks, gathers K-selection evidence (train vocab, val only; test untouched),
and writes ids, manifest and report. Deterministic: no timestamps in outputs.

Run (from assignment3/splits):  ../eda/.venv/bin/python -I make_splits.py
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EDA = HERE.parent / "eda"
sys.path.insert(0, str(EDA))
from vqa_eda import load_split, oracle_accuracy  # noqa: E402  (Phase 3 code, imported read-only)

SEED = 261
N_VAL_IMAGES = 1000
KS = (500, 1000, 2000, 3000)
TVD_LIMIT = 0.02
TOKENIZER = "bert-base-uncased"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tvd(a: pd.Series, b: pd.Series) -> float:
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())


def group_split(images, real_pairs, n_val, seed):
    """Split image ids by group (union-find over real duplicate pairs) into train/val with |val| == n_val."""
    parent = {i: i for i in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in real_pairs:
        parent[find(a)] = find(b)
    groups = {}
    for i in sorted(images):
        groups.setdefault(find(i), []).append(i)
    units = sorted(groups.values(), key=lambda g: g[0])
    random.Random(seed).shuffle(units)
    val, train = [], []
    for u in units:
        if len(val) + len(u) <= n_val:
            val.extend(u)
        else:
            train.extend(u)
    assert len(val) == n_val, "could not hit exact val size"
    return sorted(train), sorted(val), units


def clean(df):
    """Drop empty-majority questions and exact (image, question) duplicates with identical majority answer."""
    removed = []
    empty = df["mc_answer"] == ""
    removed += [(q, "empty_majority_answer") for q in df.loc[empty, "question_id"]]
    keep = df[~empty]
    dup = keep[keep.duplicated(["image_id", "question"], keep=False)]
    for _, g in dup.groupby(["image_id", "question"]):
        if g["mc_answer"].nunique() == 1:
            g = g.sort_values("question_id")
            removed += [(q, "duplicate_image_question_same_answer") for q in g["question_id"].iloc[1:]]
    rm = {q for q, _ in removed}
    return df[~df["question_id"].isin(rm)].copy(), removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub = EDA / "eda_outputs" / "subset"
    data = EDA / "data" / "vqa_v2"

    # ---- 1. inputs (Phase 3 lists, not re-sampled)
    pool_imgs = [int(x) for x in (sub / "train_pool_image_ids.txt").read_text().split()]
    test_imgs = sorted(int(x) for x in (sub / "test_image_ids.txt").read_text().split())
    pool_q = pd.read_csv(sub / "train_pool_question_ids.csv")
    test_q = pd.read_csv(sub / "test_question_ids.csv")
    assert len(pool_imgs) == 10000 and len(test_imgs) == 5000
    assert set(pool_q.image_id) == set(pool_imgs) and set(test_q.image_id) == set(test_imgs)
    review = pd.read_csv(sub / "near_dups_within_trainpool_review.csv")
    real_pairs = [(int(a), int(b)) for a, b, v in zip(review.image_id_a, review.image_id_b, review.verdict)
                  if v == "real_duplicate"]

    tr_df, _ = load_split(data, "train2014")
    te_df, _ = load_split(data, "val2014")
    pool = tr_df[tr_df.question_id.isin(pool_q.question_id)].copy()
    test = te_df[te_df.question_id.isin(test_q.question_id)].copy()
    assert len(pool) == len(pool_q) and len(test) == len(test_q)

    # ---- 2. split by image with group constraint
    train_imgs, val_imgs, units = group_split(pool_imgs, real_pairs, N_VAL_IMAGES, SEED)
    raw_train = pool[pool.image_id.isin(set(train_imgs))]
    raw_val = pool[pool.image_id.isin(set(val_imgs))]

    # ---- 3. cleaning (train/val only); test only counted
    train, rm_tr = clean(raw_train)
    val, rm_va = clean(raw_val)
    removed = pd.DataFrame([(q, r, "train") for q, r in rm_tr] + [(q, r, "val") for q, r in rm_va],
                           columns=["question_id", "reason", "split"]).sort_values("question_id")
    test_empty = int((test.mc_answer == "").sum())
    tdup = test[test.duplicated(["image_id", "question"], keep=False)]
    tgrp = tdup.groupby(["image_id", "question"]).mc_answer.nunique()
    test_dup_pairs, test_dup_same = len(tgrp), int((tgrp == 1).sum())

    # ---- 4. balance (post-cleaning)
    top20 = [t for t, _ in Counter(train.question_type).most_common(20)]
    qt = lambda s: s.where(s.isin(top20), "<other>")  # noqa: E731
    tvd_at = tvd(train.answer_type, val.answer_type)
    tvd_qt = tvd(qt(train.question_type), qt(val.question_type))

    # ---- 5. leakage checks
    ti, vi, ei = set(train_imgs), set(val_imgs), set(test_imgs)
    comp = {}
    for u in units:
        for i in u:
            comp[i] = id(u)
    cross = [u for u in units if len({("t" if i in ti else "v" if i in vi else "x") for i in u}) > 1]
    vocab_counter = Counter(train.mc_answer)  # TRAIN rows only
    ranked = sorted(vocab_counter, key=lambda a: (-vocab_counter[a], a))
    checks = [
        ("Image ids disjoint across train/val/test", not (ti & vi) and not (ti & ei) and not (vi & ei)),
        ("Question ids disjoint across train/val/test",
         not (set(train.question_id) & set(val.question_id)) and not (set(train.question_id) & set(test.question_id))
         and not (set(val.question_id) & set(test.question_id))),
        ("No near-dup group (real pairs) crosses splits", not cross),
        ("Answer vocab built from TRAIN rows only (no val/test question ids used)",
         set(train.image_id) <= ti and not (set(train.question_id) & (set(val.question_id) | set(test.question_id)))),
        (f"Train/val image counts are 9000/1000 (got {len(train_imgs)}/{len(val_imgs)})",
         len(train_imgs) == 9000 and len(val_imgs) == 1000),
        (f"TVD answer_type <= {TVD_LIMIT} ({tvd_at:.4f})", tvd_at <= TVD_LIMIT),
        (f"TVD top-20 question_type <= {TVD_LIMIT} ({tvd_qt:.4f})", tvd_qt <= TVD_LIMIT),
    ]

    # ---- 6. choosing K (train vocab; val only)
    ktab = []
    for k in KS:
        v = set(ranked[:k])
        orc = float(np.mean([oracle_accuracy(g, v) for g in val["answers"]])) * 100
        ktab.append({"K": k,
                     "val_oracle_vqa_acc_pct": round(orc, 2),
                     "train_coverage_pct": round(100 * train.mc_answer.isin(v).mean(), 2),
                     "val_coverage_pct": round(100 * val.mc_answer.isin(v).mean(), 2),
                     "kth_answer": ranked[k - 1] if k <= len(ranked) else None,
                     "kth_answer_train_count": int(vocab_counter[ranked[k - 1]]) if k <= len(ranked) else None})
    n_vocab = len(ranked)
    tok = None
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(TOKENIZER)
        L = np.array([len(tk(q, add_special_tokens=True)["input_ids"]) for q in train.question])
        tok = {"tokenizer": TOKENIZER, "includes_special_tokens": True, "n": int(len(L)),
               "p95": float(np.percentile(L, 95)), "p99": float(np.percentile(L, 99)), "max": int(L.max())}
    except Exception as e:  # noqa: BLE001
        tok = {"error": repr(e)}

    # ---- 7. questions per image
    qpi = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        c = d.image_id.value_counts()
        c = c.sort_index().sort_values(ascending=False, kind="stable")
        qpi[name] = {"max": int(c.iloc[0]), "mean": round(float(c.mean()), 2),
                     "top5": [(int(i), int(n)) for i, n in c.head(5).items()]}

    # ---- 8. write outputs
    def write_ids(name, imgs, d):
        (out / f"{name}_image_ids.txt").write_text("".join(f"{i}\n" for i in sorted(imgs)))
        d[["question_id", "image_id"]].sort_values("question_id").to_csv(out / f"{name}_question_ids.csv", index=False)
    write_ids("train", train_imgs, train)
    write_ids("val", val_imgs, val)
    write_ids("test", test_imgs, test)
    removed.to_csv(out / "removed.csv", index=False)
    files = [f"{n}_{k}" for n in ("train", "val", "test") for k in ("image_ids.txt", "question_ids.csv")]
    hashes = {f: sha256_file(out / f) for f in files}
    hashes["removed.csv"] = sha256_file(out / "removed.csv")

    counts = {
        "train": {"images": len(train_imgs), "questions_raw": len(raw_train), "removed": len(rm_tr), "questions": len(train)},
        "val": {"images": len(val_imgs), "questions_raw": len(raw_val), "removed": len(rm_va), "questions": len(val)},
        "test": {"images": len(test_imgs), "questions_raw": len(test), "removed": 0, "questions": len(test),
                 "empty_majority_answers_not_filtered": test_empty,
                 "exact_dup_image_question_pairs_not_filtered": test_dup_pairs,
                 "of_which_same_majority_answer": test_dup_same},
    }
    manifest = {
        "seed": SEED, "phase3_inputs_sha256": {p: sha256_file(sub / p) for p in (
            "train_pool_image_ids.txt", "train_pool_question_ids.csv", "test_image_ids.txt",
            "test_question_ids.csv", "near_dups_within_trainpool_review.csv")},
        "rules": {
            "split": "by image from the 10k train2014 pool; 9000 train / 1000 val; units = connected "
                     "components of real_duplicate pairs, shuffled with random.Random(261), greedily added to "
                     "val while it fits",
            "real_near_dup_pairs_grouped": real_pairs,
            "cleaning_train_val": ["drop empty normalized majority answer",
                                   "exact (image_id, question) duplicates: identical majority answer -> keep lowest "
                                   "question_id; different majority answers -> keep both"],
            "cleaning_test": "none (counts reported only)",
            "vocab": "train rows only, sorted by (-count, answer)",
        },
        "counts": counts, "sha256": hashes,
        "tvd": {"answer_type": round(tvd_at, 5), "top20_question_type": round(tvd_qt, 5), "limit": TVD_LIMIT},
        "train_vocab_size": n_vocab,
        "checks": {n: bool(ok) for n, ok in checks},
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # ---- report
    L = ["# Phase 4 split report", "", f"Seed {SEED}. Test set is never used for any choice below.", "",
         "## Counts", "", "| split | images | questions (raw) | removed | questions (final) |", "|---|---|---|---|---|"]
    for n in ("train", "val", "test"):
        c = counts[n]
        L.append(f"| {n} | {c['images']:,} | {c['questions_raw']:,} | {c['removed']} | {c['questions']:,} |")
    L += ["", f"Test not filtered: {test_empty} empty-majority questions, {test_dup_pairs} exact duplicate "
          f"(image, question) pairs ({test_dup_same} with identical majority answer). Report-only.",
          "", "Removed (train/val): " + ", ".join(f"{r}={n}" for r, n in
                                                  removed.reason.value_counts().sort_index().items()) + ".",
          "", "## Balance (train vs val, after cleaning)", "", "| measure | TVD | limit |", "|---|---|---|",
          f"| answer_type | {tvd_at:.4f} | {TVD_LIMIT} |", f"| top-20 question_type (+other) | {tvd_qt:.4f} | {TVD_LIMIT} |",
          "", "## Choosing K (train vocab, val only)", "", f"Train vocab size: {n_vocab:,} distinct answers.", "",
          "| K | val oracle VQA acc % | train coverage % | val coverage % | K-th answer | its train count |",
          "|---|---|---|---|---|---|"]
    for r in ktab:
        L.append(f"| {r['K']} | {r['val_oracle_vqa_acc_pct']} | {r['train_coverage_pct']} | {r['val_coverage_pct']} | "
                 f"{r['kth_answer']} | {r['kth_answer_train_count']} |")
    L += ["", "BERT question length (train only, bert-base-uncased, incl. [CLS]/[SEP]): " +
          (", ".join(f"{k}={tok[k]}" for k in ("p95", "p99", "max")) if "p95" in tok else str(tok)),
          "", "## Questions per image", "", "| split | max | mean | top-5 (image: n) |", "|---|---|---|---|"]
    for n in ("train", "val", "test"):
        q = qpi[n]
        L.append(f"| {n} | {q['max']} | {q['mean']} | " + ", ".join(f"{i}: {c}" for i, c in q["top5"]) + " |")
    L += ["", "## Checklist", ""] + [f"- [{'x' if ok else ' '}] **{'PASS' if ok else 'FAIL'}** {n}" for n, ok in checks]
    L += ["", "## SHA-256", ""] + [f"- `{k}`: `{v}`" for k, v in sorted(hashes.items())]
    (out / "split_report.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))
    print("\nTOKEN", tok)
    print("KTAB", json.dumps(ktab))
    print("real pairs", real_pairs, "groups crossing:", len(cross))
    print("removed", removed.to_dict("records"))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
