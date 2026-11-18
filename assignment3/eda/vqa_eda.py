"""
CO3133 Assignment 3 - M1 preliminary EDA for VQA v2 (answer-classification framing).

Covers the Dataset Proposal / report requirements in the handbook (Sec. 27, 28.1, 31):
pair counts, missing-modality rate, text length, pair quality, distribution analysis,
leakage-aware subset, plus checks specific to VQA-as-classification.

Sections written to summary.json
  meta                         run time, Python/package versions, sha256 + size of every input file
  full_train2014 / full_val2014  statistics on the complete official splits
  full_image_overlap_train_val   images present in both official splits (expected 0)
  subset_train_pool / subset_test  same statistics on the seeded image-level subset
  subset_representativeness    total-variation distance subset vs full (answer type,
                               question type, top-100 answers)
  answer_vocab_from_train_pool vocabulary options (min-count and top-K) with coverage and
                               the oracle VQA-accuracy ceiling each one allows on the test subset
  question_text_overlap        exact question strings shared by train pool and test
  language_prior               "no image" reference: global mode and mode per question_type
  token_lengths (optional)     question lengths in tokens of a chosen HF tokenizer
  subset_images (optional)     missing / corrupt / sizes / formats / near-duplicates
  clip_pair_similarity (opt.)  CLIP image-text similarity as a weak mismatch signal

Usage
  python vqa_eda.py --data-dir data/vqa_v2 --out-dir eda_outputs
  python vqa_eda.py --data-dir data/vqa_v2 --out-dir eda_outputs --images
  python vqa_eda.py --data-dir data/vqa_v2 --out-dir eda_outputs --images \
      --tokenizer bert-base-uncased --clip openai/clip-vit-base-patch32

Expected files in --data-dir (unzipped official VQA v2 "Balanced Real Images" files):
  v2_OpenEnded_mscoco_train2014_questions.json
  v2_OpenEnded_mscoco_val2014_questions.json
  v2_mscoco_train2014_annotations.json
  v2_mscoco_val2014_annotations.json
Images (only with --images): --data-dir/images/{train2014,val2014}/COCO_<split>_<id:012d>.jpg.
Missing ones are downloaded from images.cocodataset.org unless --no-download is given.
"""

import argparse
import datetime as _dt
import functools
import hashlib
import json
import math
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ("train2014", "val2014")
ANSWER_TYPES = ("yes/no", "number", "other")

# ---------------------------------------------------------------------------
# Answer normalisation - same rules as the official VQA evaluation script
# (punctuation handling, number words -> digits, articles removed, contractions)
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "its": "it's",
    "lets": "let's", "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodyd": "somebody'd", "someoned": "someone'd",
    "somethingd": "something'd", "thats": "that's", "thered": "there'd",
    "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed": "we'd",
    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "yall": "y'all", "youd": "you'd", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


@functools.lru_cache(maxsize=None)
def normalize_answer(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    out = ans
    for p in _PUNCT:
        if (p + " " in ans or " " + p in ans) or re.search(_COMMA_STRIP, ans) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    words = []
    for w in out.split():
        w = _NUMBERS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_accuracy(pred: str, gt_answers) -> float:
    """Official VQA accuracy: mean over the 10 leave-one-out subsets of
    min(1, #matches among the other 9 / 3)."""
    gts = list(gt_answers)
    n = len(gts)
    if n == 0:
        return 0.0
    m = sum(a == pred for a in gts)
    # annotators who gave `pred` see m-1 matches among the others; the rest see m
    return (m * min(1.0, (m - 1) / 3.0) + (n - m) * min(1.0, m / 3.0)) / n


def oracle_accuracy(gts, vocab) -> float:
    """Best VQA accuracy a classifier restricted to `vocab` could get on this question."""
    cands = [a for a in set(gts) if a in vocab]
    return max((vqa_accuracy(a, gts) for a in cands), default=0.0)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "matplotlib", "PIL", "imagehash", "requests",
                 "transformers", "torch"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def input_files(data_dir: Path, split: str):
    return (data_dir / f"v2_OpenEnded_mscoco_{split}_questions.json",
            data_dir / f"v2_mscoco_{split}_annotations.json")


def load_split(data_dir: Path, split: str):
    qf, af = input_files(data_dir, split)
    for f in (qf, af):
        if not f.exists():
            sys.exit(f"Missing file: {f}")
    with open(qf, encoding="utf-8") as fh:
        qjson = json.load(fh)
    with open(af, encoding="utf-8") as fh:
        ajson = json.load(fh)
    header = {k: qjson.get(k) for k in ("info", "data_type", "data_subtype", "task_type")}
    header["license"] = qjson.get("license") or ajson.get("license")
    qdf = pd.DataFrame(qjson["questions"])[["question_id", "image_id", "question"]]
    rows = []
    for a in ajson["annotations"]:
        raw = [x["answer"] for x in a["answers"]]
        rows.append({
            "question_id": a["question_id"],
            "ann_image_id": a.get("image_id"),
            "question_type": a.get("question_type", ""),
            "answer_type": a.get("answer_type", ""),
            "mc_answer": normalize_answer(a.get("multiple_choice_answer", "")),
            "answers": [normalize_answer(x) for x in raw],
            "confidences": [x.get("answer_confidence", "") for x in a["answers"]],
        })
    adf = pd.DataFrame(rows)
    df = qdf.merge(adf, on="question_id", how="outer", indicator=True)
    df["split"] = split
    return df, header


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def describe(values) -> dict:
    s = pd.Series(values).dropna()
    if s.empty:
        return {}
    return {"n": int(len(s)), "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
            "median": float(s.median()), "p95": float(s.quantile(0.95)),
            "min": float(s.min()), "max": float(s.max())}


def pct(x, total) -> float:
    return round(100.0 * x / total, 2) if total else 0.0


def top_votes(xs) -> int:
    return Counter(xs).most_common(1)[0][1] if len(xs) else 0


def split_stats(df: pd.DataFrame, top_ks=(100, 500, 1000, 2000, 3000)) -> dict:
    ok = df[df["_merge"] == "both"].copy()
    total = len(ok)
    if total == 0:
        return {"questions_with_annotations": 0}
    words = ok["question"].str.split()
    ok["q_len"] = words.str.len()
    ok["n_answers"] = ok["answers"].str.len()
    ok["agree_top"] = ok["answers"].apply(top_votes)
    ok["n_unique_ans"] = ok["answers"].apply(lambda xs: len(set(xs)))
    ok["ans_len"] = ok["mc_answer"].str.split().str.len()
    ok["mc_in_question"] = [
        bool(a) and len(a) > 1 and re.search(rf"\b{re.escape(a)}\b", q.lower()) is not None
        for a, q in zip(ok["mc_answer"], ok["question"])]
    ok["choice_question"] = ok["question"].str.lower().str.contains(r"\bor\b", regex=True)

    q_per_img = ok.groupby("image_id").size()
    ans_counts = Counter(ok["mc_answer"])
    counts_sorted = np.array(sorted(ans_counts.values(), reverse=True), dtype=float)
    p = counts_sorted / counts_sorted.sum()
    entropy = float(-(p * np.log(p)).sum())

    coverage = {}
    for k in top_ks:
        coverage[f"top{k}"] = pct(counts_sorted[:k].sum(), total)

    conf = Counter(c for cs in ok["confidences"] for c in cs)
    agree_by_type = {}
    for at, g in ok.groupby("answer_type"):
        agree_by_type[at] = {
            "mean_votes_for_top_answer": round(float(g["agree_top"].mean()), 3),
            "pct_top_answer_>=3_votes": pct((g["agree_top"] >= 3).sum(), len(g)),
        }
    qtypes = ok["question_type"].value_counts()
    mismatched_img = int((ok["ann_image_id"].notna() &
                          (ok["ann_image_id"] != ok["image_id"])).sum())

    return {
        "questions_with_annotations": int(total),
        "questions_missing_annotation": int((df["_merge"] == "left_only").sum()),
        "annotations_missing_question": int((df["_merge"] == "right_only").sum()),
        "question_annotation_image_id_mismatch": mismatched_img,
        "images": int(ok["image_id"].nunique()),
        "questions_per_image": describe(q_per_img),
        "text_hygiene": {
            "empty_questions": int((ok["question"].str.strip() == "").sum()),
            "questions_under_3_words": int((ok["q_len"] < 3).sum()),
            "questions_not_ending_with_?": int((~ok["question"].str.strip().str.endswith("?")).sum()),
            "questions_with_non_ascii": int(ok["question"].map(lambda s: not s.isascii()).sum()),
            "empty_majority_answers": int((ok["mc_answer"] == "").sum()),
            "questions_without_10_answers": int((ok["n_answers"] != 10).sum()),
        },
        "question_length_words": describe(ok["q_len"]),
        "answer_type_pct": {k: pct(v, total) for k, v in ok["answer_type"].value_counts().items()},
        "question_types": {"count": int(len(qtypes)),
                           "top15_pct": {k: pct(v, total) for k, v in qtypes.head(15).items()}},
        "majority_answer_length_words": describe(ok["ans_len"]),
        "pct_multi_word_majority_answers": pct((ok["ans_len"] > 1).sum(), total),
        "answer_distribution": {
            "unique_majority_answers": len(ans_counts),
            "unique_answers_incl_all_10": len({a for xs in ok["answers"] for a in xs}),
            "top1_answer_share_pct": pct(counts_sorted[0], total),
            "top10_answers_share_pct": pct(counts_sorted[:10].sum(), total),
            "normalized_entropy": round(entropy / math.log(len(counts_sorted)), 4)
            if len(counts_sorted) > 1 else 0.0,
            "answers_seen_once": int((counts_sorted == 1).sum()),
            "coverage_of_questions_by_topK_pct": coverage,
        },
        "annotator_agreement": {
            "mean_votes_for_top_answer": round(float(ok["agree_top"].mean()), 3),
            "pct_top_answer_>=3_votes": pct((ok["agree_top"] >= 3).sum(), total),
            "pct_all_10_agree": pct((ok["agree_top"] == 10).sum(), total),
            "pct_no_answer_>=2_votes": pct((ok["agree_top"] < 2).sum(), total),
            "mean_unique_answers": round(float(ok["n_unique_ans"].mean()), 3),
            "by_answer_type": agree_by_type,
        },
        "answer_confidence_pct": {k: pct(v, sum(conf.values())) for k, v in conf.items()},
        "leakage_in_text": {
            "pct_majority_answer_in_question": pct(ok["mc_in_question"].sum(), total),
            "pct_choice_questions_with_or": pct(ok["choice_question"].sum(), total),
        },
        "duplicates": {
            "duplicate_question_ids": int(ok["question_id"].duplicated().sum()),
            "duplicate_image_question_pairs": int(ok.duplicated(subset=["image_id", "question"]).sum()),
        },
        "top20_answers": ans_counts.most_common(20),
    }


def tvd(a: pd.Series, b: pd.Series, keys=None) -> float:
    """Total-variation distance between two categorical distributions (0 = identical, 1 = disjoint)."""
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    if keys is not None:
        pa = pa.reindex(keys, fill_value=0)
        pb = pb.reindex(keys, fill_value=0)
    idx = pa.index.union(pb.index)
    return round(0.5 * float((pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum()), 4)


def representativeness(full: pd.DataFrame, sub: pd.DataFrame) -> dict:
    full = full[full["_merge"] == "both"]
    top100 = full["mc_answer"].value_counts().head(100).index
    return {
        "tvd_answer_type": tvd(full["answer_type"], sub["answer_type"]),
        "tvd_question_type": tvd(full["question_type"], sub["question_type"]),
        "tvd_top100_answers": tvd(full["mc_answer"].where(full["mc_answer"].isin(top100), "<other>"),
                                  sub["mc_answer"].where(sub["mc_answer"].isin(top100), "<other>")),
        "mean_q_len_full_vs_subset": [round(float(full["question"].str.split().str.len().mean()), 3),
                                      round(float(sub["question"].str.split().str.len().mean()), 3)],
    }


def sample_images(df: pd.DataFrame, n: int, seed: int) -> list:
    ids = sorted(df.loc[df["_merge"] == "both", "image_id"].unique().tolist())
    if n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


def vocab_options(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    counts = Counter(train["mc_answer"])
    ranked = [a for a, _ in counts.most_common()]
    options = {}
    defs = [(f"min_count_{m}", {a for a, c in counts.items() if c >= m}) for m in (3, 5, 9)]
    defs += [(f"top_{k}", set(ranked[:k])) for k in (500, 1000, 2000, 3000)]
    for name, v in defs:
        options[name] = {
            "vocab_size": len(v),
            "train_pool_coverage_pct": pct(train["mc_answer"].isin(v).sum(), len(train)),
            "test_coverage_pct": pct(test["mc_answer"].isin(v).sum(), len(test)),
            "test_oracle_vqa_accuracy": round(100 * float(np.mean(
                [oracle_accuracy(g, v) for g in test["answers"]])), 2),
        }
    options["_note"] = ("oracle = best VQA accuracy a perfect classifier restricted to this "
                        "vocabulary could reach on the test subset (upper bound).")
    return options


def language_prior(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    global_mode = Counter(train["mc_answer"]).most_common(1)[0][0]
    by_type = train.groupby("question_type")["mc_answer"].agg(
        lambda s: Counter(s).most_common(1)[0][0]).to_dict()
    accs_global, accs_type = [], []
    per_atype = defaultdict(list)
    for qt, at, gts in zip(test["question_type"], test["answer_type"], test["answers"]):
        accs_global.append(vqa_accuracy(global_mode, gts))
        a = vqa_accuracy(by_type.get(qt, global_mode), gts)
        accs_type.append(a)
        per_atype[at].append(a)
    return {
        "global_mode_answer": global_mode,
        "acc_always_global_mode": round(100 * float(np.mean(accs_global)), 2),
        "acc_mode_per_question_type": round(100 * float(np.mean(accs_type)), 2),
        "acc_mode_per_question_type_by_answer_type": {
            k: round(100 * float(np.mean(v)), 2) for k, v in per_atype.items()},
        "_note": "No image and no learned model: a reference floor, not a baseline result.",
    }


def question_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    tr_q = set(train["question"].str.lower().str.strip())
    te_q = test["question"].str.lower().str.strip()
    tr_qa = set(zip(train["question"].str.lower().str.strip(), train["mc_answer"]))
    te_qa = list(zip(te_q, test["mc_answer"]))
    return {
        "pct_test_questions_whose_text_appears_in_train_pool": pct(te_q.isin(tr_q).sum(), len(test)),
        "pct_test_question_answer_pairs_seen_in_train_pool": pct(sum(x in tr_qa for x in te_qa), len(test)),
        "_note": "Same question text on different images is normal in VQA (e.g. 'What color is the bus?'); "
                 "images, not question strings, are the leakage unit.",
    }


def token_lengths(questions: pd.Series, tokenizer_name: str) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    lens = [len(tok(q, add_special_tokens=True)["input_ids"]) for q in questions.tolist()]
    arr = np.array(lens)
    return {"tokenizer": tokenizer_name, **describe(arr),
            "pct_over_16": pct((arr > 16).sum(), len(arr)),
            "pct_over_24": pct((arr > 24).sum(), len(arr)),
            "pct_over_32": pct((arr > 32).sum(), len(arr))}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def coco_name(split: str, image_id: int) -> str:
    return f"COCO_{split}_{int(image_id):012d}.jpg"


def ensure_images(data_dir: Path, split: str, ids: list, download: bool, workers: int) -> Path:
    img_dir = data_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in ids if not (img_dir / coco_name(split, i)).exists()]
    if missing and download:
        import requests
        sess = requests.Session()

        def fetch(i):
            url = f"http://images.cocodataset.org/{split}/{coco_name(split, i)}"
            for _ in range(3):
                try:
                    r = sess.get(url, timeout=30)
                    if r.status_code == 200:
                        tmp = img_dir / (coco_name(split, i) + ".part")
                        tmp.write_bytes(r.content)
                        tmp.replace(img_dir / coco_name(split, i))
                        return True
                    if r.status_code == 404:
                        return False
                except Exception:
                    pass
            return False

        print(f"[{split}] downloading {len(missing)} images with {workers} workers ...", flush=True)
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            for k, ok in enumerate(ex.map(fetch, missing), 1):
                done += ok
                if k % 500 == 0:
                    print(f"  {k}/{len(missing)}", flush=True)
        print(f"[{split}] downloaded {done}/{len(missing)}", flush=True)
    return img_dir


def image_checks(img_dir: Path, split: str, ids: list) -> pd.DataFrame:
    from PIL import Image
    import imagehash
    rows = []
    for i in ids:
        p = img_dir / coco_name(split, i)
        row = {"image_id": i, "split": split, "exists": p.exists(), "ok": False,
               "width": np.nan, "height": np.nan, "mode": None, "format": None,
               "file_kb": np.nan, "phash": None}
        if p.exists():
            row["file_kb"] = round(p.stat().st_size / 1024, 1)
            try:
                with Image.open(p) as im:
                    fmt = im.format
                    im.load()
                    row.update(ok=True, width=im.width, height=im.height, mode=im.mode, format=fmt,
                               phash=str(imagehash.phash(im.convert("RGB"))))
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def _hash_bits(df: pd.DataFrame):
    import imagehash
    keep = df[df["phash"].map(lambda h: isinstance(h, str))]
    ids = keep["image_id"].astype(int).tolist()
    bits = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in keep["phash"]], dtype=bool)
    return ids, bits.reshape(len(ids), -1) if len(ids) else np.zeros((0, 64), bool)


def near_dups(a: pd.DataFrame, b: pd.DataFrame, max_dist: int, same_set: bool) -> list:
    ida, ba = _hash_bits(a)
    idb, bb = _hash_bits(b)
    pairs = []
    for k in range(len(ida)):
        if len(idb) == 0:
            break
        d = (bb != ba[k]).sum(axis=1)
        hits = np.where(d <= max_dist)[0]
        for j in hits:
            if same_set and j <= k:  # each unordered pair once, no self-pairs
                continue
            pairs.append((ida[k], idb[j], int(d[j])))
    return pairs


def clip_similarity(df: pd.DataFrame, data_dir: Path, model_name: str, batch: int) -> pd.DataFrame:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    texts = [f"{q} {a}" for q, a in zip(df["question"], df["mc_answer"])]
    paths = [data_dir / "images" / s / coco_name(s, i) for s, i in zip(df["split"], df["image_id"])]
    sims = []
    with torch.no_grad():
        for s in range(0, len(df), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
            inp = proc(text=texts[s:s + batch], images=imgs, return_tensors="pt",
                       padding=True, truncation=True).to(dev)
            out = model(**inp)
            ie = torch.nn.functional.normalize(out.image_embeds, dim=-1)
            te = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            sims.extend((ie * te).sum(-1).cpu().tolist())
    res = df[["question_id", "image_id", "split", "question", "mc_answer"]].copy()
    res["clip_sim"] = sims
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig_dir: Path, name: str):
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=150)
    plt.close()


def _grid(rows, data_dir: Path, title_fn, path: Path, ncols=3):
    from PIL import Image
    rows = list(rows)
    if not rows:
        return
    nrows = math.ceil(len(rows) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for axi, r in zip(axs.flat, rows):
        p = data_dir / "images" / r.split / coco_name(r.split, r.image_id)
        try:
            with Image.open(p) as im:
                axi.imshow(im.convert("RGB"))
        except Exception:
            axi.text(0.5, 0.5, "image missing", ha="center")
        axi.set_title(title_fn(r), fontsize=8)
        axi.axis("off")
    for axi in list(axs.flat)[len(rows):]:
        axi.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def save_figures(sub_tr, sub_te, img_df, clip_df, out: Path, data_dir: Path, seed: int):
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    both = pd.concat([sub_tr.assign(subset="train pool"), sub_te.assign(subset="test")])

    # F1 answer types per subset
    ct = (both.groupby("subset")["answer_type"].value_counts(normalize=True)
          .mul(100).unstack(fill_value=0))
    ct.plot(kind="bar", figsize=(6, 4))
    plt.ylabel("% of questions"); plt.title("Answer type distribution"); plt.xticks(rotation=0)
    _save(figs, "F1_answer_types.png")

    # F2 top-30 majority answers, coloured by answer type
    top = Counter(sub_tr["mc_answer"]).most_common(30)
    atype = sub_tr.groupby("mc_answer")["answer_type"].agg(lambda s: s.mode().iat[0])
    colours = {"yes/no": "tab:blue", "number": "tab:orange", "other": "tab:green"}
    plt.figure(figsize=(10, 4))
    plt.bar([a for a, _ in top], [c for _, c in top],
            color=[colours.get(atype.get(a), "gray") for a, _ in top])
    plt.xticks(rotation=70, ha="right"); plt.ylabel("questions")
    plt.title("Top-30 majority answers (train pool)")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=c, label=t) for t, c in colours.items()], title="answer type")
    _save(figs, "F2_top30_answers.png")

    # F3 cumulative coverage of questions vs vocabulary size K
    counts = np.array(sorted(Counter(sub_tr["mc_answer"]).values(), reverse=True), dtype=float)
    cum = 100 * np.cumsum(counts) / counts.sum()
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum)
    plt.xscale("log"); plt.grid(True, which="both", alpha=0.3)
    for k in (100, 1000, 3000):
        if k <= len(cum):
            plt.axvline(k, ls="--", c="gray")
            plt.text(k * 0.92, 35, f"K={k}: {cum[k - 1]:.1f}%", rotation=90, fontsize=8, ha="right")
    plt.xlabel("answer vocabulary size K (log)"); plt.ylabel("% of train-pool questions covered")
    plt.title("Answer coverage vs vocabulary size")
    _save(figs, "F3_answer_coverage_curve.png")

    # F4 question types
    qt = sub_tr["question_type"].value_counts(normalize=True).head(20).mul(100)[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(qt.index, qt.values); plt.xlabel("% of train-pool questions")
    plt.title("Top-20 question prefixes (overlapping, not exclusive types)")
    _save(figs, "F4_question_types.png")

    # F5 question length, questions per image, answer length
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))
    axs[0].hist(both["question"].str.split().str.len(), bins=range(1, 26))
    axs[0].set_xlabel("words per question"); axs[0].set_title("Question length")
    axs[1].hist(both.groupby("image_id").size(), bins=range(1, 16))
    axs[1].set_xlabel("questions per image"); axs[1].set_title("Questions per image")
    axs[2].hist(both["mc_answer"].str.split().str.len(), bins=range(0, 8))
    axs[2].set_xlabel("words per majority answer"); axs[2].set_title("Answer length")
    for a in axs:
        a.set_ylabel("count")
    _save(figs, "F5_lengths.png")

    # F6 annotator agreement by answer type
    agree = both.assign(votes=both["answers"].apply(top_votes))
    plt.figure(figsize=(7, 4))
    data = [agree.loc[agree["answer_type"] == t, "votes"] for t in ANSWER_TYPES]
    plt.hist(data, bins=np.arange(0.5, 11.5, 1), stacked=True, label=list(ANSWER_TYPES))
    plt.axvline(2.5, ls="--", c="k")
    plt.xlabel("votes for the most common answer (of 10)"); plt.ylabel("questions")
    plt.title("Annotator agreement (right of dashed line: full VQA accuracy reachable)")
    plt.legend()
    _save(figs, "F6_annotator_agreement.png")

    rng_state = seed
    if img_df is not None and img_df["ok"].any():
        okimg = img_df[img_df["ok"]]
        # F7 image sizes and aspect ratios
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].scatter(okimg["width"], okimg["height"], s=3, alpha=0.3)
        axs[0].set_xlabel("width (px)"); axs[0].set_ylabel("height (px)"); axs[0].set_title("Image sizes")
        axs[1].hist(okimg["width"] / okimg["height"], bins=40)
        axs[1].set_xlabel("aspect ratio (w/h)"); axs[1].set_title("Aspect ratios")
        _save(figs, "F7_image_sizes.png")

        ok_ids = set(okimg["image_id"])
        pool = sub_tr[sub_tr["image_id"].isin(ok_ids)]

        def title(r):
            top3 = ", ".join(f"{a}({c})" for a, c in Counter(r.answers).most_common(3))
            return f"Q: {r.question}\nA: {top3}"

        # F8 random example pairs
        _grid(pool.sample(min(9, len(pool)), random_state=rng_state).itertuples(),
              data_dir, title, figs / "F8_example_pairs.png")
        # F9 low-agreement examples (pair-quality problems)
        low = pool[pool["answers"].apply(top_votes) < 3]
        _grid(low.sample(min(9, len(low)), random_state=rng_state).itertuples(),
              data_dir, title, figs / "F9_low_agreement_examples.png")

    if clip_df is not None and len(clip_df):
        plt.figure(figsize=(6, 4))
        plt.hist(clip_df["clip_sim"], bins=50)
        plt.xlabel("CLIP cosine(image, question + answer)"); plt.ylabel("pairs")
        plt.title("CLIP pair similarity")
        _save(figs, "F10_clip_similarity.png")
        lowest = clip_df.nsmallest(9, "clip_sim")
        _grid(lowest.itertuples(), data_dir,
              lambda r: f"sim={r.clip_sim:.3f}\nQ: {r.question}\nA: {r.mc_answer}",
              figs / "F11_lowest_clip_pairs.png")


# ---------------------------------------------------------------------------
# Proposal-ready tables
# ---------------------------------------------------------------------------
def proposal_tables(s: dict) -> str:
    def g(d, *keys, default="n/a"):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    L = ["# Proposal tables (generated by vqa_eda.py - copy, do not retype)", ""]
    L += ["## Table 1 - Official VQA v2 splits used", "",
          "| Split | Images | Questions | Answers per question | Questions per image (mean) |",
          "|---|---|---|---|---|"]
    for name in ("full_train2014", "full_val2014"):
        d = s[name]
        L.append(f"| {name.replace('full_', '')} | {d['images']:,} | {d['questions_with_annotations']:,} | 10 | "
                 f"{g(d, 'questions_per_image', 'mean')} |")
    L += ["", "## Table 2 - Image-level subset", "",
          "| Subset | Source split | Images | Questions | yes/no % | number % | other % |",
          "|---|---|---|---|---|---|---|"]
    for name, src in (("subset_train_pool", "train2014"), ("subset_test", "val2014")):
        d = s[name]
        at = d["answer_type_pct"]
        L.append(f"| {name.replace('subset_', '')} | {src} | {d['images']:,} | {d['questions_with_annotations']:,} | "
                 f"{at.get('yes/no', 0)} | {at.get('number', 0)} | {at.get('other', 0)} |")
    L += ["", "## Table 3 - Pair quality and missing data (subset)", "",
          "| Check | Train pool | Test |", "|---|---|---|"]
    rows = [
        ("Questions with >=3 votes for top answer (%)", ("annotator_agreement", "pct_top_answer_>=3_votes")),
        ("All 10 annotators agree (%)", ("annotator_agreement", "pct_all_10_agree")),
        ("Mean votes for top answer", ("annotator_agreement", "mean_votes_for_top_answer")),
        ("Majority answer appears in question (%)", ("leakage_in_text", "pct_majority_answer_in_question")),
        ("Choice questions containing 'or' (%)", ("leakage_in_text", "pct_choice_questions_with_or")),
        ("Duplicate (image, question) pairs", ("duplicates", "duplicate_image_question_pairs")),
        ("Empty questions", ("text_hygiene", "empty_questions")),
        ("Questions without 10 answers", ("text_hygiene", "questions_without_10_answers")),
        ("Question length, words (mean / p95 / max)", None),
    ]
    for label, path in rows:
        if path is None:
            vals = [f"{g(s[n], 'question_length_words', 'mean')} / {g(s[n], 'question_length_words', 'p95')} / "
                    f"{g(s[n], 'question_length_words', 'max')}" for n in ("subset_train_pool", "subset_test")]
        else:
            vals = [g(s[n], *path) for n in ("subset_train_pool", "subset_test")]
        L.append(f"| {label} | {vals[0]} | {vals[1]} |")
    if "subset_images" in s:
        im = s["subset_images"]
        L += ["", "## Table 4 - Images (subset)", "", "| Check | Value |", "|---|---|"]
        for k in ("images_expected", "images_missing", "images_corrupt",
                  "questions_whose_image_is_missing_or_corrupt", "missing_modality_rate_pct",
                  "within_train_pool_near_duplicate_pairs", "within_test_near_duplicate_pairs",
                  "cross_split_near_duplicate_pairs", "phash_max_hamming"):
            L.append(f"| {k} | {im.get(k)} |")
        L.append(f"| width px (median / min / max) | {g(im, 'width', 'median')} / {g(im, 'width', 'min')} / "
                 f"{g(im, 'width', 'max')} |")
        L.append(f"| height px (median / min / max) | {g(im, 'height', 'median')} / {g(im, 'height', 'min')} / "
                 f"{g(im, 'height', 'max')} |")
    L += ["", "## Table 5 - Answer vocabulary options (built on train pool only)", "",
          "| Option | Vocab size K | Train-pool coverage % | Test coverage % | Test oracle VQA acc % |",
          "|---|---|---|---|---|"]
    for k, v in s["answer_vocab_from_train_pool"].items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {v['vocab_size']:,} | {v['train_pool_coverage_pct']} | {v['test_coverage_pct']} | "
                 f"{v['test_oracle_vqa_accuracy']} |")
    lp = s["language_prior"]
    L += ["", "## Table 6 - Language prior (no image, no model)", "",
          "| Rule | Test VQA accuracy % |", "|---|---|",
          f"| Always answer '{lp['global_mode_answer']}' | {lp['acc_always_global_mode']} |",
          f"| Most common answer per question type | {lp['acc_mode_per_question_type']} |"]
    for at, v in lp["acc_mode_per_question_type_by_answer_type"].items():
        L.append(f"| ... on {at} questions | {v} |")
    r1, r2 = s["subset_representativeness"]["train_pool_vs_full_train"], \
        s["subset_representativeness"]["test_vs_full_val"]
    L += ["", "## Table 7 - Subset representativeness (total-variation distance, 0 = identical)", "",
          "| Distribution | Train pool vs full train2014 | Test vs full val2014 |", "|---|---|---|"]
    for k in ("tvd_answer_type", "tvd_question_type", "tvd_top100_answers"):
        L.append(f"| {k.replace('tvd_', '')} | {r1[k]} | {r2[k]} |")
    if "token_lengths" in s:
        t = s["token_lengths"]
        L += ["", f"## Table 8 - Question length in tokens ({t['tokenizer']}, incl. special tokens)", "",
              "| mean | p95 | max | >16 % | >24 % | >32 % |", "|---|---|---|---|---|---|",
              f"| {t['mean']} | {t['p95']} | {t['max']} | {t['pct_over_16']} | {t['pct_over_24']} | {t['pct_over_32']} |"]
    L += ["", "Input file checksums (sha256) are in summary.json -> meta.input_files."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("eda_outputs"))
    ap.add_argument("--n-train-images", type=int, default=10000)
    ap.add_argument("--n-test-images", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=261)
    ap.add_argument("--images", action="store_true", help="download/check the subset images")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--phash-max-dist", type=int, default=4)
    ap.add_argument("--audit-n", type=int, default=100)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name/path for token lengths")
    ap.add_argument("--clip", default=None, help="HF CLIP model name/path for pair similarity (needs --images)")
    ap.add_argument("--clip-batch", type=int, default=64)
    args = ap.parse_args()

    out = args.out_dir
    (out / "subset").mkdir(parents=True, exist_ok=True)

    meta = {"run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "args": {k: str(v) for k, v in vars(args).items()},
            "versions": package_versions(), "input_files": {}}
    for split in SPLITS:
        for f in input_files(args.data_dir, split):
            if f.exists():
                meta["input_files"][f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}

    print("Loading annotations ...", flush=True)
    data, headers = {}, {}
    for split in SPLITS:
        data[split], headers[split] = load_split(args.data_dir, split)
    meta["json_headers"] = headers
    tr, va = data["train2014"], data["val2014"]

    s = {"meta": meta}
    print("Full-split statistics ...", flush=True)
    s["full_train2014"] = split_stats(tr)
    s["full_val2014"] = split_stats(va)
    s["full_image_overlap_train_val"] = len(set(tr["image_id"].dropna()) & set(va["image_id"].dropna()))

    # image-level subset: every question of a sampled image is kept with it
    tr_ids = sample_images(tr, args.n_train_images, args.seed)
    te_ids = sample_images(va, args.n_test_images, args.seed + 1)
    sub_tr = tr[(tr["_merge"] == "both") & tr["image_id"].isin(tr_ids)].copy()
    sub_te = va[(va["_merge"] == "both") & va["image_id"].isin(te_ids)].copy()
    (out / "subset" / "train_pool_image_ids.txt").write_text("\n".join(map(str, tr_ids)))
    (out / "subset" / "test_image_ids.txt").write_text("\n".join(map(str, te_ids)))
    sub_tr[["question_id", "image_id"]].to_csv(out / "subset" / "train_pool_question_ids.csv", index=False)
    sub_te[["question_id", "image_id"]].to_csv(out / "subset" / "test_question_ids.csv", index=False)

    print("Subset statistics ...", flush=True)
    s["subset_train_pool"] = split_stats(sub_tr)
    s["subset_test"] = split_stats(sub_te)
    s["subset_representativeness"] = {"train_pool_vs_full_train": representativeness(tr, sub_tr),
                                      "test_vs_full_val": representativeness(va, sub_te)}
    s["answer_vocab_from_train_pool"] = vocab_options(sub_tr, sub_te)
    s["question_text_overlap"] = question_overlap(sub_tr, sub_te)
    s["language_prior"] = language_prior(sub_tr, sub_te)

    if args.tokenizer:
        print(f"Token lengths with {args.tokenizer} ...", flush=True)
        s["token_lengths"] = token_lengths(pd.concat([sub_tr["question"], sub_te["question"]]), args.tokenizer)

    img_df, clip_df = None, None
    if args.images:
        d_tr = ensure_images(args.data_dir, "train2014", tr_ids, not args.no_download, args.workers)
        d_te = ensure_images(args.data_dir, "val2014", te_ids, not args.no_download, args.workers)
        print("Checking images ...", flush=True)
        a = image_checks(d_tr, "train2014", tr_ids)
        b = image_checks(d_te, "val2014", te_ids)
        img_df = pd.concat([a, b], ignore_index=True)
        img_df.to_csv(out / "subset" / "image_checks.csv", index=False)
        print("Near-duplicate search ...", flush=True)
        cross = near_dups(a, b, args.phash_max_dist, same_set=False)
        within_tr = near_dups(a, a, args.phash_max_dist, same_set=True)
        within_te = near_dups(b, b, args.phash_max_dist, same_set=True)
        cols = ["image_id_a", "image_id_b", "hamming"]
        pd.DataFrame(cross, columns=cols).to_csv(out / "subset" / "near_dups_cross_trainpool_test.csv", index=False)
        pd.DataFrame(within_tr, columns=cols).to_csv(out / "subset" / "near_dups_within_trainpool.csv", index=False)
        pd.DataFrame(within_te, columns=cols).to_csv(out / "subset" / "near_dups_within_test.csv", index=False)
        bad_tr = set(a.loc[~a["ok"], "image_id"])
        bad_te = set(b.loc[~b["ok"], "image_id"])
        n_bad_q = int(sub_tr["image_id"].isin(bad_tr).sum() + sub_te["image_id"].isin(bad_te).sum())
        okimg = img_df[img_df["ok"]]
        s["subset_images"] = {
            "images_expected": int(len(img_df)),
            "images_missing": int((~img_df["exists"]).sum()),
            "images_corrupt": int((img_df["exists"] & ~img_df["ok"]).sum()),
            "questions_whose_image_is_missing_or_corrupt": n_bad_q,
            "missing_modality_rate_pct": round(100 * n_bad_q / max(1, len(sub_tr) + len(sub_te)), 3),
            "width": describe(okimg["width"]), "height": describe(okimg["height"]),
            "aspect_ratio": describe(okimg["width"] / okimg["height"]),
            "file_kb": describe(okimg["file_kb"]),
            "total_mb": round(float(okimg["file_kb"].sum()) / 1024, 1),
            "formats": okimg["format"].value_counts().to_dict(),
            "modes": okimg["mode"].value_counts().to_dict(),
            "images_smaller_than_224px_side": int(((okimg["width"] < 224) | (okimg["height"] < 224)).sum()),
            "within_train_pool_near_duplicate_pairs": len(within_tr),
            "within_test_near_duplicate_pairs": len(within_te),
            "cross_split_near_duplicate_pairs": len(cross),
            "phash_max_hamming": args.phash_max_dist,
        }
        if args.clip:
            print(f"CLIP similarity with {args.clip} ...", flush=True)
            good = pd.concat([sub_tr[~sub_tr["image_id"].isin(bad_tr)],
                              sub_te[~sub_te["image_id"].isin(bad_te)]])
            clip_df = clip_similarity(good, args.data_dir, args.clip, args.clip_batch)
            clip_df.to_csv(out / "subset" / "clip_pair_similarity.csv", index=False)
            s["clip_pair_similarity"] = {
                "model": args.clip, **describe(clip_df["clip_sim"]),
                "_note": "text = question + majority answer; low values flag pairs to inspect, "
                         "not proof of mismatch (abstract questions also score low).",
            }

    # audit sheet for the manual pair-quality check (train pool, seeded)
    audit = sub_tr.sample(min(args.audit_n, len(sub_tr)), random_state=args.seed)
    audit = audit.assign(
        image_file=[f"images/train2014/{coco_name('train2014', i)}" for i in audit["image_id"]],
        answers_top3=[", ".join(f"{a}({c})" for a, c in Counter(x).most_common(3)) for x in audit["answers"]],
        verdict="", note="")
    audit[["question_id", "image_id", "image_file", "question", "answer_type", "mc_answer",
           "answers_top3", "verdict", "note"]].to_csv(out / "audit_sample.csv", index=False)

    print("Figures ...", flush=True)
    save_figures(sub_tr, sub_te, img_df, clip_df, out, args.data_dir, args.seed)

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2, ensure_ascii=False, default=str)
    (out / "proposal_tables.md").write_text(proposal_tables(s), encoding="utf-8")
    print(f"Done. Outputs in {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
