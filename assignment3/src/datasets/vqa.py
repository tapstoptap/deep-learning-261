"""VQA v2 subset: split rows, answer vocabulary and soft-score targets.

Loading, answer normalisation and the VQA metric come from eda/vqa_eda.py so that
training, evaluation and the M1 statistics share one definition.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(A3 / "eda"))
from vqa_eda import coco_name, load_split, vqa_accuracy  # noqa: E402

DATA = A3 / "eda/data/vqa_v2"
SPLITS = A3 / "splits"
FEATURES = A3 / "features"
SOURCE = {"train": "train2014", "val": "train2014", "test": "val2014"}


def split_ids(split):
    return pd.read_csv(SPLITS / f"{split}_question_ids.csv")


def load_vocab():
    d = json.loads((SPLITS / "answer_vocab.json").read_text())
    return [d[str(i)] for i in range(len(d))]


def load_rows(splits):
    """{split: DataFrame} in the order of the committed question-id lists."""
    cache, out = {}, {}
    for s in splits:
        src = SOURCE[s]
        if src not in cache:
            df, _ = load_split(DATA, src)
            cache[src] = df.set_index("question_id")
        ids = split_ids(s)
        rows = cache[src].loc[ids.question_id].reset_index()
        assert (rows.image_id.to_numpy() == ids.image_id.to_numpy()).all(), f"image id mismatch in {s}"
        out[s] = rows
    return out


def soft_targets(rows, vocab):
    """N x K matrix: official VQA accuracy of predicting vocabulary answer k on question n."""
    index = {a: i for i, a in enumerate(vocab)}
    t = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for n, gts in enumerate(rows.answers):
        for a in set(gts):
            if a in index:
                t[n, index[a]] = vqa_accuracy(a, gts)
    return t


def load_features(rows):
    """(image N x 2048, question N x 768) float32 arrays aligned to `rows` by id."""
    img = np.load(FEATURES / "image.npy")
    txt = np.load(FEATURES / "question.npy")
    ii = pd.read_csv(FEATURES / "image_index.csv")
    qi = pd.read_csv(FEATURES / "question_index.csv")
    ipos = pd.Series(np.arange(len(ii)), index=ii.image_id)
    qpos = pd.Series(np.arange(len(qi)), index=qi.question_id)
    return (img[ipos.loc[rows.image_id].to_numpy()].astype(np.float32),
            txt[qpos.loc[rows.question_id].to_numpy()].astype(np.float32))
