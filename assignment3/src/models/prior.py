"""Sanity floor: most common train answer for each official question_type (no image, no training).

Usage: python src/models/prior.py    (prints val VQA accuracy, computed question by question)
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.vqa import load_rows, vqa_accuracy  # noqa: E402


def most_common(answers):
    """Most frequent answer; ties broken by answer string, ascending."""
    return min(Counter(answers).items(), key=lambda kv: (-kv[1], kv[0]))[0]


def fit(train):
    table = {qt: most_common(g.mc_answer) for qt, g in train.groupby("question_type")}
    return table, most_common(train.mc_answer)


def predict(model, rows):
    table, fallback = model
    return [table.get(qt, fallback) for qt in rows.question_type]


if __name__ == "__main__":
    data = load_rows(["train", "val"])
    preds = predict(fit(data["train"]), data["val"])
    acc = sum(vqa_accuracy(p, g) for p, g in zip(preds, data["val"].answers)) / len(preds)
    print(f"prior val VQA accuracy: {100 * acc:.4f}%")
