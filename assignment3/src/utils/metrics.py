"""VQA metrics from predicted vocabulary indices and the soft-score matrix."""
import numpy as np

ANSWER_TYPES = ["yes/no", "number", "other"]


def per_question(pred_idx, rows, vocab, targets):
    answers = np.array(vocab, dtype=object)[pred_idx]
    vqa = targets[np.arange(len(rows)), pred_idx]
    exact = (answers == rows.mc_answer.to_numpy()).astype(np.float32)
    return answers, vqa, exact


def summarise(vqa, exact, rows):
    """Percentages: overall and per answer type, plus the 'what color' question types."""
    out = {"vqa_acc": 100 * float(vqa.mean()), "exact_match": 100 * float(exact.mean())}
    for t in ANSWER_TYPES:
        m = (rows.answer_type == t).to_numpy()
        out[f"vqa_acc_{t}"] = 100 * float(vqa[m].mean())
    m = rows.question_type.str.startswith("what color").to_numpy()
    out["vqa_acc_what_color"] = 100 * float(vqa[m].mean())
    return out
