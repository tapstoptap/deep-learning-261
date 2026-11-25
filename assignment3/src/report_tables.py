"""Collect eval_val.json, eval_test.json and the training logs into m2_results.{json,md}.

Usage: python src/report_tables.py
"""
import json

import numpy as np

from datasets.vqa import A3, load_rows

R = A3 / "results"
NAMES = [("prior", "a: prior (no training)"), ("unimodal_a", "c: image only (unimodal A)"),
         ("unimodal_b", "b: text only (unimodal B)"), ("fusion_simple", "d: concat fusion")]
COLS = [("vqa_acc", "VQA acc"), ("exact_match", "exact match"), ("vqa_acc_yes/no", "yes/no"),
        ("vqa_acc_number", "number"), ("vqa_acc_other", "other"), ("vqa_acc_what_color", "what color")]
TYPES = ["vqa_acc", "vqa_acc_yes/no", "vqa_acc_number", "vqa_acc_other", "vqa_acc_what_color"]


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def main():
    ev = {s: json.loads((R / f"eval_{s}.json").read_text()) for s in ("val", "test")}
    data = load_rows(["val", "test"])
    out, md = {"splits": ev, "training": {}, "gaps": {}}, ["# M2 results\n"]
    md.append("All accuracies in %. Trained models: mean ± sample std over seeds 261, 262, 263.\n")
    for s, e in ev.items():
        counts = data[s].answer_type.value_counts()
        md.append(f"## {s} (n = {e['n_questions']:,}; yes/no {counts['yes/no']:,}, number {counts['number']:,}, "
                  f"other {counts['other']:,}; oracle ceiling {e['oracle_vqa_acc']:.2f})\n")
        rows = []
        for key, label in NAMES:
            m = e["models"][key]
            rows.append([label] + [f"{m['mean'][c]:.2f}" if key == "prior" else
                                   f"{m['mean'][c]:.2f} ± {m['std'][c]:.2f}" for c, _ in COLS])
        md.append(table(["model"] + [c for _, c in COLS], rows))
    md.append("## Per-seed gaps (percentage points)\n")
    for label, hi, lo in (("b − c (text only minus image only)", "unimodal_b", "unimodal_a"),
                          ("d − b (fusion minus text only)", "fusion_simple", "unimodal_b")):
        rows = []
        for s, e in ev.items():
            for seed in ("261", "262", "263"):
                a, b = e["models"][hi]["seeds"][seed], e["models"][lo]["seeds"][seed]
                gap = {c: a[c] - b[c] for c in TYPES}
                out["gaps"].setdefault(f"{hi}-{lo}", {}).setdefault(s, {})[seed] = gap
                rows.append([s, seed] + [f"{gap[c]:+.2f}" for c in TYPES])
        md.append(f"### {label}\n")
        md.append(table(["split", "seed", "overall", "yes/no", "number", "other", "what color"], rows))
    md.append("## Training cost (heads only, cached features)\n")
    rows = []
    for key, label in NAMES[1:]:
        logs = [json.loads(p.read_text()) for p in sorted((R / "logs").glob(f"{key}_s*.json"))]
        t = {"params": logs[0]["params"],
             "mean_epoch_seconds": float(np.mean([x["mean_epoch_seconds"] for x in logs])),
             "train_seconds_all_seeds": float(sum(x["train_seconds_total"] for x in logs)),
             "best_epochs": [x["best_epoch"] for x in logs]}
        out["training"][key] = t
        rows.append([label, f"{t['params']:,}", f"{t['mean_epoch_seconds']:.2f}",
                     f"{t['train_seconds_all_seeds']:.1f}", ", ".join(map(str, t["best_epochs"]))])
    md.append(table(["model", "parameters", "s/epoch", "train s, 3 seeds", "best epoch (261, 262, 263)"], rows))
    (R / "m2_results.json").write_text(json.dumps(out, indent=1))
    (R / "m2_results.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
