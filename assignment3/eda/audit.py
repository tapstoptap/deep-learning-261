"""
Manual pair-quality audit for the A3 Dataset Proposal.

Step 1 - build a viewing page next to the data (images are linked, not copied):
    python audit.py html --audit eda_outputs/audit_sample.csv --data-dir data/vqa_v2 \
        --out eda_outputs/audit_sheet.html

Step 2 - open audit_sheet.html in a browser, look at each image + question + answers and
fill the `verdict` column of audit_sample.csv (Excel / LibreOffice / a text editor) with one of:
    matched     the question is clearly about this image and the majority answer is right
    weak        the question relates to the image but is ambiguous / subjective, or the
                majority answer is debatable
    mismatched  the question does not fit the image, or the majority answer is wrong
Use the `note` column for the reason (e.g. "blurry", "subjective", "answer wrong: 3 not 2").

Step 3 - summarise (counts, %, Wilson 95% confidence intervals):
    python audit.py summary --audit eda_outputs/audit_sample.csv --out eda_outputs/audit_summary.json
"""

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

VERDICTS = ("matched", "weak", "mismatched")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1))


def make_html(audit_csv: Path, data_dir: Path, out: Path):
    df = pd.read_csv(audit_csv, dtype=str).fillna("")
    out.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, r in enumerate(df.itertuples(), 1):
        img = Path(data_dir) / r.image_file
        rel = os.path.relpath(img.resolve(), out.parent.resolve()).replace(os.sep, "/")
        exists = img.exists()
        img_tag = (f'<img src="{html.escape(rel)}" loading="lazy">' if exists
                   else '<div class="missing">image not found - run vqa_eda.py with --images</div>')
        cards.append(f"""
<div class="card">
  <div class="num">#{i} &middot; qid {html.escape(r.question_id)} &middot; img {html.escape(r.image_id)}</div>
  {img_tag}
  <div class="q">Q: {html.escape(r.question)}</div>
  <div class="a">Majority: <b>{html.escape(r.mc_answer)}</b> &middot; {html.escape(r.answer_type)}</div>
  <div class="a">Top answers: {html.escape(r.answers_top3)}</div>
  <div class="v">Current verdict: <b>{html.escape(r.verdict) or "-"}</b> {html.escape(r.note)}</div>
</div>""")
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>VQA v2 pair audit</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#fafafa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}}
.card img{{width:100%;height:240px;object-fit:contain;background:#eee}}
.num{{font-size:12px;color:#666;margin-bottom:6px}} .q{{font-weight:600;margin-top:8px}}
.a,.v{{font-size:14px;margin-top:4px}} .missing{{height:240px;display:flex;align-items:center;
justify-content:center;background:#fee;color:#900}}
</style></head><body>
<h1>VQA v2 pair-quality audit ({len(df)} pairs)</h1>
<p>Fill the <code>verdict</code> column in <code>{html.escape(audit_csv.name)}</code> with
<b>matched</b>, <b>weak</b> or <b>mismatched</b> (row order = card number). Reload this page
after re-running <code>audit.py html</code> to see your verdicts.</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({len(df)} cards)")


def summary(audit_csv: Path, out: Path):
    df = pd.read_csv(audit_csv, dtype=str).fillna("")
    df["verdict"] = df["verdict"].str.strip().str.lower()
    filled = df[df["verdict"] != ""]
    bad = sorted(set(filled["verdict"]) - set(VERDICTS))
    if bad:
        sys.exit(f"Unknown verdict values {bad}; use one of {VERDICTS}")
    n = len(filled)
    res = {"audited": n, "total_rows": len(df), "not_yet_filled": int(len(df) - n), "verdicts": {}}
    for v in VERDICTS:
        k = int((filled["verdict"] == v).sum())
        res["verdicts"][v] = {"count": k, "pct": round(100 * k / n, 1) if n else 0.0,
                              "wilson95_pct": wilson(k, n)}
    res["by_answer_type"] = {
        at: {v: int((g["verdict"] == v).sum()) for v in VERDICTS}
        for at, g in filled.groupby("answer_type")}
    res["notes_on_non_matched"] = filled.loc[filled["verdict"] != "matched",
                                             ["question_id", "question", "mc_answer", "verdict", "note"]
                                             ].to_dict(orient="records")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("audited", "not_yet_filled", "verdicts")}, indent=2))
    if res["not_yet_filled"]:
        print(f"WARNING: {res['not_yet_filled']} rows have no verdict yet.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("html")
    h.add_argument("--audit", type=Path, required=True)
    h.add_argument("--data-dir", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary")
    s.add_argument("--audit", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "html":
        make_html(a.audit, a.data_dir, a.out)
    else:
        summary(a.audit, a.out)


if __name__ == "__main__":
    main()
