"""Training curves (mean and min-max range over seeds) from results/logs.

Usage: python src/plot_curves.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from datasets.vqa import A3  # noqa: E402

SERIES = [("unimodal_a", "c: image only", "#257de0"), ("unimodal_b", "b: text only", "#f47230"),
          ("fusion_simple", "d: concat fusion", "#00b894")]
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
PANELS = [("train_loss", "Training loss (log scale)"), ("val_vqa_acc", "Validation VQA accuracy (%)")]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, title) in zip(axes, PANELS):
        for name, label, colour in SERIES:
            runs = [json.loads((A3 / "results/logs" / f"{name}_s{s}.json").read_text())["history"]
                    for s in (261, 262, 263)]
            x = [e["epoch"] for e in runs[0]]
            y = np.array([[e[key] for e in r] for r in runs])
            ax.fill_between(x, y.min(0), y.max(0), color=colour, alpha=0.2, linewidth=0)
            ax.plot(x, y.mean(0), color=colour, linewidth=2, label=label)
            # direct label with the final mean, so identity does not rest on colour alone
            ax.annotate(f"{label[0]}  {y.mean(0)[-1]:.2f}", (x[-1], y.mean(0)[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlabel("epoch", color=MUTED)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(1, 23)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_yscale("log")
    axes[0].set_yticks([2, 3, 5, 10, 15])
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = A3 / "results" / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "training_curves.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
