"""Figures for VQA_FINAL_REPORT.md: official VQA v2 test-standard results (EvalAI) vs the base paper.

Finglish: adad ha az file haye natije-ye EvalAI (vqa_final_t3 / vqa_final_t4) + jadval-e 6-e maghale
(Sharma & Jalal 2021). Khorooji: report_figs_official/*.png

    python scripts/official_results_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_figs_official"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def evalai(p):
    d = {}
    for x in json.loads(Path(p).read_text(encoding="utf-8")):
        d.update(x)
    return d


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("->", OUT / name)


def main():
    cap = evalai(ROOT / "vqa_final_t3/vqa_final/Ensemble-VQA-With-Large-Blip-Test-Standard-Result-EvalAI.json")
    base = evalai(ROOT / "vqa_final_t4/vqa_final_baseline/Ensemble-VQA-BaseLine-Test-Standard-Result-EvalAI.json")
    paper = {"overall": 70.67, "yes/no": 86.63, "number": 48.23, "other": 59.98}  # Table 6, Baseline + Att + Cap
    keys, labs = ["overall", "yes/no", "number", "other"], ["Overall", "Yes/no", "Number", "Other"]
    series = [("Base paper: Baseline + Att + Cap", paper), ("Ours: MCAN, no captions", base["test-standard"]),
              ("Ours: MCAN + BLIP-large question-dependent captions", cap["test-standard"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    w = 0.26
    for j, (nm, d) in enumerate(series):
        for k, key in enumerate(keys):
            x = k + (j - 1) * w
            ax.bar(x, d[key], w * 0.9, color=SLOT[j], label=nm if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, d[key] + 0.6, f"{d[key]:.2f}", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_ylim(40, 95)
    ax.set_ylabel("VQA accuracy (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xticks(range(4), labs)
    ax.set_title("Official VQA v2 test-standard (EvalAI)", loc="left")
    ax.legend(frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.4)
    save(fig, "fig1_test_standard.png")

    # Table 6 of the paper + ours, overall
    rows = [("Goyal et al. (MCB)", 62.27, 0), ("Anderson et al. (BUTD)", 65.32, 0), ("Teney et al.", 65.67, 0),
            ("Nguyen & Okatani", 67.00, 0), ("Zhang et al.", 67.34, 0), ("Paper: Baseline", 61.21, 1),
            ("Paper: Baseline + Att", 63.32, 1), ("Paper: Baseline + Cap", 69.91, 1),
            ("Paper: Baseline + Att + Cap", 70.67, 1), ("Ours: MCAN, no captions", base["test-standard"]["overall"], 2),
            ("Ours: MCAN + BLIP-large captions", cap["test-standard"]["overall"], 3)]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    cols = ["#b9b7af", SLOT[0], SLOT[1], SLOT[2]]
    for i, (nm, v, g) in enumerate(rows):
        ax.barh(i, v, 0.62, color=cols[g], edgecolor=SURFACE, linewidth=1.2)
        ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=7.2, color=INK, fontweight="bold" if g >= 2 else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.6)
    ax.set_xlim(55, 76)
    ax.set_xlabel("overall accuracy on VQA v2 test-standard (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[0], label="other methods (paper's Table 6)"),
                       Patch(color=cols[1], label="base paper"), Patch(color=cols[2], label="ours, no captions"),
                       Patch(color=cols[3], label="ours, + captions")],
              frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Test-standard: the base paper's Table 6 and our models", loc="left")
    save(fig, "fig2_table6.png")


if __name__ == "__main__":
    main()
