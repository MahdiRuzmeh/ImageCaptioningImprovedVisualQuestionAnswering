"""Figures for VQA_REPORT.md: baseline vs + cross-fitted BLIP captions (all train2014, test = 10,001 val2014).

Finglish: az runs/*/results.json, history.json, test_predictions.json + captions_crossfit.json → report_figs_vqa/*.png.
Rang: slot 1 (blue) = baseline, slot 2 (orange) = + captions; label-e adadi roye har bar.

    python scripts/report_vqa_figures.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent / "QuestionDependentImageCaptioner"))
from metrics import answer_in_caption, caption_negated  # noqa: E402
from train import tok  # noqa: E402

OUT = ROOT / "report_figs_vqa"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"nocap": "#2a78d6", "blip": "#eb6834"}
LABEL = {"nocap": "Baseline (no caption)", "blip": "+ cross-fitted BLIP captions"}
DIR = {"nocap": ROOT / "vqa_full_baseline", "blip": ROOT / "vqa_full_crossfit"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def runs(m):
    return [load(DIR[m] / "runs" / f"{m}_s{k}" / "results.json") for k in range(3)]


def style(ax, ylabel, ymin=0, ymax=100, grid_axis="y"):
    ax.set_ylim(ymin, ymax) if grid_axis == "y" else ax.set_xlim(ymin, ymax)
    ax.set_ylabel(ylabel) if grid_axis == "y" else ax.set_xlabel(ylabel)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left" if grid_axis == "y" else "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_overall(R):
    cats = [("Overall", None), ("Yes/no", "yes/no"), ("Number", "number"), ("Other", "other")]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    style(ax, "VQA accuracy (%)")
    w = 0.36
    for j, m in enumerate(("nocap", "blip")):
        for k, (lab, t) in enumerate(cats):
            v = [r["test"]["acc"] if t is None else r["test"]["answer_type"][t]["acc"] for r in R[m]]
            mu, sd = statistics.mean(v), statistics.stdev(v)
            x = k + (j - 0.5) * w
            ax.bar(x, mu, w * 0.92, color=C[m], label=LABEL[m] if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.errorbar(x, mu, yerr=sd, color=INK2, capsize=2.5, linewidth=0.9)
            ax.text(x, mu + sd + 1.2, f"{mu:.1f}", ha="center", va="bottom", fontsize=7.6, color=INK)
    ax.set_xticks(range(len(cats)), [c[0] for c in cats])
    ax.set_title("Test accuracy, mean ± std over 3 seeds (10,001 val2014 questions)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.8)
    ax.set_ylim(0, 108)
    save(fig, "fig1_overall.png")


def fig_types(R):
    qt = {}
    for t, v in R["nocap"][0]["test"]["question_type"].items():
        qt[t] = {m: statistics.mean(r["test"]["question_type"][t]["acc"] for r in R[m]) for m in ("nocap", "blip")}
        qt[t]["n"] = v["n"]
    top = sorted(qt.items(), key=lambda kv: -kv[1]["n"])[:30][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    style(ax, "VQA accuracy (%)", 0, 104, grid_axis="x")
    for i, (t, v) in enumerate(top):
        ax.plot([v["nocap"], v["blip"]], [i, i], color=GRID, linewidth=2.2, zorder=1)
        for m in ("nocap", "blip"):
            ax.scatter(v[m], i, s=34, color=C[m], zorder=2, edgecolor=SURFACE, linewidth=1,
                       label=LABEL[m] if i == len(top) - 1 else None)
        d = v["blip"] - v["nocap"]
        ax.text(max(v["blip"], v["nocap"]) + 1.5, i, f"{d:+.1f}", va="center", fontsize=7.2,
                color=INK if d >= 0 else "#b3261e")
    ax.set_yticks(range(len(top)), [f"{t}  (n={v['n']})" for t, v in top], fontsize=7.6)
    ax.set_title("Accuracy per question type: 30 most frequent types (mean of 3 seeds)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.6, bbox_to_anchor=(0, -0.06))
    save(fig, "fig2_question_types.png")


def fig_curves():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    style(ax, "dev accuracy (%)", 40, 70)
    for m in ("nocap", "blip"):
        hs = [load(DIR[m] / "runs" / f"{m}_s{k}" / "history.json") for k in range(3)]
        n = min(len(h) for h in hs)
        mu = [statistics.mean(h[e]["dev"]["acc"] for h in hs) for e in range(n)]
        ax.plot(range(1, n + 1), mu, color=C[m], linewidth=2, marker="o", markersize=3.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL[m])
        ax.text(n + 0.25, mu[-1], f"{mu[-1]:.1f}", va="center", fontsize=7.6, color=INK)
    ax.set_xlim(0.5, n + 1.2)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("epoch")
    ax.set_title("Dev accuracy per epoch (mean of 3 seeds, 5,000 val2014 questions)", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=7.8)
    save(fig, "fig3_training_curves.png")


def fig_right_wrong():
    caps_rows = {}
    P = {m: [{p["question_id"]: p for p in load(DIR[m] / "runs" / f"{m}_s{k}" / "test_predictions.json")}
             for k in range(3)] for m in ("nocap", "blip")}
    names = {"lex_ok": "open question,\ncaption has the answer", "lex_bad": "open question,\ncaption doesn't",
             "yn_ok": "yes/no,\ncaption polarity right", "yn_bad": "yes/no,\ncaption polarity wrong"}
    keyed = {k: [] for k in names}
    for q, x in P["blip"][0].items():
        c, g = tok(x["caption"]), x["gt"]
        if g in ("yes", "no"):
            keyed["yn_ok" if caption_negated(c) == (g == "no") else "yn_bad"].append(q)
        elif tok(g):
            keyed["lex_ok" if answer_in_caption(tok(g), c) else "lex_bad"].append(q)
    groups = {names[k]: v for k, v in keyed.items()}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    style(ax, "VQA accuracy (%)", 0, 118)
    ax.set_yticks(range(0, 101, 20))
    w = 0.36
    for k, (lab, ids) in enumerate(groups.items()):
        for j, m in enumerate(("nocap", "blip")):
            v = 100 * statistics.mean(statistics.mean(P[m][s][q]["score"] for q in ids) for s in range(3))
            x = k + (j - 0.5) * w
            ax.bar(x, v, w * 0.92, color=C[m], label=LABEL[m] if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, color=INK)
        caps_rows[lab] = len(ids)
    ax.set_xticks(range(4), [f"{lab}\n(n={caps_rows[lab]:,})" for lab in groups], fontsize=7.6)
    ax.set_title("Accuracy when the caption is right vs wrong (test, mean of 3 seeds)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.8)
    save(fig, "fig4_caption_right_wrong.png")


def main():
    R = {m: runs(m) for m in ("nocap", "blip")}
    fig_overall(R)
    fig_types(R)
    fig_curves()
    fig_right_wrong()


if __name__ == "__main__":
    main()
