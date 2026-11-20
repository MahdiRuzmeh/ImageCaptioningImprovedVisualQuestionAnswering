"""Figures for VQA_ROUND2_REPORT.md: caption dropout (round 1b), better fold captioners (round 2), ensemble.

Finglish: hame az file haye local (results.json, test_predictions.json, vqa_ensemble*.json, fold_eval.json,
train.log) → report_figs_round2/*.png. Rang = slot haye categorical be tartib-e sabet; har bar adad dare.

    python scripts/report_round2_figures.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QDIC = ROOT.parent / "QuestionDependentImageCaptioner"
sys.path.insert(0, str(QDIC))
from metrics import answer_in_caption, caption_negated  # noqa: E402
from train import tok  # noqa: E402

OUT = ROOT / "report_figs_round2"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})

VQA = {  # Finglish: esm → folder-e run ha (seed 0, 1)
    "Baseline (MCAN)": ROOT / "vqa_strong_baseline/runs/nocap",
    "+ captions (Stage A+B)": ROOT / "vqa_strong_crossfit/runs/blip",
    "+ captions, b64 + 840B (1b-C)": ROOT / "vqa_b64_crossfit/runs/blip",
    "+ captions, b64 + 840B + dropout 0.3 (1b-B)": ROOT / "vqa_b64_crossfit_drop/runs/blip",
}
CAPTIONERS = {
    "current (frozen ViT, 4 ep)": QDIC / "outputs/fold_eval/fold0_current.json",
    "A: top-4 ViT + 6 ep": QDIC / "outputs/round2/model_b/b_blip_fold0_v2A/fold_eval.json",
    "B: A + answer words x3": QDIC / "outputs/round2/model_b/b_blip_fold0_v2B/fold_eval.json",
}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def style(ax, ylabel, ymin, ymax, grid_axis="y"):
    ax.set_ylim(ymin, ymax) if grid_axis == "y" else ax.set_xlim(ymin, ymax)
    ax.set_ylabel(ylabel) if grid_axis == "y" else ax.set_xlabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("->", OUT / name)


def bars(ax, groups, series, values, w=None, fmt="{:.1f}", fs=7.2):
    """Finglish: grouped bars, rang = slot-e sabet-e har series, adad bala-ye har bar."""
    w = w or 0.8 / len(series)
    for j, s in enumerate(series):
        for k, g in enumerate(groups):
            v = values[s][g]
            x = k + (j - (len(series) - 1) / 2) * w
            ax.bar(x, v, w * 0.9, color=SLOT[j], label=s if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.6, fmt.format(v), ha="center", va="bottom", fontsize=fs, color=INK)
    ax.set_xticks(range(len(groups)), groups)


def vqa_mean(path, key):
    R = [load(f"{path}_s{s}/results.json") for s in (0, 1)]
    if key == "overall":
        return statistics.mean(r["test"]["acc"] for r in R)
    if key == "shuffled":
        return statistics.mean(r["test_shuffled_captions"]["acc"] for r in R)
    return statistics.mean(r["test"]["answer_type"][key]["acc"] for r in R)


def fig_answer_types():
    ens = {c["name"]: c for c in load(ROOT / "vqa_ensemble_1b.json")["combos"]}
    best = next(v for k, v in ens.items() if k.startswith("weighted"))
    groups = ["overall", "yes/no", "number", "other"]
    vals = {s: {g: vqa_mean(p, g) for g in groups} for s, p in VQA.items()}
    vals["Ensemble (best, w on dev)"] = {g: best["test"][g] for g in groups}
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    style(ax, "VQA accuracy (%)", 40, 90)
    bars(ax, ["Overall", "Yes/no", "Number", "Other"], list(vals),
         {s: dict(zip(["Overall", "Yes/no", "Number", "Other"], v.values())) for s, v in vals.items()}, fs=6.0)
    ax.set_title("Test accuracy by answer type (10,001 questions; single models = mean of 2 seeds)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.2)
    save(fig, "fig1_answer_types.png")


def fig_right_wrong():
    keys = {"Baseline (MCAN)": VQA["Baseline (MCAN)"],
            "+ captions, no dropout (1b-C)": VQA["+ captions, b64 + 840B (1b-C)"],
            "+ captions, dropout 0.3 (1b-B)": VQA["+ captions, b64 + 840B + dropout 0.3 (1b-B)"]}
    P = {m: [{p["question_id"]: p for p in load(f"{d}_s{s}/test_predictions.json")} for s in (0, 1)]
         for m, d in keys.items()}
    names = {"lex_ok": "open question,\ncaption has the answer", "lex_bad": "open question,\ncaption doesn't",
             "yn_ok": "yes/no, caption\npolarity right", "yn_bad": "yes/no, caption\npolarity wrong"}
    ids = {k: [] for k in names}
    for q, x in P["+ captions, no dropout (1b-C)"][0].items():
        c, g = tok(x["caption"]), x["gt"]
        if g in ("yes", "no"):
            ids["yn_ok" if caption_negated(c) == (g == "no") else "yn_bad"].append(q)
        elif tok(g):
            ids["lex_ok" if answer_in_caption(tok(g), c) else "lex_bad"].append(q)
    labels = [f"{names[k]}\n(n={len(v):,})" for k, v in ids.items()]
    vals = {m: {lab: 100 * statistics.mean(statistics.mean(P[m][s][q]["score"] for q in v) for s in (0, 1))
                for lab, v in zip(labels, ids.values())} for m in keys}
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    style(ax, "VQA accuracy (%)", 0, 112)
    ax.set_yticks(range(0, 101, 20))
    bars(ax, labels, list(keys), vals)
    ax.tick_params(axis="x", labelsize=7.4)
    ax.set_title("Accuracy when the caption is right vs wrong (test, mean of 2 seeds)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7.4)
    save(fig, "fig2_caption_right_wrong.png")
    return {m: dict(zip(names, vals[m].values())) for m in keys}, {k: len(v) for k, v in ids.items()}


def captioner_stats():
    out = {}
    for name, p in CAPTIONERS.items():
        preds = load(p)["predictions"]
        hit = {"number": [], "other": []}
        pol = {"yes": [], "no": []}
        neg = []
        for x in preds:
            a, c = str(x["answer"]).strip().lower(), tok(x["pred"])
            if a in ("yes", "no"):
                pol[a].append(caption_negated(c) == (a == "no"))
                neg.append(caption_negated(c))
            else:
                hit["number" if a.isdigit() else "other"].append(answer_in_caption(tok(a), c))
        allhit = hit["number"] + hit["other"]
        allpol = pol["yes"] + pol["no"]
        pct = lambda v: 100 * sum(v) / len(v)
        out[name] = {"AnswerHit": pct(allhit), "number": pct(hit["number"]), "other": pct(hit["other"]),
                     "Polarity": pct(allpol), "answer yes": pct(pol["yes"]), "answer no": pct(pol["no"]),
                     "negated": pct(neg)}
    return out


def fig_captioners(S):
    groups = ["AnswerHit", "number", "other", "Polarity", "answer yes", "answer no"]
    labels = ["AnswerHit\n(all open)", "AnswerHit\nnumber", "AnswerHit\nother", "Polarity\n(all yes/no)",
              "Polarity\nanswer = yes", "Polarity\nanswer = no"]
    vals = {s: dict(zip(labels, (S[s][g] for g in groups))) for s in S}
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    style(ax, "% of the 10,000 fold-1 rows", 30, 100)
    bars(ax, labels, list(S), vals, fs=6.6)
    ax.tick_params(axis="x", labelsize=7.4)
    ax.set_title("Fold-0 captioners on images they never saw (10,000 QD rows, beam 3)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7.4)
    save(fig, "fig3_captioners.png")


def fig_captioner_curves():
    rx = re.compile(r"epoch (\d+)/\d+ train_loss=([\d.]+) val_CIDEr=([\d.]+).*val_AnswerHit=([\d.]+)")
    runs = {"A: top-4 ViT + 6 ep": "b_blip_fold0_v2A", "B: A + answer words x3": "b_blip_fold0_v2B"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for j, (name, d) in enumerate(runs.items()):
        rows = [rx.search(l) for l in (QDIC / "outputs/round2/model_b" / d / "train.log").read_text().splitlines()]
        rows = [tuple(float(x) for x in m.groups()) for m in rows if m]
        for ax, col, lab in ((axes[0], 3, "val AnswerHit (%)"), (axes[1], 2, "val CIDEr")):
            ys = [r[col] * (100 if col == 3 else 1) for r in rows]
            ax.plot([r[0] for r in rows], ys, color=SLOT[j + 1], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=name)
            ax.text(rows[-1][0] + 0.15, ys[-1], f"{ys[-1]:.{1 if col == 3 else 2}f}", va="center", fontsize=7.2,
                    color=INK)
            ax.set_ylabel(lab)
    for ax, (lo, hi), t in ((axes[0], (51, 58), "Val AnswerHit per epoch"), (axes[1], (5.8, 6.35), "Val CIDEr per epoch")):
        style(ax, ax.get_ylabel(), lo, hi)
        ax.set_xlabel("epoch")
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.7, 6.9)
        ax.set_title(t, loc="left")
    axes[0].legend(frameon=False, fontsize=7.2, loc="lower right")
    save(fig, "fig4_captioner_curves.png")


def fig_ensemble_weight():
    old = load(ROOT / "vqa_ensemble.json")["dev_by_caption_weight"]
    new = load(ROOT / "vqa_ensemble_1b.json")["dev_by_caption_weight"]
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    for j, (name, d) in enumerate((("4 Stage A+B models (2 caption + 2 baseline)", old),
                                   ("4 round-1b caption models + 2 baselines", new))):
        w = sorted(d, key=float)
        ys = [d[k] for k in w]
        ax.plot([float(k) for k in w], ys, color=SLOT[j + 1], linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        b = max(range(len(w)), key=lambda i: ys[i])
        ax.annotate(f"best w = {float(w[b]):.1f}: {ys[b]:.2f}", (float(w[b]), ys[b]), textcoords="offset points",
                    xytext=(-60, 8), fontsize=7.4, color=INK)
    style(ax, "dev accuracy (%)", 66, 71.5)
    ax.set_xlabel("weight w of the caption models (1 − w for the baselines)")
    ax.set_title("Choosing the ensemble weight on dev (test never used)", loc="left")
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    save(fig, "fig5_ensemble_weight.png")


def fig_progress():
    rows = [("UpDn, 31k questions", 46.56, 59.79), ("UpDn, all train2014", 60.03, 66.08),
            ("MCAN + GloVe 6B, train + val", 65.74, 67.80), ("MCAN, batch 64 + GloVe 840B", 66.39, 68.08),
            ("+ caption dropout 0.3", None, 68.31), ("Ensemble, Stage A+B models", None, 69.11),
            ("Ensemble, round-1b models", None, 69.56)]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, b, c) in zip(y, rows):
        if b is not None:
            ax.barh(yi + 0.19, b, 0.36, color=SLOT[0], edgecolor=SURFACE, linewidth=1.2,
                    label="no captions" if yi == y[0] else None)
            ax.text(b + 0.3, yi + 0.19, f"{b:.2f}", va="center", fontsize=7.2, color=INK)
        off = -0.19 if b is not None else 0
        ax.barh(yi + off, c, 0.36, color=SLOT[1], edgecolor=SURFACE, linewidth=1.2,
                label="with captions" if yi == y[0] else None)
        ax.text(c + 0.3, yi + off, f"{c:.2f}", va="center", fontsize=7.2, color=INK, fontweight="bold")
    ax.axvline(70.67, color=INK2, linewidth=1, linestyle="--")
    ax.text(70.5, y[0] + 0.55, "paper 70.67\n(test-std)", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.8)
    style(ax, "VQA accuracy on our 10,001 test questions (%)", 40, 74, grid_axis="x")
    ax.set_title("Progress across all experiments", loc="left")
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig6_progress.png")


def main():
    fig_answer_types()
    rw, n = fig_right_wrong()
    S = captioner_stats()
    fig_captioners(S)
    fig_captioner_curves()
    fig_ensemble_weight()
    fig_progress()
    print(json.dumps({"right_wrong": rw, "n": n, "captioners": S}, indent=1))


if __name__ == "__main__":
    main()
