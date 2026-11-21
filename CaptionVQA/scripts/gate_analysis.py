"""Caption gate v1 analysis (round 3): G1 (gate + reliability loss) vs G2 (gate, VQA loss only) vs 1b-B (no gate).

Finglish: hame az file haye local (results/history/test_predictions/test_scores) → gate_analysis.json + report_figs_gate/.
  1. jadval-e model ha (2 seed), 4 halat-e caption (dorost/ghalat) ba miangin-e r
  2. r chi yad gerefte: AUC kolli, AUC dar har goroh-e soal, AUC-e "faghat no'-e soal" (2-fold), AUC ba label-e soft
  3. r be onvan-e switch (bedoon tuning): p = r·S_caption + (1 − r)·S_baseline, va hard switch (r < 0.5 → baseline)
  4. figure ha

    python scripts/gate_analysis.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from caption_error_analysis import analyse, group  # noqa: E402
from common import vqa_score  # noqa: E402
from train_vqa import auc  # noqa: E402

OUT = ROOT / "report_figs_gate"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
RUNS = {"Baseline (no caption, 1b-A)": "vqa_b64_baseline/runs/nocap",
        "+ captions, dropout (1b-B, no gate)": "vqa_b64_crossfit_drop/runs/blip",
        "G1: gate + reliability loss": "vqa_gate_aux/runs/blip",
        "G2: gate, VQA loss only": "vqa_gate_noaux/runs/blip"}
CASES = ("open, caption right", "open, caption wrong", "yes/no, polarity right", "yes/no, polarity wrong")
GROUPS = ["open: color", "open: count 0-3", "open: count 4+ / other number", "open: read text / time",
          "open: other (objects, places, activities ...)", "yes/no: is there / are there", "yes/no: other"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def style(ax, ylabel, ymin, ymax):
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("->", OUT / name)


def scores_on(run, s, answers):
    """Finglish: test_scores-e yek run roye list-e moshtarak-e javab ha (float32)."""
    a = load(ROOT / f"{run}_s{s}/answers.json")
    sc = np.load(ROOT / f"{run}_s{s}/test_scores.npy").astype(np.float32)
    full = np.zeros((sc.shape[0], len(answers)), dtype=np.float32)
    idx = {x: i for i, x in enumerate(answers)}
    full[:, [idx[x] for x in a]] = sc
    return full


def main():
    recs = load(ROOT / "outputs/blip_strong/split/test.json")
    caps = load(ROOT / "vqa_strong_crossfit/captions/captions_crossfit.json")
    info = []
    for r in recs:
        a = analyse(r, caps[str(r["question_id"])])
        a["group"] = group(r, a["gt"])
        info.append(a)
    res = {}

    # --- 1. models: overall, types, shuffled, cases (mean of 2 seeds) ---
    P = {}
    table = {}
    for name, run in RUNS.items():
        R = [load(ROOT / f"{run}_s{s}/results.json") for s in (0, 1)]
        P[name] = [{p["question_id"]: p for p in load(ROOT / f"{run}_s{s}/test_predictions.json")} for s in (0, 1)]
        m = lambda f: statistics.mean(f(x) for x in R)  # noqa: E731
        sd = lambda f: statistics.stdev(f(x) for x in R)  # noqa: E731
        row = {"dev": m(lambda x: x["dev"]["acc"]), "test": m(lambda x: x["test"]["acc"]),
               "test_sd": sd(lambda x: x["test"]["acc"]), "best_epoch": [x["best_epoch"] for x in R],
               **{t: m(lambda x, t=t: x["test"]["answer_type"][t]["acc"]) for t in ("yes/no", "number", "other")},
               "shuffled": m(lambda x: x["test_shuffled_captions"]["acc"]) if "test_shuffled_captions" in R[0] else None,
               "params_M": R[0]["params_M"], "caption_dropout": R[0].get("caption_dropout"),
               "caption_gate": R[0].get("caption_gate"), "gate_aux_weight": R[0].get("gate_aux_weight")}
        # Finglish: 4 halat — baraye hame model ha ba hamoon caption + label (az analyse)
        groups = defaultdict(list)
        for i, (rec, a) in enumerate(zip(recs, info)):
            yn = a["gt"] in ("yes", "no")
            if not yn and not a["gt"]:
                continue
            groups[CASES[2 * int(yn) + int(not a["right"])]].append(rec["question_id"])
        row["cases"] = {}
        for k in CASES:
            qs = groups[k]
            acc = 100 * statistics.mean(statistics.mean(P[name][s][q]["score"] for q in qs) for s in (0, 1))
            rr = [statistics.mean(P[name][s][q]["r"] for q in qs) for s in (0, 1)] if "r" in P[name][0][qs[0]] else None
            row["cases"][k] = {"n": len(qs), "acc": acc, "r": statistics.mean(rr) if rr else None}
        if "r" in next(iter(P[name][0].values())):
            row["shuffled_r"] = m(lambda x: x["test_shuffled_captions"]["gate_mean"])
            row["auc"] = m(lambda x: x["test"]["gate"]["auc"])
        table[name] = row
    res["models"] = table

    # --- 2. what did r learn? (G1) ---
    g1 = P["G1: gate + reliability loss"]
    qids = [r["question_id"] for r in recs]
    r_mean = {q: statistics.mean(g1[s][q]["r"] for s in (0, 1)) for q in qids}
    lab = {r["question_id"]: int(a["right"]) for r, a in zip(recs, info) if a["gt"]}
    soft = {r["question_id"]: a["soft"] for r, a in zip(recs, info) if a["gt"]}
    L = [q for q in qids if q in lab]
    res["auc_r_seed_mean"] = auc([r_mean[q] for q in L], [lab[q] for q in L])
    # Finglish: label-e soft (har javab-e ensan): caption "ghabel-e ghabool" = soft ≥ 0.3
    res["auc_r_soft_label"] = auc([r_mean[q] for q in L], [int(soft[q] >= 0.3) for q in L])
    # Finglish: AUC-e prior-e "no'-e soal" (question_type), 2-fold roye test (image-disjoint nist, vali bedoon tuning)
    qt = {r["question_id"]: r["question_type"] for r in recs}
    rng = random.Random(0)
    fold = {q: rng.random() < 0.5 for q in L}
    prior = {}
    for f in (True, False):
        stat = defaultdict(list)
        for q in L:
            if fold[q] != f:
                stat[qt[q]].append(lab[q])
        base = statistics.mean(lab[q] for q in L if fold[q] != f)
        for q in L:
            if fold[q] == f:
                v = stat.get(qt[q])
                prior[q] = statistics.mean(v) if v else base
    res["auc_question_type_prior"] = auc([prior[q] for q in L], [lab[q] for q in L])
    per_group = {}
    for g in GROUPS:
        qs = [r["question_id"] for r, a in zip(recs, info) if a["group"] == g and r["question_id"] in lab]
        ok = [q for q in qs if lab[q] == 1]
        bad = [q for q in qs if lab[q] == 0]
        per_group[g] = {"n": len(qs), "right_share": 100 * len(ok) / len(qs),
                        "r_right": statistics.mean(r_mean[q] for q in ok) if ok else None,
                        "r_wrong": statistics.mean(r_mean[q] for q in bad) if bad else None,
                        "auc_within": auc([r_mean[q] for q in qs], [lab[q] for q in qs])}
    res["per_group"] = per_group
    # Finglish: r baraye caption haye ghalat ke "javab-e ensan-e dige" daran (soft > 0) vs vaghean ghalat
    wrong_soft = [q for q in L if lab[q] == 0 and soft[q] > 0]
    wrong_hard = [q for q in L if lab[q] == 0 and soft[q] == 0 and q in soft and info[qids.index(q)]["gt"] not in ("yes", "no")]
    res["r_wrong_but_other_human_answer"] = statistics.mean(r_mean[q] for q in wrong_soft)
    res["r_wrong_open_really"] = statistics.mean(r_mean[q] for q in wrong_hard)

    # --- 3. r as a per-question switch (no tuning) ---
    answers = sorted(load(ROOT / "vqa_gate_aux/runs/blip_s0/answers.json"))
    S = {k: np.mean([scores_on(RUNS[k], s, answers) for s in (0, 1)], axis=0)
         for k in ("Baseline (no caption, 1b-A)", "+ captions, dropout (1b-B, no gate)", "G1: gate + reliability loss")}
    rv = np.array([r_mean[q] for q in qids], dtype=np.float32)[:, None]

    def acc(scores):
        pred = scores.argmax(1)
        return 100 * statistics.mean(vqa_score(answers[p], r["answers"]) for p, r in zip(pred, recs))

    base = S["Baseline (no caption, 1b-A)"]
    sw = {}
    for capname in ("G1: gate + reliability loss", "+ captions, dropout (1b-B, no gate)"):
        c = S[capname]
        sw[capname] = {"caption model, 2 seeds averaged": acc(c),
                       "soft switch r·caption + (1−r)·baseline": acc(rv * c + (1 - rv) * base),
                       "hard switch (r < 0.5 → baseline answer)": acc(np.where(rv < 0.5, base, c)),
                       "fixed mix 0.5·caption + 0.5·baseline": acc(0.5 * c + 0.5 * base)}
    sw["baseline, 2 seeds averaged"] = acc(base)
    res["switch"] = sw
    # Finglish: dar har bazeye r: accuracy-e baseline vs model-e caption (1b-B, G1) — switch chizi dare ke begire?
    B, D = P["Baseline (no caption, 1b-A)"], P["+ captions, dropout (1b-B, no gate)"]
    sc = lambda M, q: 100 * statistics.mean(m[q]["score"] for m in M)  # noqa: E731
    bins = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        b = [q for q in qids if lo <= r_mean[q] < hi]
        bl = [lab[q] for q in b if q in lab]
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": len(b), "right": 100 * statistics.mean(bl),
                     "base": statistics.mean(sc(B, q) for q in b), "cap": statistics.mean(sc(D, q) for q in b),
                     "g1": statistics.mean(sc(g1, q) for q in b),
                     "best": statistics.mean(max(sc(B, q), sc(D, q)) for q in b)})
    res["r_bins"] = bins
    (ROOT / "gate_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))

    # --- 4. figures ---
    names = list(RUNS)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    style(ax, "VQA accuracy (%)", 0, 112)
    ax.set_yticks(range(0, 101, 20))
    w = 0.2
    for j, nm in enumerate(names):
        for k, c in enumerate(CASES):
            v = table[nm]["cases"][c]["acc"]
            x = k + (j - 1.5) * w
            ax.bar(x, v, w * 0.9, color=SLOT[j], label=nm if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(range(4), [f"{c}\n(n={table[names[0]]['cases'][c]['n']:,})" for c in CASES], fontsize=7.6)
    ax.set_title("The four caption cases (test, mean of 2 seeds)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.4)
    save(fig, "fig1_cases.png")

    # R distribution, G1
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    edges = np.linspace(0, 1, 26)
    for j, (lab_v, nm) in enumerate(((1, "caption right"), (0, "caption wrong"))):
        v = [r_mean[q] for q in L if lab[q] == lab_v]
        ax.hist(v, bins=edges, color=SLOT[j], alpha=0.75, label=f"{nm} (n={len(v):,}, mean r = {statistics.mean(v):.2f})",
                edgecolor=SURFACE, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("gate r (G1, mean of 2 seeds)")
    ax.set_ylabel("test questions")
    ax.set_title(f"G1: r for right vs wrong captions (AUC {res['auc_r_seed_mean']:.3f})", loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    save(fig, "fig2_r_distribution.png")

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for j, nm in enumerate(names[1:]):
        H = [load(ROOT / f"{RUNS[nm]}_s{s}/history.json") for s in (0, 1)]
        ep = [x["epoch"] for x in H[0]]
        axes[0].plot(ep, [statistics.mean(h[i]["dev"]["acc"] for h in H) for i in range(len(ep))], color=SLOT[j + 1],
                     linewidth=2, marker="o", markersize=3.5, markeredgecolor=SURFACE, label=nm)
        if "gate_mean" in H[0][0]["dev"]:
            axes[1].plot(ep, [statistics.mean(h[i]["dev"]["gate_mean"] for h in H) for i in range(len(ep))],
                         color=SLOT[j + 1], linewidth=2, marker="o", markersize=3.5, markeredgecolor=SURFACE,
                         label=f"{nm.split(':')[0]}: mean r")
            if nm.startswith("G1"):
                axes[1].plot(ep, [statistics.mean(h[i]["dev"]["gate"]["auc"] for h in H) for i in range(len(ep))],
                             color=SLOT[j + 1], linewidth=2, linestyle="--", label="G1: r-AUC")
    style(axes[0], "dev accuracy (%)", 50, 71)
    axes[0].set_title("Dev accuracy per epoch", loc="left")
    axes[0].legend(frameon=False, fontsize=6.6, loc="lower right")
    style(axes[1], "value", 0.4, 1.05)
    axes[1].set_title("Gate on dev per epoch", loc="left")
    axes[1].axvline(2.5, color=INK2, linewidth=0.8, linestyle=":")
    axes[1].text(2.6, 0.43, "gate acts", fontsize=6.8, color=INK2)
    axes[1].legend(frameon=False, fontsize=6.6, loc="center right")
    for a in axes:
        a.set_xlabel("epoch")
        a.set_xticks(range(1, 14, 2))
    save(fig, "fig3_training.png")

    # per group r
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    style(ax, "mean r (G1)", 0, 1.08)
    labels = ["color", "count\n0-3", "count 4+ /\nother number", "read text\n/ time", "other\nopen",
              "y/n: is there\n/ are there", "y/n:\nother"]
    for j, (key, nm) in enumerate((("r_right", "caption right"), ("r_wrong", "caption wrong"))):
        for k, g in enumerate(GROUPS):
            v = per_group[g][key]
            x = k + (j - 0.5) * 0.36
            ax.bar(x, v, 0.33, color=SLOT[j], label=nm if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks(range(len(GROUPS)), [f"{l}\nAUC {per_group[g]['auc_within']:.2f}" for l, g in zip(labels, GROUPS)],
                  fontsize=7)
    ax.set_title("G1: mean r per question group, right vs wrong captions", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.4)
    save(fig, "fig4_r_per_group.png")

    # accuracy per r bin
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    style(ax, "VQA accuracy (%)", 0, 108)
    series = (("base", "Baseline (no caption, 1b-A)"), ("cap", "+ captions, dropout (1b-B)"), ("g1", "G1"))
    for j, (key, nm) in enumerate(series):
        for k, b in enumerate(bins):
            x = k + (j - 1) * 0.26
            ax.bar(x, b[key], 0.24, color=SLOT[j], label=nm if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, b[key] + 0.8, f"{b[key]:.1f}", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks(range(len(bins)), [f"r {b['lo']:.1f}–{b['hi']:.1f}\nn = {b['n']:,}\ncaptions right {b['right']:.0f}%"
                                     for b in bins], fontsize=7)
    ax.set_title("Accuracy by G1's r: where r is low, the baseline is no better", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7.4)
    save(fig, "fig5_r_bins.png")


if __name__ == "__main__":
    main()
