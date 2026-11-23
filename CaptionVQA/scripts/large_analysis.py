"""VQA with BLIP-large captions (round 4) vs the baseline and vs BLIP-base captions — analysis + figures.

Finglish: se model ba hamoon training (MCAN, batch 64, GloVe 840B), test = 10,001 soal:
  base  = 1b-A (bedoon caption) | blip  = 1b-B (BLIP-base + caption dropout 0.3) | large = BLIP-large + caption dropout 0.3
  - paired bootstrap (miangin-e 2 seed baraye har soal)
  - goroh-e soal (caption_error_analysis.group), 4 halat ba caption-e khod-e har model
  - soal haei ke caption-eshoon az BLIP-base be BLIP-large avaz shod (ghalat→dorost, …)
Khorooji: large_analysis.json + report_figs_large/*.png

    python scripts/large_analysis.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from caption_error_analysis import analyse, group  # noqa: E402

OUT = ROOT / "report_figs_large"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
RUNS = {"base": ("Baseline, no caption (1b-A)", "vqa_b64_baseline/runs/nocap"),
        "blip": ("+ BLIP-base captions + dropout (1b-B)", "vqa_b64_crossfit_drop/runs/blip"),
        "large": ("+ BLIP-large captions + dropout", "vqa_large_drop/runs/blip")}
CAPS = {"blip": "vqa_strong_crossfit/captions/captions_crossfit.json",
        "large": "vqa_large_drop/captions/captions_crossfit.json"}
GROUPS = ["open: color", "open: count 0-3", "open: count 4+ / other number", "open: read text / time",
          "open: other (objects, places, activities ...)", "yes/no: is there / are there", "yes/no: other"]
CASES = ("open, caption right", "open, caption wrong", "yes/no, polarity right", "yes/no, polarity wrong")


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def style(ax, ylabel, lo, hi, axis="y"):
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    (ax.set_ylabel if axis == "y" else ax.set_xlabel)(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("->", OUT / name)


def boot(d, n=1000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    xs = sorted(100 * sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return [xs[int(0.025 * n)], xs[int(0.975 * n) - 1]]


def bars(ax, labels, series, vals, fs=6.6, w=None):
    w = w or 0.8 / len(series)
    for j, s in enumerate(series):
        for k, lab in enumerate(labels):
            v = vals[s][k]
            x = k + (j - (len(series) - 1) / 2) * w
            ax.bar(x, v, w * 0.9, color=SLOT[j], label=s if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=fs, color=INK)
    ax.set_xticks(range(len(labels)), labels)


def main():
    recs = load("outputs/blip_strong/split/test.json")
    qids = [r["question_id"] for r in recs]
    caps = {k: load(v) for k, v in CAPS.items()}
    info = {k: {r["question_id"]: analyse(r, caps[k][str(r["question_id"])]) for r in recs} for k in caps}
    grp = {r["question_id"]: group(r, info["blip"][r["question_id"]]["gt"]) for r in recs}
    R, S = {}, {}
    for k, (_, run) in RUNS.items():
        R[k] = [load(f"{run}_s{s}/results.json") for s in (0, 1)]
        P = [{p["question_id"]: p["score"] for p in load(f"{run}_s{s}/test_predictions.json")} for s in (0, 1)]
        S[k] = {q: statistics.mean(p[q] for p in P) for q in qids}
    res = {"models": {}, "paired": {}, "groups": {}, "cases": {}, "transitions": {}}
    for k, (name, _) in RUNS.items():
        m = lambda f: statistics.mean(f(x) for x in R[k])  # noqa: E731
        res["models"][k] = {
            "name": name, "dev": m(lambda x: x["dev"]["acc"]), "test": m(lambda x: x["test"]["acc"]),
            "test_seeds": [x["test"]["acc"] for x in R[k]],
            "test_sd": statistics.stdev(x["test"]["acc"] for x in R[k]),
            **{t: m(lambda x, t=t: x["test"]["answer_type"][t]["acc"]) for t in ("yes/no", "number", "other")},
            "shuffled": m(lambda x: x["test_shuffled_captions"]["acc"]) if "test_shuffled_captions" in R[k][0] else None,
            "best_epoch": [x["best_epoch"] for x in R[k]], "vocab": R[k][0]["vocab_size"], "params_M": R[k][0]["params_M"]}
    for a, b in (("large", "blip"), ("large", "base"), ("blip", "base")):
        d = [S[a][q] - S[b][q] for q in qids]
        res["paired"][f"{a}-{b}"] = {"diff": 100 * statistics.mean(d), "ci": boot(d)}
    # question groups
    for g in GROUPS + ["all"]:
        qs = qids if g == "all" else [q for q in qids if grp[q] == g]
        res["groups"][g] = {"n": len(qs),
                            "cap_right_blip": 100 * statistics.mean(info["blip"][q]["right"] for q in qs),
                            "cap_right_large": 100 * statistics.mean(info["large"][q]["right"] for q in qs),
                            **{k: 100 * statistics.mean(S[k][q] for q in qs) for k in S}}
    # 4 cases with each captioner's own labels
    for capk in ("blip", "large"):
        cs = defaultdict(list)
        for q in qids:
            a = info[capk][q]
            if not a["gt"]:
                continue
            yn = a["gt"] in ("yes", "no")
            cs[CASES[2 * int(yn) + int(not a["right"])]].append(q)
        res["cases"][capk] = {c: {"n": len(cs[c]), **{k: 100 * statistics.mean(S[k][q] for q in cs[c]) for k in S}}
                              for c in CASES}
    # transitions: caption base → large
    for lab, f in (("wrong → right", lambda q: not info["blip"][q]["right"] and info["large"][q]["right"]),
                   ("right → wrong", lambda q: info["blip"][q]["right"] and not info["large"][q]["right"]),
                   ("right → right", lambda q: info["blip"][q]["right"] and info["large"][q]["right"]),
                   ("wrong → wrong", lambda q: not info["blip"][q]["right"] and not info["large"][q]["right"])):
        qs = [q for q in qids if info["blip"][q]["gt"] and f(q)]
        res["transitions"][lab] = {"n": len(qs), **{k: 100 * statistics.mean(S[k][q] for q in qs) for k in S}}
    # question types with the largest change large vs blip (n ≥ 100)
    qt = defaultdict(list)
    for r in recs:
        qt[r["question_type"]].append(r["question_id"])
    rows = []
    for t, qs in qt.items():
        if len(qs) >= 100:
            rows.append({"type": t, "n": len(qs), **{k: 100 * statistics.mean(S[k][q] for q in qs) for k in S}})
    res["question_types"] = sorted(rows, key=lambda x: -x["n"])
    (ROOT / "large_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "question_types"}, indent=1))

    # --- figures ---
    names = [RUNS[k][0] for k in ("base", "blip", "large")]
    labs = ["Overall", "Yes/no", "Number", "Other"]
    vals = {RUNS[k][0]: [res["models"][k]["test"], res["models"][k]["yes/no"], res["models"][k]["number"],
                         res["models"][k]["other"]] for k in ("base", "blip", "large")}
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    style(ax, "VQA accuracy (%)", 40, 92)
    bars(ax, labs, names, vals, fs=6.8)
    ax.set_title("Test accuracy by answer type (10,001 questions, mean of 2 seeds)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.4)
    save(fig, "fig1_answer_types.png")

    short = ["color", "count\n0-3", "count 4+ /\nother number", "read text\n/ time", "other\nopen",
             "y/n: is there\n/ are there", "y/n:\nother"]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6))
    style(axes[0], "captions right (%)", 0, 100)
    bars(axes[0], short, ["BLIP-base captions", "BLIP-large captions"],
         {"BLIP-base captions": [res["groups"][g]["cap_right_blip"] for g in GROUPS],
          "BLIP-large captions": [res["groups"][g]["cap_right_large"] for g in GROUPS]}, w=0.36)
    axes[0].set_title("How often the caption is right, per question group (test)", loc="left")
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=7.4)
    style(axes[1], "VQA accuracy (%)", 0, 100)
    bars(axes[1], short, names, {RUNS[k][0]: [res["groups"][g][k] for g in GROUPS] for k in ("base", "blip", "large")},
         fs=5.6)
    axes[1].set_title("VQA accuracy per question group (test, mean of 2 seeds)", loc="left")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7)
    for a in axes:
        a.tick_params(axis="x", labelsize=7)
    save(fig, "fig2_groups.png")

    tl = list(res["transitions"])
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    style(ax, "VQA accuracy (%)", 0, 105)
    bars(ax, [f"{t}\n(n={res['transitions'][t]['n']:,})" for t in tl], names,
         {RUNS[k][0]: [res["transitions"][t][k] for t in tl] for k in ("base", "blip", "large")})
    ax.set_title("Caption from BLIP-base to BLIP-large: what happens to the answer", loc="left")
    ax.set_xlabel("caption right/wrong: BLIP-base → BLIP-large")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=7)
    save(fig, "fig3_transitions.png")

    # training curves (dev, mean of 2 seeds)
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for j, k in enumerate(("base", "blip", "large")):
        H = [load(f"{RUNS[k][1]}_s{s}/history.json") for s in (0, 1)]
        ep = [x["epoch"] for x in H[0]]
        ys = [statistics.mean(h[i]["dev"]["acc"] for h in H) for i in range(len(ep))]
        ax.plot(ep, ys, color=SLOT[j], linewidth=2, marker="o", markersize=3.5, markeredgecolor=SURFACE,
                label=RUNS[k][0])
        b = max(range(len(ys)), key=lambda i: ys[i])
        ax.text(ep[b], ys[b] + 0.5, f"{ys[b]:.1f}", ha="center", fontsize=7, color=INK)
    style(ax, "dev accuracy (%)", 45, 73)
    ax.set_xlabel("epoch")
    ax.set_xticks(range(1, 14))
    ax.set_title("Dev accuracy per epoch (mean of 2 seeds)", loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    save(fig, "fig4_training.png")

    # progress across all experiments
    prog = [("UpDn, 31k questions", 46.56, 59.79, "BLIP-base"), ("UpDn, all train2014", 60.03, 66.08, "BLIP-base"),
            ("MCAN + GloVe 6B, train + val", 65.74, 67.80, "BLIP-base"),
            ("MCAN, batch 64 + GloVe 840B", 66.39, 68.08, "BLIP-base"),
            ("+ caption dropout 0.3", None, 68.31, "BLIP-base"), ("+ caption gate (G1 / G2)", None, 68.53, "BLIP-base"),
            ("+ BLIP-large captions, dropout 0.3", None, res["models"]["large"]["test"], "BLIP-large"),
            ("Ensemble, 2 BLIP-large seeds", None, 70.27, "BLIP-large")]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = list(range(len(prog)))[::-1]
    for yi, (lab, b, c, cap) in zip(y, prog):
        if b is not None:
            ax.barh(yi + 0.19, b, 0.36, color=SLOT[0], edgecolor=SURFACE, linewidth=1.2,
                    label="no captions" if yi == y[0] else None)
            ax.text(b + 0.3, yi + 0.19, f"{b:.2f}", va="center", fontsize=7, color=INK)
        off = -0.19 if b is not None else 0
        col = SLOT[2] if cap == "BLIP-large" else SLOT[1]
        ax.barh(yi + off, c, 0.36, color=col, edgecolor=SURFACE, linewidth=1.2)
        ax.text(c + 0.3, yi + off, f"{c:.2f}", va="center", fontsize=7, color=INK, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SLOT[0], label="no captions"), Patch(color=SLOT[1], label="+ BLIP-base captions"),
                       Patch(color=SLOT[2], label="+ BLIP-large captions")],
              frameon=False, fontsize=7.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.axvline(70.67, color=INK2, linewidth=1, linestyle="--")
    ax.text(70.5, y[0] + 0.55, "paper 70.67\n(test-standard)", fontsize=6.8, color=INK2, ha="right", va="bottom")
    ax.set_yticks(y, [p[0] for p in prog], fontsize=7.6)
    style(ax, "VQA accuracy on our 10,001 test questions (%)", 40, 74, axis="x")
    ax.set_title("Progress across all experiments", loc="left")
    save(fig, "fig5_progress.png")


if __name__ == "__main__":
    main()
