"""Final models (T3: BLIP-large captions + dropout, T4: baseline), trained on all train + val, answers for test2015.

Finglish: test2015 javab nadare → inja faghat chiz haei ke bedoon javab ma'na daran:
  - sehhat-e file ha (447,793 soal, format-e EvalAI), no'-e javab ha (yes / no / adad / baghie)
  - tavafogh-e seed ha va tavafogh-e model-e caption ba baseline
  - kalibre ba 10k val (hamoon model ha bedoon dev/test dar train): vaghti caption javab ro avaz mikone, ki dorost-e?
Khorooji: final_runs_analysis.json + report_figs_final/*.png

    python scripts/final_runs_analysis.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import vqa_score  # noqa: E402

OUT = ROOT / "report_figs_final"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
T3, T4 = ROOT / "vqa_final_t3/vqa_final", ROOT / "vqa_final_t4/vqa_final_baseline"
FILES = {"cap_s0": T3 / "vqa_final_blip_s0_test2015.json", "cap_s1": T3 / "vqa_final_blip_s1_test2015.json",
         "cap_ens": T3 / "vqa_final_ensemble_test2015.json",
         "base_s0": T4 / "vqa_final_baseline_nocap_s0_test2015.json",
         "base_s1": T4 / "vqa_final_baseline_nocap_s1_test2015.json",
         "base_ens": T4 / "vqa_final_baseline_ensemble_test2015.json"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def kind(a: str) -> str:
    return a if a in ("yes", "no") else ("number" if a.isdigit() else "other")


def style(ax, ylabel, lo, hi):
    ax.set_ylim(lo, hi)
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


def val_answers(run: str):
    """Finglish: 10k val: javab-e har seed va ensemble-e 2 seed (miangin-e score ha), az test_scores.npy."""
    out = {}
    sc = []
    for s in (0, 1):
        a = load(ROOT / f"{run}_s{s}/answers.json")
        x = np.load(ROOT / f"{run}_s{s}/test_scores.npy").astype(np.float32)
        out[f"s{s}"] = [a[i] for i in x.argmax(1)]
        sc.append((a, x))
    assert sc[0][0] == sc[1][0]
    out["ens"] = [sc[0][0][i] for i in ((sc[0][1] + sc[1][1]) / 2).argmax(1)]
    return out


def main():
    A = {k: {r["question_id"]: r["answer"] for r in load(f)} for k, f in FILES.items()}
    qids = list(A["cap_ens"])
    res = {"n": len(qids), "files_ok": all(set(A[k]) == set(qids) for k in A)}
    agree = lambda a, b, qs: 100 * sum(A[a][q] == A[b][q] for q in qs) / len(qs)  # noqa: E731
    res["test2015_agreement"] = {"caption seeds": agree("cap_s0", "cap_s1", qids),
                                 "baseline seeds": agree("base_s0", "base_s1", qids),
                                 "caption vs baseline (ensembles)": agree("cap_ens", "base_ens", qids)}
    res["test2015_kinds"] = {k: {c: 100 * v / len(qids) for c, v in Counter(kind(A[k][q]) for q in qids).items()}
                             for k in ("cap_ens", "base_ens")}
    res["test2015_top_answers"] = {k: Counter(A[k].values()).most_common(10) for k in ("cap_ens", "base_ens")}
    # Finglish: vaghti javab fargh dare, che no' javab haei avaz mishan
    diff = [q for q in qids if A["cap_ens"][q] != A["base_ens"][q]]
    res["test2015_changed"] = {"n": len(diff), "share": 100 * len(diff) / len(qids),
                               "kind_pairs": Counter(f"{kind(A['base_ens'][q])} -> {kind(A['cap_ens'][q])}"
                                                     for q in diff).most_common(8)}

    # --- calibration on our 10k val (same setup, trained without dev/test) ---
    recs = load(ROOT / "outputs/blip_strong/split/test.json")
    V = {"cap": val_answers("vqa_large_drop/runs/blip"), "base": val_answers("vqa_b64_baseline/runs/nocap")}
    sc = {m: {k: [vqa_score(a, r["answers"]) for a, r in zip(V[m][k], recs)] for k in V[m]} for m in V}
    n = len(recs)
    va = lambda x, y: 100 * sum(a == b for a, b in zip(x, y)) / n  # noqa: E731
    dis = [i for i in range(n) if V["cap"]["ens"][i] != V["base"]["ens"][i]]
    agr = [i for i in range(n) if V["cap"]["ens"][i] == V["base"]["ens"][i]]
    cap_better = sum(sc["cap"]["ens"][i] > sc["base"]["ens"][i] for i in dis)
    base_better = sum(sc["cap"]["ens"][i] < sc["base"]["ens"][i] for i in dis)
    res["val_calibration"] = {
        "agreement": {"caption seeds": va(V["cap"]["s0"], V["cap"]["s1"]),
                      "baseline seeds": va(V["base"]["s0"], V["base"]["s1"]),
                      "caption vs baseline (ensembles)": va(V["cap"]["ens"], V["base"]["ens"])},
        "acc_ens": {m: 100 * statistics.mean(sc[m]["ens"]) for m in sc},
        "acc_single_mean": {m: 100 * statistics.mean(statistics.mean(sc[m][k]) for k in ("s0", "s1")) for m in sc},
        "disagree": {"n": len(dis), "share": 100 * len(dis) / n,
                     "caption_acc": 100 * statistics.mean(sc["cap"]["ens"][i] for i in dis),
                     "baseline_acc": 100 * statistics.mean(sc["base"]["ens"][i] for i in dis),
                     "caption_better": 100 * cap_better / len(dis), "baseline_better": 100 * base_better / len(dis)},
        "agree_acc": 100 * statistics.mean(sc["cap"]["ens"][i] for i in agr)}
    # Finglish: takhmin-e sade: hamoon kesr-e "taghir" ba hamoon sood dar har taghir
    c = res["val_calibration"]["disagree"]
    gain_per_change = (c["caption_acc"] - c["baseline_acc"]) / 100
    res["projected_gain_test2015"] = res["test2015_changed"]["share"] * gain_per_change
    (ROOT / "final_runs_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "test2015_top_answers"}, indent=1))

    # --- figures ---
    # 1. training curves (train loss; dev here = seen in training)
    H = {"Final + BLIP-large captions (T3)": [load(T3 / f"runs/blip_s{s}/history.json") for s in (0, 1)],
         "Final baseline (T4)": [load(T4 / f"runs/nocap_s{s}/history.json") for s in (0, 1)],
         "Round 4 (dev/test held out)": [load(ROOT / f"vqa_large_drop/runs/blip_s{s}/history.json") for s in (0, 1)]}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for j, (nm, hs) in enumerate(H.items()):
        ep = [x["epoch"] for x in hs[0]]
        loss = [statistics.mean(h[i]["train_loss"] for h in hs) for i in range(len(ep))]
        dev = [statistics.mean(h[i]["dev"]["acc"] for h in hs) for i in range(len(ep))]
        axes[0].plot(ep[1:], loss[1:], color=SLOT[j], linewidth=2, marker="o", markersize=3.5, markeredgecolor=SURFACE,
                     label=nm)
        axes[1].plot(ep, dev, color=SLOT[j], linewidth=2, marker="o", markersize=3.5, markeredgecolor=SURFACE, label=nm)
    style(axes[0], "train loss", 2.3, 4.2)
    axes[0].set_title("Training loss (epochs 2+)", loc="left")
    style(axes[1], "dev accuracy (%)", 45, 82)
    axes[1].set_title("Dev accuracy (seen in final runs!)", loc="left")
    axes[1].legend(frameon=False, fontsize=6.6, loc="lower right")
    for a in axes:
        a.set_xlabel("epoch")
        a.set_xticks(range(1, 14, 2))
    save(fig, "fig1_training.png")

    # 2. agreement: test2015 vs val
    labs = list(res["test2015_agreement"])
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    style(ax, "same answer (%)", 60, 100)
    w = 0.36
    for j, (nm, vals) in enumerate((("test2015 (final models)", [res["test2015_agreement"][l] for l in labs]),
                                    ("our 10k val (round-4 models)",
                                     [res["val_calibration"]["agreement"][l] for l in labs]))):
        for k, v in enumerate(vals):
            x = k + (j - 0.5) * w
            ax.bar(x, v, w * 0.9, color=SLOT[j], label=nm if k == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(labs)), labs, fontsize=7.6)
    ax.set_title("How often two models give the same answer", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.4)
    save(fig, "fig2_agreement.png")


if __name__ == "__main__":
    main()
