"""Stronger captioners on the same 10,000 fold-1 QD rows: BLIP-base (current) vs BLIP-large (fine-tuned) vs BLIP-2 zero-shot.

Finglish: har row → AnswerHit (javab dar caption) ya polarity (yes/no); moghayese-ye paired (bootstrap) ba captioner-e
feli, goroh-e soal (rang, shomaresh, matn/saat, baghie, yes/no), bias-e "no", va "ki chi midoone" (union).
Khorooji: outputs/round3/stronger_captioners.json + report_figs_scale/*.png

    python scripts/stronger_captioner_analysis.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import answer_in_caption, caption_negated  # noqa: E402
from train import tok  # noqa: E402

R3 = ROOT / "outputs/round3"
MODELS = {
    "BLIP-base, fine-tuned (current)": ROOT / "outputs/fold_eval/fold0_current.json",
    "BLIP-large, fine-tuned": R3 / "model_b/b_blip_fold0_large/fold_eval.json",
    "BLIP-2 FlanT5-XL, zero-shot (short)": R3 / "blip2_probe/blip2_flan_t5_xl_short.json",
    "BLIP-2 FlanT5-XL, zero-shot (sentence)": R3 / "blip2_probe/blip2_flan_t5_xl_sentence.json",
    "BLIP-2 OPT-2.7B, zero-shot (short)": R3 / "blip2_probe/blip2_opt_2_7b_short.json",
    "BLIP-2 OPT-2.7B, zero-shot (sentence)": R3 / "blip2_probe/blip2_opt_2_7b_sentence.json",
}
READ = re.compile(r"\b(brand|name|written|say|says|sign|letter|word|logo|read|team|company|airline|website|text|"
                  r"number on|jersey|what time|clock|label|title)\b")
GROUPS = ["color", "count 0-3", "count 4+", "read text / time", "other open", "yes/no"]
OUT = ROOT / "report_figs_scale"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def group(q: str, a: str) -> str:
    q, a = q.lower(), a.lower().strip()
    if a in ("yes", "no"):
        return "yes/no"
    if READ.search(q) or ":" in a:
        return "read text / time"
    if a.isdigit():
        return "count 0-3" if int(a) <= 3 else "count 4+"
    if "color" in q:
        return "color"
    return "other open"


def right(pred: str, ans: str) -> int:
    a, c = ans.lower().strip(), tok(pred)
    if a in ("yes", "no"):
        return int(caption_negated(c) == (a == "no"))
    return int(answer_in_caption(tok(a), c))


def boot(diff, n=1000, seed=0):
    rng = random.Random(seed)
    m = len(diff)
    xs = sorted(100 * sum(diff[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


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


def main():
    P = {k: {p["question_id"]: p for p in json.loads(Path(v).read_text(encoding="utf-8"))["predictions"]}
         for k, v in MODELS.items()}
    base = list(MODELS)[0]
    qids = list(P[base])
    for k in P:
        assert set(P[k]) == set(qids), k
    rows = {q: P[base][q] for q in qids}
    grp = {q: group(r["question"], str(r["answer"])) for q, r in rows.items()}
    yn = {q for q in qids if grp[q] == "yes/no"}
    op = [q for q in qids if q not in yn]
    hit = {k: {q: right(P[k][q]["pred"], str(rows[q]["answer"])) for q in qids} for k in P}
    res = {"n_open": len(op), "n_yes_no": len(yn), "models": {}}
    for k in P:
        o = {"AnswerHit": 100 * sum(hit[k][q] for q in op) / len(op),
             "Polarity": 100 * sum(hit[k][q] for q in yn) / len(yn),
             "polarity_answer_yes": 100 * sum(hit[k][q] for q in yn if rows[q]["answer"].lower() == "yes")
             / sum(rows[q]["answer"].lower() == "yes" for q in yn),
             "polarity_answer_no": 100 * sum(hit[k][q] for q in yn if rows[q]["answer"].lower() == "no")
             / sum(rows[q]["answer"].lower() == "no" for q in yn),
             "negated_share": 100 * sum(caption_negated(tok(P[k][q]["pred"])) for q in yn) / len(yn),
             "mean_words": sum(len(tok(P[k][q]["pred"])) for q in qids) / len(qids),
             "groups": {g: 100 * sum(hit[k][q] for q in qids if grp[q] == g) / max(1, sum(grp[q] == g for q in qids))
                        for g in GROUPS}}
        if k != base:
            d_open = [hit[k][q] - hit[base][q] for q in op]
            d_yn = [hit[k][q] - hit[base][q] for q in yn]
            o["d_AnswerHit"] = 100 * sum(d_open) / len(d_open)
            o["d_AnswerHit_ci"] = boot(d_open)
            o["d_Polarity"] = 100 * sum(d_yn) / len(d_yn)
            o["d_Polarity_ci"] = boot(d_yn, seed=1)
            # Finglish: ki chi midoone: faghat in model / faghat BLIP-base / har do (open)
            o["only_this"] = 100 * sum(hit[k][q] and not hit[base][q] for q in op) / len(op)
            o["only_base"] = 100 * sum(hit[base][q] and not hit[k][q] for q in op) / len(op)
            o["union_AnswerHit"] = 100 * sum(hit[k][q] or hit[base][q] for q in op) / len(op)
        res["models"][k] = o
    res["group_n"] = {g: sum(grp[q] == g for q in qids) for g in GROUPS}
    # Finglish: mesal: BLIP-large dorost, BLIP-base ghalat (va baraks)
    lg = "BLIP-large, fine-tuned"
    ex = [q for q in op if hit[lg][q] and not hit[base][q]][:6] + [q for q in op if hit[base][q] and not hit[lg][q]][:3]
    fl = "BLIP-2 FlanT5-XL, zero-shot (short)"
    res["examples"] = [{"question": rows[q]["question"], "answer": rows[q]["answer"], "base": P[base][q]["pred"],
                        "large": P[lg][q]["pred"], "blip2": P[fl][q]["pred"]} for q in ex]
    out = R3 / "stronger_captioners.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in o.items() if kk != "groups"}
                      for k, o in res["models"].items()}, indent=1, ensure_ascii=False))
    for k, o in res["models"].items():
        print(f"{k:42s}", "  ".join(f"{g} {v:.1f}" for g, v in o["groups"].items()))

    # --- figures ---
    show = [base, lg, "BLIP-2 FlanT5-XL, zero-shot (short)", "BLIP-2 OPT-2.7B, zero-shot (short)"]
    cols = ["AnswerHit", "Polarity", "polarity_answer_yes", "polarity_answer_no"]
    labs = ["AnswerHit\n(open, 6,719)", "Polarity\n(yes/no, 3,281)", "Polarity\nanswer = yes", "Polarity\nanswer = no"]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    style(ax, "% of rows", 20, 100)
    w = 0.2
    for j, k in enumerate(show):
        for i, c in enumerate(cols):
            v = res["models"][k][c]
            x = i + (j - 1.5) * w
            ax.bar(x, v, w * 0.9, color=SLOT[j], label=k if i == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(range(len(cols)), labs, fontsize=7.6)
    ax.set_title("Captioners on the same 10,000 unseen fold-1 rows", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.4)
    save(fig, "fig1_overall.png")

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    style(ax, "AnswerHit / polarity (%)", 0, 100)
    for j, k in enumerate(show):
        for i, g in enumerate(GROUPS):
            v = res["models"][k]["groups"][g]
            x = i + (j - 1.5) * w
            ax.bar(x, v, w * 0.9, color=SLOT[j], label=k if i == 0 else None, edgecolor=SURFACE, linewidth=1.2)
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(range(len(GROUPS)), [f"{g}\n(n={res['group_n'][g]:,})" for g in GROUPS], fontsize=7.4)
    ax.set_title("By question group", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.4)
    save(fig, "fig2_groups.png")

    # training curve BLIP-large vs round-2 variant A (both: val AnswerHit per epoch)
    rx = re.compile(r"epoch (\d+)/\d+ train_loss=([\d.]+) val_CIDEr=([\d.]+).*val_AnswerHit=([\d.]+)")
    runs = {"BLIP-large (frozen ViT-L, 4 ep)": R3 / "model_b/b_blip_fold0_large/train.log",
            "BLIP-base, top-4 ViT layers (round 2, A)": ROOT / "outputs/round2/model_b/b_blip_fold0_v2A/train.log"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for j, (name, f) in enumerate(runs.items()):
        r = [tuple(float(x) for x in m.groups()) for m in map(rx.search, Path(f).read_text().splitlines()) if m]
        for ax, col, lab in ((axes[0], 3, "val AnswerHit (%)"), (axes[1], 2, "val CIDEr")):
            ys = [x[col] * (100 if col == 3 else 1) for x in r]
            ax.plot([x[0] for x in r], ys, color=SLOT[j + 1], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, label=name)
            ax.text(r[-1][0] + 0.15, ys[-1], f"{ys[-1]:.{1 if col == 3 else 2}f}", va="center", fontsize=7, color=INK)
            ax.set_ylabel(lab)
    for ax, (lo, hi), t in ((axes[0], (51, 59), "Val AnswerHit per epoch"), (axes[1], (5.8, 6.5), "Val CIDEr per epoch")):
        style(ax, ax.get_ylabel(), lo, hi)
        ax.set_xlabel("epoch")
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.7, 6.9)
        ax.set_title(t, loc="left")
    axes[0].legend(frameon=False, fontsize=6.8, loc="lower right")
    save(fig, "fig3_training.png")


if __name__ == "__main__":
    main()
