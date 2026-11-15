"""Figures for CAPTIONER_COMPARISON.md (PNG, light, print) from the result JSON files.

Finglish: az test_metrics.json / grounding.json / history.json / train.log → report_figs/*.png.
Rang ha: palette-e reference (dataviz skill), slot-e sabet: A = blue, A-large = orange, BLIP = aqua;
GT / prior = gray. Har bar label-e adadi dare (aqua < 3:1 contrast → label lazem).

    python scripts/report_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import NON_LEXICAL_ANSWERS, answer_in_caption  # noqa: E402
from train import tok  # noqa: E402

OUT = ROOT / "report_figs"
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
COL = {"A": "#2a78d6", "A-large": "#eb6834", "BLIP": "#1baf7a"}
RES = {"A": ROOT / "a_full_butd", "A-large": ROOT / "a_full_butd_large", "BLIP": ROOT / "kaggle_Image_captioner_BLIP_full"}
TEST = {"A": "test_metrics.json", "A-large": "test_metrics.json", "BLIP": "test_metrics.json.json"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def style(ax, ylabel: str, ymax: float = 100) -> None:
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)


def bar(ax, x, h, w, color, label=None, fmt="{:.1f}", dy=1.2):
    """Finglish: bar ba gooshe-ye gerd (4px-ish) bala, sabet roye baseline + label-e adad."""
    r = min(w * 0.18, 0.06)
    p = FancyBboxPatch((x - w / 2, 0), w, max(h, 0.001), boxstyle=f"round,pad=0,rounding_size={r}",
                       mutation_aspect=ax.get_ylim()[1] / 2.2, facecolor=color, edgecolor=SURFACE, linewidth=1.2,
                       label=label)
    ax.add_patch(p)
    ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7.5, color=INK)


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_overall(test, gt):
    """AnswerHit + YesNoPolarity: do panel (har kodom yek axis)."""
    names = ["Ground truth", "A", "A-large", "BLIP", "Answer prior"]
    colors = [NEUTRAL, COL["A"], COL["A-large"], COL["BLIP"], NEUTRAL]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, title in ((axes[0], "AnswerHit", "AnswerHit (non-yes/no, 6,639 rows)"),
                           (axes[1], "YesNoPolarity", "Yes/no polarity (3,358 rows)")):
        style(ax, "%")
        vals = [gt[key], test["A"][key], test["A-large"][key], test["BLIP"][key], gt["prior_" + key]]
        for i, (v, c) in enumerate(zip(vals, colors)):
            bar(ax, i, 100 * v, 0.62, c)
        ax.set_xticks(range(len(names)), names, rotation=0, fontsize=7.8)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(title, loc="left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig1_overall.png")


def fig_types(preds):
    """AnswerHit per question type, 3 model + GT marker."""
    p0 = preds["A"]
    types = defaultdict(list)
    for i, p in enumerate(p0):
        t = tok(p["answer"])
        if t and " ".join(t) not in NON_LEXICAL_ANSWERS:
            types[p["rule"]].append(i)
    order = sorted(types, key=lambda t: -len(types[t]))
    label = {"llm_fallback": "open (LLM-written)", "what_color": "what color", "how_many": "how many",
             "what_is_doing": "what is … doing", "who": "who"}
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    style(ax, "AnswerHit %", 108)
    w = 0.25
    for k, t in enumerate(order):
        idx = types[t]
        g = 100 * sum(answer_in_caption(tok(p0[i]["answer"]), tok(p0[i]["gt"])) for i in idx) / len(idx)
        ax.plot([k - 0.42, k + 0.42], [g, g], color=NEUTRAL, linewidth=2, solid_capstyle="round",
                label="Ground truth" if k == 0 else None)
        for j, m in enumerate(("A", "A-large", "BLIP")):
            v = 100 * sum(answer_in_caption(tok(preds[m][i]["answer"]), tok(preds[m][i]["pred"])) for i in idx) / len(idx)
            bar(ax, k + (j - 1) * w, v, w * 0.92, COL[m], label=m if k == 0 else None, fmt="{:.0f}")
    ax.set_xticks(range(len(order)), [f"{label.get(t, t)}\n(n={len(types[t]):,})" for t in order], fontsize=7.8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_title("AnswerHit by question type (test)", loc="left")
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=7.8)
    save(fig, "fig2_types.png")


COND = [("normal", "own", "normal"), ("no_image", "own", "just question\n(no image)"),
        ("swap_image", "own", "same question,\nother image"), ("no_question", "own", "just image\n(no question)"),
        ("swap_question", "own", "same image,\nother question\n(old answer)"),
        ("swap_question", "follow", "same image,\nother question\n(new answer)")]


def fig_grounding(gr, key: str, name: str, title: str, ref: float, ref_label: str):
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    style(ax, f"{key} %" if key == "AnswerHit" else "polarity %", 100)
    w = 0.27
    conds = COND if key == "AnswerHit" else COND[:5]
    for k, (c, part, _) in enumerate(conds):
        for j, m in enumerate(("A", "A-large", "BLIP")):
            v = gr[m]["conditions"][c][part].get(key)
            bar(ax, k + (j - 1) * w, 100 * v, w * 0.92, COL[m], label=m if k == 0 else None, fmt="{:.0f}")
    ax.axhline(100 * ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(conds) - 0.33, 100 * ref, ref_label, ha="left", va="center", fontsize=7.5, color=INK2,
            clip_on=False)
    ax.set_xticks(range(len(conds)), [c[2] for c in conds], fontsize=7.3)
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7.8)
    save(fig, name)


def fig_curves(hist):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, key, title, ymax in ((axes[0], "CIDEr", "Validation CIDEr per epoch", 7),
                                 (axes[1], "AnswerHit", "Validation AnswerHit per epoch (%)", 70)):
        style(ax, "", ymax)
        ax.set_ylim(4.5 if key == "CIDEr" else 20, ymax)
        for m in ("A", "A-large", "BLIP"):
            e = [h["epoch"] for h in hist[m]]
            v = [h[key] * (100 if key == "AnswerHit" else 1) for h in hist[m]]
            ax.plot(e, v, color=COL[m], linewidth=2, marker="o", markersize=3.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=m)
            # Finglish: A va A-large akharesh rooye ham → label-e A paeen, A-large bala
            off = {"A": -0.12, "A-large": 0.12, "BLIP": 0}[m] * (1 if key == "CIDEr" else 12)
            ax.text(e[-1] + 0.4, v[-1] + off, f"{m} {v[-1]:.2f}" if key == "CIDEr" else f"{m} {v[-1]:.1f}",
                    va="center", fontsize=7.2, color=INK)
        ax.set_xlim(0.5, 18.5)
        ax.set_xlabel("epoch")
        ax.set_title(title, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_curves.png")


def main() -> None:
    tm = {m: load(RES[m] / TEST[m]) for m in RES}
    gr = {m: load(RES[m] / "grounding.json") for m in RES}
    test = {m: gr[m]["conditions"]["normal"]["own"] for m in RES}
    preds = {m: tm[m]["predictions"] for m in RES}
    p0 = preds["A"]
    lex = [p for p in p0 if tok(p["answer"]) and " ".join(tok(p["answer"])) not in NON_LEXICAL_ANSWERS]
    from metrics import caption_negated
    yn = [p for p in p0 if p["answer"] in ("yes", "no")]
    gt = {"AnswerHit": sum(answer_in_caption(tok(p["answer"]), tok(p["gt"])) for p in lex) / len(lex),
          "YesNoPolarity": sum(caption_negated(tok(p["gt"])) == (p["answer"] == "no") for p in yn) / len(yn),
          "prior_AnswerHit": gr["A"]["prior"]["AnswerHit"], "prior_YesNoPolarity": gr["A"]["prior"]["YesNoPolarity"]}
    fig_overall(test, gt)
    fig_types(preds)
    fig_grounding(gr, "AnswerHit", "fig3_grounding_answerhit.png",
                  "Grounding: AnswerHit when one input is changed (test)", gt["prior_AnswerHit"], "answer\nprior\n14.1%")
    fig_grounding(gr, "YesNoPolarity", "fig4_grounding_polarity.png",
                  "Grounding: yes/no polarity when one input is changed (test)", 0.5, "chance\n50%")
    hist = {m: load(RES[m] / "history.json") for m in ("A", "A-large")}
    hist = {m: [{"epoch": h["epoch"], "CIDEr": h["val_CIDEr"], "AnswerHit": h["val_AnswerHit"]} for h in v]
            for m, v in hist.items()}
    log = (RES["BLIP"] / "train.log").read_text(encoding="utf-8", errors="replace")
    hist["BLIP"] = [{"epoch": int(m.group(1)), "CIDEr": float(m.group(2)), "AnswerHit": float(m.group(3))}
                    for m in re.finditer(r"^epoch (\d+)/\d+ .*?val_CIDEr=([\d.]+).*?val_AnswerHit=([\d.]+)", log, re.M)]
    fig_curves(hist)


if __name__ == "__main__":
    main()
