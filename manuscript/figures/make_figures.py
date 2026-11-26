"""Manuscript figures as vector graphics (SVG + PDF) plus a 300-dpi PNG for the Word build.

All numbers come from committed files: manuscript/evidence/paper_evidence.json, CaptionVQA/large_analysis.json,
CaptionVQA run results, the EvalAI result JSONs, and the captioner comparison (QuestionDependentImageCaptioner).

    python manuscript/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VQA = REPO / "CaptionVQA"
INK, INK2, GRID, SURF = "#111111", "#4d4d4d", "#d9d9d6", "#ffffff"
BLUE, ORANGE, GREEN, GOLD, GREY, PURPLE = "#2a6fc9", "#e0662b", "#1a9e6e", "#d99a00", "#9a9890", "#7a4fb3"
LIGHT = {BLUE: "#e6eef9", ORANGE: "#fcebe2", GREEN: "#e2f3ec", GOLD: "#fbf1d6", GREY: "#f0f0ee", PURPLE: "#efe8f7"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def evalai(p):
    d = {}
    for x in load(p):
        d.update(x)
    return d


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("->", HERE / name)


# ------------------------------------------------------------------ drawing helpers
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fs=7.6, bold=False, dashed=False, fill=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.2", linewidth=1.1,
                                edgecolor=color, facecolor=LIGHT.get(color, SURF) if fill else SURF,
                                linestyle=(0, (3, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else None, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, color=INK2, dashed=False, text=None, fs=6.6, rad=0.0, toff=(0, 1.2)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha="center", va="bottom",
                fontsize=fs, color=color)


def R(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def Bo(b):
    return (b[0] + b[2] / 2, b[1])


def style(ax, lo, hi, axis="y", label="VQA accuracy (%)"):
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    (ax.set_ylabel if axis == "y" else ax.set_xlabel)(label)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ------------------------------------------------------------------ Fig. 1 data flow
def fig_data_flow(ev):
    f = ev["data_flow"]
    s = f["screening_and_repair"]
    fp = f["first_pass"]
    fig, ax = canvas(7.2, 3.3)
    y = 21
    b1 = box(ax, 1, y, 11, 8, f"VQA v2 train2014\nquestions\n{f['train2014_questions']:,}", GREY)
    b2 = box(ax, 15, y, 13, 8, f"Classification,\ngeneration, fast check\n(Qwen2.5-3B)\n{fp['caption_records']:,}", BLUE)
    b3 = box(ax, 31, y, 12, 8, f"Re-classification\n(parse-failure fix,\nQwen2.5-3B)\n{f['reclassification']['rows_after']:,}", BLUE)
    b4 = box(ax, 46, y, 12, 8, f"Semantic screening\n(Qwen2.5-7B)\nok {s['initial']['ok']:,}\nWRONG {s['initial']['WRONG']:,}", ORANGE)
    b5 = box(ax, 61, y, 10.5, 8, f"Regenerate\nWRONG (≤ 2\nrounds, 7B)\nfixed {s['fixed']:,}", ORANGE, fs=7.0)
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        arrow(ax, R(a), L(b))
    ax.text(21.5, 18.5, f"excluded: not visual {fp['blacklist_not_visual']:,}, OCR {fp['ocr_excluded']:,},\n"
            f"duplicates/empty {fp['duplicates'] + fp['dropped_empty']}, failed checks {fp['validation_failures']}",
            ha="center", va="top", fontsize=6.2, color=INK2)
    lab = f["reclassification"]["labels"]
    ax.text(37, 18.5, f"not visual {lab['NOT_DIRECTLY_VISUAL']:,}\n(opinion {lab['NEEDS_OPINION']:,},\nknowledge "
            f"{lab['NEEDS_KNOWLEDGE']:,},\nOCR {lab['NEEDS_OCR']:,})", ha="center", va="top", fontsize=6.2, color=INK2)
    fin = s["final"]
    c1 = box(ax, 18, 2, 21, 7.5, f"ok: {fin['ok']:,} rows\n{f['ok_images']:,} images\n→ captioner training",
             GREEN, bold=True)
    c2 = box(ax, 42, 2, 13, 7.5, f"GRAMMAR_ISSUE_ONLY\n{fin['GRAMMAR_ISSUE_ONLY']:,}\nnot used", GREY)
    c3 = box(ax, 58, 2, 13, 7.5, f"still WRONG\n{fin['WRONG']:,} ({f['wrong_share_of_screened_pct']:.2f}%)\nnot used", GREY)
    arrow(ax, Bo(b5), T(c3), rad=0)
    arrow(ax, (52, y), T(c2))
    arrow(ax, (49, y), (35, 9.5), rad=-0.1)
    arrow(ax, (62, y), (37, 9.5), rad=-0.15)
    ax.text(1, 6, f"{f['ok_rows_with_suspicious_flag']:,} ok rows carry a\nheuristic 'suspicious' flag\n"
            "(extra words / polarity);\nkept after the 7B judge", fontsize=6.2, color=INK2, va="center")
    save(fig, "fig1_data_flow")


# ------------------------------------------------------------------ Fig. 2 cross-fitting
def fig_crossfit(ev):
    x = ev["crossfit"]
    f0, f1 = x["folds"]["0"], x["folds"]["1"]
    tr = x["train2014_images_per_fold"]
    fig, ax = canvas(7.2, 3.4)
    a0 = box(ax, 1, 22, 15, 8, f"train2014 fold F0\n{tr['0']:,} images\n{f0['rows']:,} QD rows", BLUE)
    a1 = box(ax, 1, 6, 15, 8, f"train2014 fold F1\n{tr['1']:,} images\n{f1['rows']:,} QD rows", PURPLE)
    g0 = box(ax, 26, 22, 14, 8, f"captioner g0\n(BLIP-large)\nselection: {f0['selection_rows']:,} F0 rows", BLUE, bold=False)
    g1 = box(ax, 26, 6, 14, 8, f"captioner g1\n(BLIP-large)\nselection: {f1['selection_rows']:,} F1 rows", PURPLE)
    arrow(ax, R(a0), L(g0), BLUE, text="fine-tune")
    arrow(ax, R(a1), L(g1), PURPLE, text="fine-tune")
    c1 = box(ax, 52, 26, 19, 6, "captions for F1 train images", BLUE, fs=7.2)
    c0 = box(ax, 52, 18, 19, 6, "captions for F0 train images", PURPLE, fs=7.2)
    v = box(ax, 52, 9.5, 19, 6, "val2014: fold value (≈ half each)", GREY, fs=7.2)
    t = box(ax, 52, 1.5, 19, 6, "test2015: image_id mod 2", GREY, fs=7.2)
    arrow(ax, R(g0), L(c1), BLUE)
    arrow(ax, R(g1), L(c0), PURPLE, rad=0.15)
    arrow(ax, R(g0), L(v), BLUE, dashed=True, rad=0.1)
    arrow(ax, R(g1), L(v), PURPLE, dashed=True)
    arrow(ax, R(g0), L(t), BLUE, dashed=True, rad=0.15)
    arrow(ax, R(g1), L(t), PURPLE, dashed=True)
    ax.text(36, 0.6, f"Checked: {x['vqa_questions_checked']:,} train+val questions, "
            f"{x['questions_captioned_by_a_model_trained_on_that_image']} captioned by a model trained on their image; "
            f"test2015 images in the QD corpus: {x['test2015_images_in_qd']}. Dashed = inference only, no answers.",
            ha="center", fontsize=6.0, color=INK2)
    save(fig, "fig2_crossfit")


# ------------------------------------------------------------------ Fig. 3 overall inference
def fig_overview():
    fig, ax = canvas(7.2, 2.5)
    img = box(ax, 1, 14, 10, 7, "image I", GREY)
    q = box(ax, 1, 3, 10, 7, "question q", GREY)
    cap = box(ax, 18, 14, 15, 7, "QD captioner\n(fold BLIP-large, beam 3)", BLUE)
    c = box(ax, 39, 14, 12, 7, "caption ĉ\n(text)", BLUE, fs=7.2)
    det = box(ax, 18, 6.2, 15, 5.5, "BUTD detector\n37 × 2048 regions", GREY)
    vqa = box(ax, 56, 6, 10, 12, "MCAN\nVQA model", GREEN, bold=True)
    out = box(ax, 69, 9, 3, 6, "â", GREEN)
    arrow(ax, R(img), L(cap))
    arrow(ax, (11, 8), (18, 16), rad=-0.1)
    arrow(ax, (6, 14), (18, 10), rad=0.0)
    arrow(ax, R(cap), L(c), BLUE)
    arrow(ax, R(c), (56, 15), BLUE)
    arrow(ax, R(det), (56, 10), INK2)
    arrow(ax, (11, 4), (56, 7), INK2)
    arrow(ax, R(vqa), L(out), GREEN)
    ax.text(36, 23.5, "No reference answer is used at inference; the caption is extra text evidence.", ha="center",
            fontsize=6.6, color=INK2)
    save(fig, "fig3_overview")


# ------------------------------------------------------------------ Fig. 4 captioner
def fig_captioner():
    fig, ax = canvas(7.2, 2.7)
    img = box(ax, 1, 15, 11, 7, "image I\n384 × 384", GREY)
    vit = box(ax, 16, 15, 15, 7, "ViT-L/16 image encoder\n(frozen)", GREY)
    pr = box(ax, 1, 4, 11, 7, "prompt\n\"question: q caption:\"", GREY, fs=6.8)
    dec = box(ax, 36, 8, 17, 11, "BERT text decoder\n(trained)\ncross-attention to image", BLUE, bold=False)
    out = box(ax, 59, 15, 13, 7, "beam 3 → caption ĉ\n(length-normalised\nlog-probability)", BLUE, fs=6.6)
    tgt = box(ax, 59, 3, 13, 7, "target t (train only)\nlabel-smoothed CE\non caption tokens", ORANGE, fs=6.6, dashed=True)
    arrow(ax, R(img), L(vit))
    arrow(ax, R(vit), (36, 16.5), INK2, text="image tokens", toff=(3, 1.5))
    arrow(ax, R(pr), (36, 10.5), INK2, text="prompt tokens (loss masked)", toff=(0, -3.0))
    arrow(ax, (53, 16.5), L(out), BLUE)
    arrow(ax, L(tgt), (53, 10), ORANGE, dashed=True, text="teacher forcing", toff=(0, -4.5))
    ax.text(36, 25.0, "AdamW, lr 3e-5, weight decay 0.05, 1,000 warm-up steps + cosine, 4 epochs, batch 48 per GPU, ε = 0.1\n"
            "checkpoint = most right captions (AnswerHit + yes/no polarity) on own-fold selection rows", ha="center", fontsize=6.2, color=INK2)
    save(fig, "fig4_captioner")


# ------------------------------------------------------------------ Fig. 5 MCAN fusion
def fig_mcan():
    fig, ax = canvas(7.2, 3.3)
    q = box(ax, 1, 24, 13, 6, "question words\nGloVe 840B + LSTM", GREY, fs=6.8)
    enc = box(ax, 18, 24, 17, 6, "encoder: 6 × self-attention\n(question Q)", GREEN, fs=6.8)
    r = box(ax, 1, 12, 13, 6, "37 regions × 2048\nlinear → d = 512", GREY, fs=6.8)
    c = box(ax, 1, 2, 13, 6, "caption words\nGloVe 840B + own LSTM", BLUE, fs=6.8)
    drop = box(ax, 17, 2, 13, 6, "caption dropout\nwhole caption → PAD, p = 0.3\n(training only)", ORANGE, fs=6.2, dashed=True)
    x = box(ax, 35, 8, 8, 9, "X = [V ; C]\n+ type\nembedding", BLUE, fs=6.6)
    dec = box(ax, 47, 8, 13, 12, "decoder: 6 ×\nself-attn over X\n→ guided attn to Q\n→ FFN", GREEN, fs=6.6)
    pool = box(ax, 63, 13, 9, 9, "AttFlat(Q)\n+ AttFlat(X)\n→ LN →\n3,148 sigmoid", GREEN, fs=6.0)
    arrow(ax, R(q), L(enc))
    arrow(ax, R(r), (35, 14), INK2)
    arrow(ax, R(c), L(drop), BLUE)
    arrow(ax, R(drop), (35, 10), BLUE)
    arrow(ax, R(x), (47, 14), INK2)
    arrow(ax, Bo(enc), (53.5, 20), GREEN, text="keys / values", toff=(6, 0))
    arrow(ax, R(dec), (63, 16), GREEN)
    arrow(ax, R(enc), (63, 21), GREEN, rad=-0.2)
    ax.text(36, 31.5, "MCAN-small: d = 512, 8 heads, FFN 2048, dropout 0.1; question ≤ 14 tokens, caption ≤ 24 tokens",
            ha="center", fontsize=6.4, color=INK2)
    save(fig, "fig5_mcan_fusion")


# ------------------------------------------------------------------ Fig. 6 official results
def fig_official(ev):
    cap = evalai(VQA / "vqa_final_t3/vqa_final/Ensemble-VQA-With-Large-Blip-Test-Standard-Result-EvalAI.json")
    base = evalai(VQA / "vqa_final_t4/vqa_final_baseline/Ensemble-VQA-BaseLine-Test-Standard-Result-EvalAI.json")
    paper = {"overall": 70.67, "yes/no": 86.63, "number": 48.23, "other": 59.98}
    keys, labs = ["overall", "yes/no", "number", "other"], ["Overall", "Yes/no", "Number", "Other"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    series = [("Sharma & Jalal [4], Baseline + Att + Cap", paper, GREY), ("MCAN, no captions (ours)", base["test-standard"], BLUE),
              ("MCAN + QD captions (ours)", cap["test-standard"], ORANGE)]
    w = 0.26
    for j, (nm, d, col) in enumerate(series):
        for k, key in enumerate(keys):
            xx = k + (j - 1) * w
            a.bar(xx, d[key], w * 0.92, color=col, label=nm if k == 0 else None)
            a.text(xx, d[key] + 0.6, f"{d[key]:.1f}", ha="center", va="bottom", fontsize=5.8, color=INK)
    a.set_xticks(range(4), labs)
    style(a, 40, 95)
    a.set_title("(a) test-standard, 2-seed ensembles", loc="left")
    a.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    o = ev["official"]
    rows = [("test-dev, single seeds (mean)", o["gain_single_mean"]["overall"]),
            ("test-dev, ensembles", o["gain_ensemble_test_dev"]["overall"]),
            ("test-standard, ensembles", round(cap["test-standard"]["overall"] - base["test-standard"]["overall"], 2))]
    for i, (nm, v) in enumerate(rows):
        b.barh(i, v, 0.55, color=ORANGE if i == 2 else "#f0a57f")
        b.text(v + 0.05, i, f"+{v:.2f}", va="center", fontsize=7, color=INK)
    b.set_yticks(range(3), [r[0] for r in rows], fontsize=6.8)
    b.invert_yaxis()
    style(b, 0, 4.6, axis="x", label="caption gain (pp)")
    b.set_title("(b) gain over the matched baseline", loc="left")
    fig.tight_layout()
    save(fig, "fig6_official")


# ------------------------------------------------------------------ Fig. 7 transitions
def fig_transitions():
    t = load(VQA / "large_analysis.json")["transitions"]
    labs = ["wrong → right", "right → wrong", "right → right", "wrong → wrong"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.26
    for j, (k, nm, col) in enumerate((("base", "no captions", GREY), ("blip", "BLIP-base captions", BLUE),
                                      ("large", "BLIP-large captions", ORANGE))):
        for i, lab in enumerate(labs):
            v = t[lab][k]
            ax.bar(i + (j - 1) * w, v, w * 0.92, color=col, label=nm if i == 0 else None)
            ax.text(i + (j - 1) * w, v + 0.8, f"{v:.1f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(4), [f"{lab}\n(n = {t[lab]['n']:,})" for lab in labs])
    style(ax, 0, 100)
    ax.set_xlabel("caption correctness, BLIP-base → BLIP-large (10,000 labelled diagnostic questions)")
    ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper left")
    save(fig, "fig7_transitions")


# ------------------------------------------------------------------ Fig. 8 perturbations
def fig_perturb():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    cond = [("normal", 58.9), ("no\nimage", 15.2), ("other\nimage", 12.4), ("no\nquestion", 11.0),
            ("other question\n(new answer)", 58.9), ("answer\nprior", 14.1)]
    a.bar(range(len(cond)), [c[1] for c in cond], 0.6, color=[BLUE, GREY, GREY, GREY, BLUE, "#cfcdc6"])
    for i, (_, v) in enumerate(cond):
        a.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=6.4)
    a.set_xticks(range(len(cond)), [c[0] for c in cond], fontsize=6.2)
    style(a, 0, 70, label="caption AnswerHit (%)")
    a.set_title("(a) BLIP-base captioner, input perturbations", loc="left")
    res = {}
    for k, p in (("no dropout\nBLIP-base", "vqa_b64_crossfit"), ("dropout 0.3\nBLIP-base", "vqa_b64_crossfit_drop"),
                 ("dropout 0.3\nBLIP-large", "vqa_large_drop")):
        r = [load(VQA / f"{p}/runs/blip_s{s}/results.json") for s in (0, 1)]
        res[k] = (sum(x["test"]["acc"] for x in r) / 2, sum(x["test_shuffled_captions"]["acc"] for x in r) / 2)
    w = 0.36
    for i, (k, (n, s)) in enumerate(res.items()):
        b.bar(i - w / 2, n, w * 0.92, color=ORANGE, label="own captions" if i == 0 else None)
        b.bar(i + w / 2, s, w * 0.92, color="#f0a57f", label="shuffled captions" if i == 0 else None)
        b.text(i - w / 2, n + 0.3, f"{n:.1f}", ha="center", fontsize=6.2)
        b.text(i + w / 2, s + 0.3, f"{s:.1f}", ha="center", fontsize=6.2)
    b.axhline(66.39, color=BLUE, linewidth=1, linestyle=(0, (3, 2)), label="no captions (66.4)")
    b.set_xticks(range(3), list(res), fontsize=6.4)
    style(b, 55, 75)
    b.set_title("(b) VQA, 10k diagnostic set (seed means)", loc="left")
    b.legend(frameon=False, fontsize=6.2, loc="upper left")
    fig.tight_layout()
    save(fig, "fig8_perturbations")


def main():
    ev = load(REPO / "manuscript/evidence/paper_evidence.json")
    fig_data_flow(ev)
    fig_crossfit(ev)
    fig_overview()
    fig_captioner()
    fig_mcan()
    fig_official(ev)
    fig_transitions()
    fig_perturb()


if __name__ == "__main__":
    main()
