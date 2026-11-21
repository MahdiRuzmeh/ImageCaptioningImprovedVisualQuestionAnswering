"""Summary of the BLIP VQA runs: baseline vs + BLIP captions (vs oracle), mean ± std over seeds.

Finglish: hame ``work_dir/runs/*/results.json`` → Markdown: overall + answer type (yes/no, number,
other) + question type (65 VQA v2 type), shuffled-caption test, gate. Group = caption source.

    python summarize_vqa.py --config configs/blip.yaml        # → work_dir/summary.md
    python summarize_vqa.py --runs vqa_baseline vqa_blip --out vqa_summary.md   # two Kaggle notebooks combined
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from common import load_config, read_json, write_json

ORDER = ["none", "blip", "oracle"]
LABEL = {"none": "Baseline (no caption)", "blip": "+ BLIP captions", "oracle": "+ oracle captions (ceiling)"}


def ms(xs):
    if not xs:
        return "—"
    return f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/blip.yaml")
    p.add_argument("--runs", nargs="*", help="Folders holding <run>/results.json (e.g. the Kaggle outputs)")
    p.add_argument("--out", default=None, help="Markdown output (default work_dir/summary.md)")
    args = p.parse_args()
    if args.runs:
        dirs = [Path(d) for d in args.runs]
        out_md = Path(args.out or "vqa_summary.md")
    else:
        work = Path(load_config(args.config)["work_dir"])
        dirs, out_md = [work / "runs"], Path(args.out or work / "summary.md")
    runs = defaultdict(list)
    # Finglish: results.json (ya result.json age download esm ro avaz karde), har jaye folder
    for d in dirs:
        for f in sorted(list(d.rglob("results.json")) + list(d.rglob("result.json"))):
            r = read_json(f)
            runs[r["captions"]].append(r)
    groups = [g for g in ORDER if g in runs]
    if not groups:
        raise SystemExit(f"no results under {dirs}")

    L = ["# VQA: baseline vs + BLIP captions", "",
         f"Train {runs[groups[0]][0]['train_questions']:,} questions (train2014), "
         f"test {runs[groups[0]][0]['test']['n']:,} questions (val2014). Official VQA v2 accuracy (%), "
         "mean ± std over seeds, best dev epoch.", "",
         "## Overall and by answer type", "",
         "| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for g in groups:
        rs = runs[g]
        at = lambda t: [r["test"]["answer_type"][t]["acc"] for r in rs if t in r["test"]["answer_type"]]  # noqa: E731
        sh = [r["test_shuffled_captions"]["acc"] for r in rs if "test_shuffled_captions" in r]
        gate = [r["test"]["gate_mean"] for r in rs if "gate_mean" in r["test"]]
        overall = [r["test"]["acc"] for r in rs]
        summary[g] = {"overall": overall, "yes/no": at("yes/no"), "number": at("number"), "other": at("other")}
        L.append(f"| {LABEL[g]} | {len(rs)} | **{ms(overall)}** | {ms(at('yes/no'))} | {ms(at('number'))} | "
                 f"{ms(at('other'))} | {ms(sh)} | {ms(gate) if gate else '—'} |")
    n_at = runs[groups[0]][0]["test"]["answer_type"]
    L += ["", "Test questions per answer type: " + ", ".join(f"{t} {v['n']:,}" for t, v in n_at.items()), ""]
    if "none" in summary and "blip" in summary:
        L += ["**Gain of BLIP captions over the baseline** (mean): " + ", ".join(
            f"{k} {statistics.mean(summary['blip'][k]) - statistics.mean(summary['none'][k]):+.2f}"
            for k in ("overall", "yes/no", "number", "other")), ""]

    qt = defaultdict(dict)
    for g in groups:
        for t in runs[g][0]["test"]["question_type"]:
            qt[t][g] = statistics.mean(r["test"]["question_type"][t]["acc"] for r in runs[g])
            qt[t]["n"] = runs[g][0]["test"]["question_type"][t]["n"]
    L += ["## By question type (65 VQA v2 types, mean over seeds)", "",
          "| Question type | Test n | " + " | ".join(LABEL[g] for g in groups)
          + (" | BLIP − baseline |" if {"none", "blip"} <= set(groups) else " |"),
          "|---|---:|" + "---:|" * len(groups) + ("---:|" if {"none", "blip"} <= set(groups) else "")]
    for t, v in sorted(qt.items(), key=lambda kv: -kv[1]["n"]):
        row = f"| {t} | {v['n']:,} | " + " | ".join(f"{v.get(g, float('nan')):.1f}" for g in groups)
        if {"none", "blip"} <= set(groups):
            row += f" | {v['blip'] - v['none']:+.1f}"
        L.append(row + " |")
    text = "\n".join(L) + "\n"
    out_md.write_text(text, encoding="utf-8")
    write_json(out_md.with_suffix(".json"), {g: {"runs": [r["name"] for r in runs[g]], **summary[g]} for g in groups})
    print(text)
    print(f"-> {out_md}")


if __name__ == "__main__":
    main()
