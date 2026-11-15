"""AnswerHit table: does the answer appear in the ground-truth caption, and in the predicted caption?

Finglish: az ``test_metrics.json`` (benchmark.py / train_blip.py --test) har row →
question, answer, gt caption, hit dar gt, pred caption, hit dar pred. Hamoon rule-e
``metrics.answer_in_caption`` (yes/no skip). Khorooji CSV (UTF-8 BOM → Excel) + summary.

    python scripts/answerhit_table.py kaggle_Image_captioner_BLIP_full/test_metrics.json.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import NON_LEXICAL_ANSWERS, answer_in_caption  # noqa: E402
from train import tok  # noqa: E402


def hit(answer: str, caption: str) -> str:
    a = tok(answer)
    if not a or " ".join(a) in NON_LEXICAL_ANSWERS:
        return "skip (yes/no)"
    return "yes" if answer_in_caption(a, tok(caption)) else "no"


def pct(n: int, d: int) -> str:
    return f"{n:,}/{d:,} = {100 * n / d:.1f}%" if d else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("metrics_json")
    p.add_argument("--out", default=None, help="CSV path (default: answerhit_table.csv next to the JSON)")
    args = p.parse_args()
    src = Path(args.metrics_json)
    r = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for x in r["predictions"]:
        rows.append({
            "question": x["question"], "answer": x["answer"], "type": x.get("rule") or "",
            "ground_truth_caption": x["gt"], "answer_hit_in_gt": hit(x["answer"], x["gt"]),
            "pred_caption": x["pred"], "answer_hit_in_pred": hit(x["answer"], x["pred"]),
        })
    out = Path(args.out) if args.out else src.parent / "answerhit_table.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lex = [x for x in rows if x["answer_hit_in_gt"] != "skip (yes/no)"]
    n = len(lex)
    gt = sum(x["answer_hit_in_gt"] == "yes" for x in lex)
    pr = sum(x["answer_hit_in_pred"] == "yes" for x in lex)
    both = Counter((x["answer_hit_in_gt"], x["answer_hit_in_pred"]) for x in lex)
    lines = [
        f"# AnswerHit: ground truth vs prediction ({r.get('name')}, {r.get('split')} split, beam {r.get('beam')})",
        "",
        f"Rows: {len(rows):,} ({len(rows) - n:,} yes/no skipped, {n:,} counted)",
        "",
        "| | Answer in caption |",
        "|---|---:|",
        f"| Ground-truth caption | {pct(gt, n)} |",
        f"| Predicted caption | {pct(pr, n)} |",
        "",
        "| GT hit | Pred hit | Rows |",
        "|---|---|---:|",
    ] + [f"| {g} | {q} | {both[(g, q)]:,} |" for g in ("yes", "no") for q in ("yes", "no")]
    lines += ["", "| Type | Rows | GT hit | Pred hit |", "|---|---:|---:|---:|"]
    by = defaultdict(list)
    for x in lex:
        by[x["type"]].append(x)
    for t, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        g = sum(x["answer_hit_in_gt"] == "yes" for x in xs)
        q = sum(x["answer_hit_in_pred"] == "yes" for x in xs)
        lines.append(f"| {t} | {len(xs):,} | {100 * g / len(xs):.1f}% | {100 * q / len(xs):.1f}% |")
    text = "\n".join(lines)
    (out.with_suffix(".md")).write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nCSV -> {out}\nsummary -> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
