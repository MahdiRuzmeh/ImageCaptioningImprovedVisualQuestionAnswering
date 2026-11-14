"""Results table for the captioner comparison: BLEU-4, METEOR, ROUGE-L, CIDEr, AnswerHit (test split).

Finglish: har ``test_metrics.json`` (benchmark.py) = yek satr. Model A, model B (BLIP) va ... hame
hamin format ro minevisan → yek jadval-e Markdown.

    python scripts/results_table.py outputs/a_full_butd/test_metrics.json outputs/a_full_butd_large/test_metrics.json
    python scripts/results_table.py --glob "/kaggle/working/model_a/*/test_metrics.json" --out results.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

COLS = ("BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "AnswerHit")


def fmt(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="*")
    p.add_argument("--glob", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    files = list(args.files) + (sorted(glob.glob(args.glob)) if args.glob else [])
    lines = [
        "| Model | Class | Params | Best epoch | Beam | " + " | ".join(COLS) + " |",
        "|---|---|---:|---:|---:|" + "---:|" * len(COLS),
    ]
    for f in files:
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        o = r["overall"]
        lines.append(
            f"| {r['name']} | {r.get('model_class', '')} | {r.get('trainable_params', 0) / 1e6:.1f}M | "
            f"{r.get('best_epoch')} | {r.get('beam')} | " + " | ".join(fmt(o.get(c)) for c in COLS) + " |"
        )
    n = None
    if files:
        o = json.loads(Path(files[0]).read_text(encoding="utf-8"))["overall"]
        n = (int(o.get("n", 0)), int(o.get("AnswerHit_n", 0)))
    text = "\n".join(lines)
    if n:
        text += f"\n\nTest rows: {n[0]:,}; AnswerHit counts {n[1]:,} non-yes/no rows. METEOR = NLTK meteor_score."
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
