"""Capacity control ("null caption"): the same fixed caption for every question, in the format the VQA runner reads.

The caption branch (embedding, LSTM, type vector, decoder tokens) stays active, but the caption carries no information
about the image or the question. An all-padding caption would be masked out and would test nothing.
Ablation only: the main pipeline does not use this script.

    python scripts/fixed_captions.py --questions <train q.json> <val q.json> --out captions_crossfit_fixed.json
    python scripts/fixed_captions.py --questions <test2015 q.json> --out captions_test2015_fixed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIXED = "an image."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--questions", nargs="+", required=True, help="VQA v2 question JSON files")
    p.add_argument("--out", required=True)
    p.add_argument("--text", default=FIXED)
    args = p.parse_args()
    caps = {}
    for f in args.questions:
        for q in json.loads(Path(f).read_text(encoding="utf-8"))["questions"]:
            caps[str(q["question_id"])] = args.text
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(caps), encoding="utf-8")
    print(f"{len(caps)} questions -> {args.out} (caption: {args.text!r})")


if __name__ == "__main__":
    main()
