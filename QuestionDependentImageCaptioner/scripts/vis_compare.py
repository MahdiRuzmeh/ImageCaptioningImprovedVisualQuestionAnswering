"""Visual-feature comparison table: probe JSONs of outputs/vis/<variant>/probe_test_s*.json.

Finglish:
    Har variant (base / v0_frcnn_box / v1_butd / v2_siglip / v3_dinov2) → mean ± std roye seed ha.
    Grounding metric-e asli: ImageDep = metric(normal) − metric(shuffle) (tasvir-e dige, hamoon soal),
    kol + har qtype. AC = AnswerConsistency (yes/no → Polarity, number → Count, baghie → AnswerHit).

    python scripts/vis_compare.py base v0_frcnn_box v2_siglip v3_dinov2
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "vis"
QTYPES = ("yes/no", "number", "color", "other")


def ms(v):
    if not v:
        return "-"
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def main() -> None:
    names = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    runs = {n: [json.loads(f.read_text())["results"] for f in sorted((ROOT / n).glob("probe_test_s*.json"))]
            for n in names}
    rows = [
        ("seeds", lambda r: None),
        ("CIDEr (QD ref)", lambda r: r["normal"]["all"]["CIDEr"]),
        ("AnswerConsistency", lambda r: r["normal"]["all"]["AnswerConsistency"]),
        ("AnswerHit", lambda r: r["normal"]["all"]["AnswerHit"]),
        ("PolarityAcc (yes/no)", lambda r: r["normal"]["all"]["PolarityAcc"]),
        ("CountAcc (number)", lambda r: r["normal"]["all"]["CountAcc"]),
        ("AC color", lambda r: r["normal"]["color"]["AnswerConsistency"]),
        ("AC other", lambda r: r["normal"]["other"]["AnswerConsistency"]),
        ("ImageDep AC (all)", lambda r: r["normal"]["all"]["AnswerConsistency"] - r["shuffle"]["all"]["AnswerConsistency"]),
    ] + [
        (f"ImageDep AC {t}", (lambda t: lambda r: r["normal"][t]["AnswerConsistency"] - r["shuffle"][t]["AnswerConsistency"])(t))
        for t in QTYPES
    ] + [
        ("CIDEr vs COCO (normal)", lambda r: r["coco_reference"]["normal"]["CIDEr"]),
    ]
    print("| Metric (test) | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, fn in rows:
        if label == "seeds":
            print("| seeds | " + " | ".join(str(len(runs[n])) for n in names) + " |")
            continue
        print(f"| {label} | " + " | ".join(ms([fn(r) for r in runs[n]]) for n in names) + " |")


if __name__ == "__main__":
    main()
