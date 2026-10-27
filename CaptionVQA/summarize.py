"""Collect runs/*/results.json into markdown tables (mean ± std over seeds).

    python summarize.py --config configs/small.yaml
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from common import load_config, read_json

COLS = [("acc", "Overall"), ("acc_yes/no", "Yes/No"), ("acc_number", "Number"), ("acc_other", "Other")]


def fmt(vals):
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small.yaml")
    args = p.parse_args()
    work = Path(load_config(args.config)["work_dir"])
    groups = defaultdict(list)
    for f in sorted((work / "runs").glob("*/results.json")):
        r = read_json(f)
        src = r.get("caption_source", "generated") if r["use_caption"] else "no"
        groups[(r["model"], src)].append(r)

    print("| Model | Caption | Seeds | " + " | ".join(c for _, c in COLS) + " | Shuffled caption | Δ shuffle | Dev | Params (M) | s/epoch |")
    print("|---|---|---|" + "---|" * (len(COLS) + 5))
    order = {"generated": 0, "human": 1, "no": 2}
    for (m, cap), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        test = [fmt([r["test"][k] for r in rs if k in r["test"]]) for k, _ in COLS]
        sh = [r["test_shuffled_caption"]["acc"] for r in rs if "test_shuffled_caption" in r]
        delta = [r["test"]["acc"] - r["test_shuffled_caption"]["acc"] for r in rs if "test_shuffled_caption" in r]
        print(f"| {m} | {cap} | {len(rs)} | " + " | ".join(test)
              + f" | {fmt(sh)} | {fmt(delta)} | {fmt([r['dev']['acc'] for r in rs])}"
              + f" | {rs[0]['params_M']:.1f} | {statistics.mean(r['secs_per_epoch'] for r in rs):.0f} |")
    sv = work / "runs" / "SimpleVQA" / "results_test.json"
    if sv.exists():
        r = read_json(sv)
        print(f"| SimpleVQA | yes | 1 | " + " | ".join(f"{r[k]:.2f}" for k, _ in COLS) + " | – | – | – | – | – |")

    print("\nExtra diagnostics (test, caption models):")
    for (m, cap), rs in sorted(groups.items()):
        if cap == "no":
            continue
        for r in rs:
            t = r["test"]
            extra = []
            if "gate_mean" in t:
                extra.append(f"gate {t['gate_mean']:.3f}")
            if "pointer_rate" in t:
                extra.append(f"pointer {t['pointer_rate']:.3f} (shuffled {r['test_shuffled_caption'].get('pointer_rate', 0):.3f})")
            print(f"- {r['name']}: best epoch {r['best_epoch']}, " + ", ".join(extra))


if __name__ == "__main__":
    main()
