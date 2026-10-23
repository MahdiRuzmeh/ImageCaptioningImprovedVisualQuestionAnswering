"""Hame ``outputs/bench/<arch>/seed*/test_metrics.json`` ro jam kon → markdown table.

Finglish:
    Har architecture: mean ± std roye seed ha (BLEU-4, CIDEr, ROUGE-L, AnswerHit),
    + per question type, params, sorat decode. Age run_config ha dar key haye
    gheyr-model ham farghe dashte bashan warning mide (protocol yeksan nist).

    python scripts/bench_compare.py --root outputs/bench --out outputs/bench/summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

METRICS = ["BLEU-1", "BLEU-4", "CIDEr", "ROUGE-L", "AnswerHit"]
# Finglish: in key ha mitoonan beyn architecture ha fargh konan
MODEL_KEYS = {
    "model_class", "seed", "save_dir", "optimizer", "learning_rate",
    "warmup_steps", "weight_decay", "cap_vocab_include_question", "dropout",
    "word_dim", "hidden_dim", "embed_dim", "num_layers", "num_heads",
    "ff_dim", "fuse_mode", "lr_decay_factor", "lr_decay_interval",
    "scheduled_sampling_max_prob", "_cli", "_config_path",
}


def fmt(vals: List[float], scale: float = 1.0) -> str:
    vals = [v * scale for v in vals]
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/bench")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    configs: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(args.root).glob("**/test_metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        runs[r["name"]].append(r)
        rc = f.parent / "run_config.yaml"
        if rc.exists():
            configs[str(f.parent)] = yaml.safe_load(rc.read_text(encoding="utf-8"))

    lines = ["## Test results (mean ± std over seeds)", ""]
    lines.append("| Model | seeds | params | " + " | ".join(METRICS) + " | items/s |")
    lines.append("|---" * (len(METRICS) + 4) + "|")
    for name, rs in runs.items():
        cells = [fmt([r["overall"][m] for r in rs]) for m in METRICS]
        lines.append(
            f"| {name} | {len(rs)} | {rs[0]['trainable_params'] / 1e6:.1f}M | "
            + " | ".join(cells)
            + f" | {statistics.mean(r['items_per_sec'] for r in rs):.0f} |"
        )

    rules = sorted({k for rs in runs.values() for r in rs for k in r["per_rule"]})
    for metric in ("CIDEr", "AnswerHit"):
        lines += ["", f"## {metric} per question type", ""]
        lines.append("| Model | " + " | ".join(rules) + " |")
        lines.append("|---" * (len(rules) + 1) + "|")
        for name, rs in runs.items():
            cells = []
            for rule in rules:
                vals = [r["per_rule"][rule][metric] for r in rs if rule in r["per_rule"]]
                cells.append(fmt(vals) if vals else "-")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        n_row = [str(int(next(iter(runs.values()))[0]["per_rule"].get(r, {}).get("n", 0))) for r in rules]
        lines.append("| *n* | " + " | ".join(n_row) + " |")

    # Finglish: protocol check — key haye gheyr-model bayad yeksan bashan
    protocol: Dict[str, set] = defaultdict(set)
    for cfg in configs.values():
        for k, v in cfg.items():
            if k not in MODEL_KEYS:
                protocol[k].add(json.dumps(v, sort_keys=True))
    diff = {k: v for k, v in protocol.items() if len(v) > 1}
    if diff:
        lines += ["", "**Warning — protocol keys differ between runs:**", ""]
        lines += [f"- `{k}`: {sorted(v)}" for k, v in diff.items()]

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
