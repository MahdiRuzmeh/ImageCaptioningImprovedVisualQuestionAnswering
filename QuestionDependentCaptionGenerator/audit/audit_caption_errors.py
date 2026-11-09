"""Score the caption-error judge (caption_errors.judge) against hand labels.

The gold file holds real captions with ``manual_verdict`` (ok /
GRAMMAR_ISSUE_ONLY / WRONG) and ``manual_reason``. This script judges every
item with the same code as the full run and prints:

* precision / recall of WRONG (the number that matters: WRONG rows get
  regenerated, everything else is kept);
* the 3x3 confusion matrix and reason agreement on rows both call WRONG;
* every disagreement, so the prompt can be tuned.

Usage (from QuestionDependentCaptionGenerator/)::

    python audit/audit_caption_errors.py --model qwen2.5:3b-instruct-q4_K_M
    python audit/audit_caption_errors.py audit/DevAuditor/caption_error_audit_manual.json \
        --model qwen2.5:7b-instruct-q4_K_M --llm-parallel 8 --output scored.json

Target for the full run: WRONG recall >= 0.90 and precision >= 0.80.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

GEN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GEN_DIR))

from caption_errors.judge import (  # noqa: E402
    JUDGE_PROMPT_VERSION, VERDICTS, WRONG, OllamaChat, judge_items,
)

DEFAULT_GOLD = GEN_DIR / "audit" / "DevAuditor" / "caption_error_audit_manual.json"
TARGET_RECALL, TARGET_PRECISION = 0.90, 0.80


def score(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [(it["manual_verdict"], it.get("judge_verdict")) for it in items]
    tp = sum(1 for g, p in pairs if g == WRONG and p == WRONG)
    fp = sum(1 for g, p in pairs if g != WRONG and p == WRONG)
    fn = sum(1 for g, p in pairs if g == WRONG and p != WRONG)
    both = [it for it in items
            if it["manual_verdict"] == WRONG and it.get("judge_verdict") == WRONG]
    return {
        "n": len(items),
        "unjudged": sum(1 for _g, p in pairs if p is None),
        "wrong_precision": tp / (tp + fp) if tp + fp else 0.0,
        "wrong_recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "accuracy_binary": sum(1 for g, p in pairs if (g == WRONG) == (p == WRONG))
                           / max(1, len(pairs)),
        "correct_kept": (sum(1 for g, p in pairs if g != WRONG and p != WRONG)
                         / max(1, sum(1 for g, _p in pairs if g != WRONG))),
        "accuracy_3way":sum(1 for g, p in pairs if g == p) / max(1, len(pairs)),
        "reason_agreement": (sum(1 for it in both if it.get("judge_reason")
                                 == it.get("manual_reason")) / len(both))
                            if both else 0.0,
        "confusion": dict(Counter(f"{g} -> {p}" for g, p in pairs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gold_json", nargs="?", default=str(DEFAULT_GOLD))
    ap.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--llm-parallel", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--output", default=None,
                    help="scored copy (default: <gold stem>_scored.json next to it)")
    args = ap.parse_args()

    gold_path = Path(args.gold_json)
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = doc["items"]
    client = OllamaChat(args.ollama_host, args.model, num_ctx=args.num_ctx,
                        timeout_s=args.timeout)
    batches = [items[i:i + args.batch_size]
               for i in range(0, len(items), args.batch_size)]

    def run(batch: List[Dict[str, Any]]):
        return judge_items(client, [(it["question"], it["answer"], it["caption"])
                                    for it in batch], retries=args.retries)

    print(f"Judging {len(items)} gold items with {args.model} "
          f"(prompt {JUDGE_PROMPT_VERSION}) ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.llm_parallel) as pool:
        for batch, res in zip(batches, pool.map(run, batches)):
            for it, r in zip(batch, res):
                it["judge_verdict"], it["judge_reason"] = r if r else (None, None)

    s = score(items)
    print("\nDisagreements (manual -> judge):")
    for it in items:
        if it["manual_verdict"] != it.get("judge_verdict"):
            print(f"  [{it['manual_verdict']}/{it.get('manual_reason')} -> "
                  f"{it.get('judge_verdict')}/{it.get('judge_reason')}] "
                  f"{it['question']} | {it['answer']} | {it['caption']}")
    print("\nConfusion (manual -> judge):")
    for g in VERDICTS:
        row = {p: s["confusion"].get(f"{g} -> {p}", 0) for p in (*VERDICTS, None)}
        print(f"  {g:>18}: {row}")
    print(f"\nWRONG precision {s['wrong_precision']:.3f} | recall "
          f"{s['wrong_recall']:.3f} (tp {s['tp']}, fp {s['fp']}, fn {s['fn']}) | "
          f"3-way accuracy {s['accuracy_3way']:.3f} | reason agreement "
          f"{s['reason_agreement']:.3f} | unjudged {s['unjudged']}")
    ok = s["wrong_recall"] >= TARGET_RECALL and s["wrong_precision"] >= TARGET_PRECISION
    n_wrong = s["tp"] + s["fn"]
    n_flag = s["tp"] + s["fp"]
    print(f"\n========== JUDGE QUALITY ({args.model}, {s['n']} test captions: "
          f"{n_wrong} wrong, {s['n'] - n_wrong} correct) ==========")
    print(f"  Wrong captions found (recall):          {100 * s['wrong_recall']:5.1f}%  "
          f"({s['tp']} of {n_wrong})")
    print(f"  Flagged WRONG that really are (prec.):  {100 * s['wrong_precision']:5.1f}%  "
          f"({s['tp']} of {n_flag})")
    print(f"  Correct captions kept (not flagged):    {100 * s['correct_kept']:5.1f}%  "
          f"({s['n'] - n_wrong - s['fp']} of {s['n'] - n_wrong})")
    print(f"  Overall WRONG / not-WRONG accuracy:     {100 * s['accuracy_binary']:5.1f}%")
    print(f"  Target recall >= {100 * TARGET_RECALL:.0f}% and precision >= "
          f"{100 * TARGET_PRECISION:.0f}%: {'MET -> run mode full' if ok else 'NOT met -> tune the prompt first'}")
    print("=" * 60, flush=True)

    out = Path(args.output) if args.output else gold_path.with_name(
        gold_path.stem + "_scored.json")
    doc["info"] = dict(doc.get("info") or {}, scored_with={
        "model": args.model, "judge_prompt_version": JUDGE_PROMPT_VERSION, **s})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Scored copy -> {out}", flush=True)


if __name__ == "__main__":
    main()
