"""Score GoldAuditor classifier labels with the production question classifier.

Uses the same Fast Path / default visual / LLM confirm gate as ``generate.py``
(``filter_non_visual_questions`` / ``QuestionClassifier``).

Usage (from QuestionDependentCaptionGenerator/):

    python audit/classify_questions.py
    python audit/classify_questions.py --batch-size 10
    python audit/classify_questions.py audit/GoldAuditor/classifier_audit_manual.json --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from question_classifier import (  # noqa: E402
    CLASSIFIER_PROMPT_VERSION,
    QuestionClassifier,
    VISUAL_FILTER_DEFAULT,
    VISUAL_FILTER_FAST_PATH,
    VISUAL_FILTER_LLM,
    is_fast_path_visual,
    is_non_visual_candidate,
)

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_GOLD = AUDIT_DIR / "GoldAuditor" / "classifier_audit_manual.json"


def load_gold(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load gold classifier JSON (``info`` + ``records``)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    info = data.get("info")
    records = data.get("records")
    if not isinstance(info, dict):
        raise ValueError("info must be an object")
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    return dict(info), list(records)


def _gate_record(row: Dict[str, Any], *, fast_path: bool) -> Dict[str, Any]:
    """Apply fast/default gate; leave LLM rows pending."""
    out = dict(row)
    question = str(row.get("question") or "").strip()
    out.pop("classifier_label", None)
    out.pop("visual_filter_source", None)
    out.pop("detail", None)
    out.pop("non_visual_reason", None)

    if fast_path and is_fast_path_visual(question):
        out["classifier_label"] = "DIRECTLY_VISUAL"
        out["visual_filter_source"] = VISUAL_FILTER_FAST_PATH
        out["detail"] = "fast_path"
        return out

    if not is_non_visual_candidate(question):
        out["classifier_label"] = "DIRECTLY_VISUAL"
        out["visual_filter_source"] = VISUAL_FILTER_DEFAULT
        out["detail"] = "default_visual"
        return out

    out["classifier_label"] = None
    out["visual_filter_source"] = VISUAL_FILTER_LLM
    out["detail"] = "pending"
    return out


def classify_gold_records(
    records: Sequence[Dict[str, Any]],
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fast_path: bool = True,
) -> List[Dict[str, Any]]:
    """Score gold records with the production classifier gate."""
    classifier = QuestionClassifier(host=host, model=model)
    batch_n = max(1, int(batch_size))
    scored: List[Dict[str, Any]] = [
        _gate_record(row, fast_path=fast_path) for row in records
    ]
    llm_indices = [
        i
        for i, rec in enumerate(scored)
        if rec.get("classifier_label") is None
        and rec.get("visual_filter_source") == VISUAL_FILTER_LLM
    ]
    print(
        f"  gates done: {len(scored)} rows, "
        f"{len(llm_indices)} llm_confirm (batch_size={batch_n})",
        flush=True,
    )

    for start in range(0, len(llm_indices), batch_n):
        chunk_idxs = llm_indices[start : start + batch_n]
        questions = [str(scored[i].get("question") or "") for i in chunk_idxs]
        end = start + len(chunk_idxs)
        print(
            f"  llm_confirm batch: {start + 1}-{end}/{len(llm_indices)} "
            f"(size={len(chunk_idxs)})",
            flush=True,
        )
        results, detail = classifier.classify_batch(questions)
        if results is None:
            for i, q in zip(chunk_idxs, questions):
                label, one_detail, reason = classifier.classify_one(q)
                if label is None:
                    scored[i]["classifier_label"] = "NOT_DIRECTLY_VISUAL"
                    scored[i]["detail"] = one_detail or detail or "parse_fail"
                else:
                    scored[i]["classifier_label"] = label
                    scored[i]["detail"] = one_detail or detail
                    if reason:
                        scored[i]["non_visual_reason"] = reason
        else:
            for i, (label, reason) in zip(chunk_idxs, results):
                if label is None:
                    scored[i]["classifier_label"] = "NOT_DIRECTLY_VISUAL"
                    scored[i]["detail"] = detail or "parse_fail"
                else:
                    scored[i]["classifier_label"] = label
                    scored[i]["detail"] = detail
                    if reason:
                        scored[i]["non_visual_reason"] = reason
                    else:
                        scored[i].pop("non_visual_reason", None)

    return scored


def print_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Print accuracy / confusion vs ``manual_label``."""
    tp = fp = tn = fn = 0
    skipped = 0
    pred_counts: Counter = Counter()
    source_counts: Counter = Counter()

    for row in records:
        gold = str(row.get("manual_label") or "").strip().upper()
        pred = str(row.get("classifier_label") or "").strip().upper()
        pred_counts[pred or "None"] += 1
        source_counts[str(row.get("visual_filter_source") or "None")] += 1

        if gold not in {"DIRECTLY_VISUAL", "NOT_DIRECTLY_VISUAL"}:
            skipped += 1
            continue
        if pred not in {"DIRECTLY_VISUAL", "NOT_DIRECTLY_VISUAL"}:
            skipped += 1
            continue

        # Treat DIRECTLY_VISUAL as the positive class for confusion reporting.
        pred_pos = pred == "DIRECTLY_VISUAL"
        gold_pos = gold == "DIRECTLY_VISUAL"
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and not gold_pos:
            tn += 1
        else:
            fn += 1

    labeled = tp + fp + tn + fn
    accuracy = (tp + tn) / labeled if labeled else 0.0
    summary = {
        "tp_directly_visual": tp,
        "fp_directly_visual": fp,
        "tn_not_directly_visual": tn,
        "fn_missed_directly_visual": fn,
        "skipped": skipped,
        "accuracy": round(accuracy, 4),
        "classifier_label_counts": dict(pred_counts),
        "visual_filter_source_counts": dict(source_counts),
    }
    print(
        f"Metrics vs manual_label: accuracy={accuracy:.4f} "
        f"(tp={tp} fp={fp} tn={tn} fn={fn}; skipped={skipped})",
        flush=True,
    )
    print(f"  classifier_label_counts={dict(pred_counts)}", flush=True)
    print(f"  visual_filter_source_counts={dict(source_counts)}", flush=True)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score GoldAuditor classifier_audit_manual.json with the "
            "production question classifier."
        ),
    )
    parser.add_argument(
        "gold_json",
        nargs="?",
        type=Path,
        default=DEFAULT_GOLD,
        help=f"Gold classifier JSON (default: {DEFAULT_GOLD})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write scored JSON here (default: <stem>_scored.json next to input)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input gold JSON instead of writing a sibling scored file",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama host (default {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"llm_confirm items per Ollama call (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-fast-path",
        action="store_true",
        help="Disable Fast Path exemption (same as generate.py --no-fast-path)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    gold_path = args.gold_json.resolve()
    if not gold_path.is_file():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 1

    info, records = load_gold(gold_path)
    fast_path = not args.no_fast_path
    print(
        f"Gold classifier audit: {gold_path.name} n={len(records)} "
        f"model={args.model} batch_size={args.batch_size} "
        f"fast_path={fast_path} prompt={CLASSIFIER_PROMPT_VERSION}",
        flush=True,
    )
    scored = classify_gold_records(
        records,
        host=args.host,
        model=args.model,
        batch_size=args.batch_size,
        fast_path=fast_path,
    )
    metrics = print_metrics(scored)

    out_info = dict(info)
    out_info["classifier_scoring"] = {
        "prompt_version": CLASSIFIER_PROMPT_VERSION,
        "model": args.model,
        "host": args.host,
        "batch_size": args.batch_size,
        "fast_path_enabled": fast_path,
        **metrics,
    }

    if args.in_place:
        out_path = gold_path
    elif args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = gold_path.with_name(f"{gold_path.stem}_scored.json")

    payload = {"info": out_info, "records": scored}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(scored)} scored records -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
