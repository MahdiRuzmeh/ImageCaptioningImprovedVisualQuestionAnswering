"""Score DevAuditor classifier labels with the production question classifier.

Uses the same blacklist→NDV | UNKNOWN→parallel LLM cascade as ``generate.py``
(``filter_non_visual_questions`` / ``QuestionClassifier``).

Usage (from QuestionDependentCaptionGenerator/):

    python audit/classify_questions.py
    python audit/classify_questions.py --batch-size 10 --llm-parallel 4
    python audit/classify_questions.py audit/DevAuditor/classifier_audit_manual.json --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from question_classifier import (  # noqa: E402
    QuestionClassifier,
    VISUAL_FILTER_BLACKLIST,
    VISUAL_FILTER_LLM,
    is_non_visual_candidate,
)

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
DEFAULT_LLM_PARALLEL = 4
AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_GOLD = AUDIT_DIR / "DevAuditor" / "classifier_audit_manual.json"

_VALID_LABELS = frozenset({"DIRECTLY_VISUAL", "NOT_DIRECTLY_VISUAL"})


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


def _gate_record(row: Dict[str, Any], *, blacklist_drop: bool) -> Dict[str, Any]:
    """Apply fast blacklist NDV gate; leave UNKNOWN rows pending for LLM."""
    out = dict(row)
    question = str(row.get("question") or "").strip()
    out.pop("classifier_label", None)
    out.pop("visual_filter_source", None)
    out.pop("detail", None)
    out.pop("non_visual_reason", None)
    out.pop("agreement", None)

    if blacklist_drop and is_non_visual_candidate(question):
        out["classifier_label"] = "NOT_DIRECTLY_VISUAL"
        out["visual_filter_source"] = VISUAL_FILTER_BLACKLIST
        out["detail"] = "blacklist"
        out["non_visual_reason"] = "BLACKLIST"
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
    llm_parallel: int = DEFAULT_LLM_PARALLEL,
) -> List[Dict[str, Any]]:
    """Score gold records with the production classifier cascade."""
    classifier = QuestionClassifier(
        host=host, model=model, parallel=llm_parallel
    )
    batch_n = max(1, int(batch_size))
    blacklist_drop = bool(fast_path)
    scored: List[Dict[str, Any]] = [
        _gate_record(row, blacklist_drop=blacklist_drop) for row in records
    ]
    llm_indices = [
        i
        for i, rec in enumerate(scored)
        if rec.get("classifier_label") is None
        and rec.get("visual_filter_source") == VISUAL_FILTER_LLM
    ]
    print(
        f"  gates done: {len(scored)} rows, "
        f"{len(scored) - len(llm_indices)} blacklist NDV, "
        f"{len(llm_indices)} UNKNOWN->llm "
        f"(batch_size={batch_n}, parallel={classifier.parallel})",
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
        for i, (label, reason) in zip(chunk_idxs, results):
            if label is None:
                scored[i]["classifier_label"] = "DIRECTLY_VISUAL"
                scored[i]["detail"] = detail or "parse_fail_keep"
                scored[i].pop("non_visual_reason", None)
            else:
                scored[i]["classifier_label"] = label
                scored[i]["detail"] = detail
                if reason:
                    scored[i]["non_visual_reason"] = reason
                else:
                    scored[i].pop("non_visual_reason", None)

    return scored


def row_agreement(row: Dict[str, Any]) -> Optional[bool]:
    """True/False when manual vs classifier can be compared; else None."""
    gold = str(row.get("manual_label") or "").strip().upper()
    pred = str(row.get("classifier_label") or "").strip().upper()
    if gold not in _VALID_LABELS or pred not in _VALID_LABELS:
        return None
    return gold == pred


def annotate_agreement(records: Sequence[Dict[str, Any]]) -> int:
    """Set ``agreement`` on each row; return disagreement count."""
    disagreements = 0
    for row in records:
        agreed = row_agreement(row)
        if agreed is None:
            row.pop("agreement", None)
            continue
        row["agreement"] = agreed
        if not agreed:
            disagreements += 1
    return disagreements


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score DevAuditor classifier_audit_manual.json with the "
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
        help=(
            f"UNKNOWN items flushed together to parallel classify_batch "
            f"(default {DEFAULT_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--llm-parallel",
        type=int,
        default=DEFAULT_LLM_PARALLEL,
        help=(
            f"Max concurrent classifier Ollama requests "
            f"(default {DEFAULT_LLM_PARALLEL}; set OLLAMA_NUM_PARALLEL "
            f">= this on the server)"
        ),
    )
    parser.add_argument(
        "--no-blacklist-drop",
        "--no-fast-path",
        action="store_true",
        help=(
            "Disable blacklist auto-NDV; send every question to the batched "
            "LLM (same as generate.py --no-blacklist-drop / --no-fast-path)"
        ),
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
    if args.llm_parallel < 1:
        print("--llm-parallel must be >= 1", file=sys.stderr)
        return 1

    info, records = load_gold(gold_path)
    info.pop("classifier_scoring", None)
    info.pop("disagreement_count", None)

    blacklist_drop = not args.no_blacklist_drop
    print(
        f"Gold classifier audit: {gold_path.name} n={len(records)} "
        f"llm={True} blacklist_drop={blacklist_drop} "
        f"llm_parallel={args.llm_parallel}",
        flush=True,
    )
    scored = classify_gold_records(
        records,
        host=args.host,
        model=args.model,
        batch_size=args.batch_size,
        fast_path=blacklist_drop,
        llm_parallel=args.llm_parallel,
    )
    disagreement_count = annotate_agreement(scored)
    print(
        f"Disagreements (manual_label vs classifier_label): {disagreement_count}",
        flush=True,
    )

    out_info = dict(info)
    out_info["disagreement_count"] = disagreement_count

    if args.in_place:
        out_path = gold_path
    elif args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = gold_path.with_name(f"{gold_path.stem}_scored.json")
        if gold_path.stem.endswith("_scored"):
            out_path = gold_path

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
