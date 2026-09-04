"""Re-run the binary question classifier on a not_directly_visual sidecar.

Usage (from QuestionDependentCaptionGenerator/):

    python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json
    python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json 9002
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from question_classifier import (  # noqa: E402
    QuestionClassifier,
    is_fast_path_visual,
    is_non_visual_suspect,
)

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
OUTPUTS_DIR = _PKG_DIR / "outputs"


def load_sidecar(path: Path) -> List[Dict[str, Any]]:
    """Load annotations from a not_directly_visual sidecar JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object with info + annotations, got {type(data).__name__}"
        )
    rows = data.get("annotations")
    if not isinstance(rows, list):
        raise ValueError("annotations must be a list")
    return rows


def select_rows(
    rows: Sequence[Dict[str, Any]],
    question_id: Optional[int],
) -> List[Dict[str, Any]]:
    """Return all rows, or the single row matching question_id."""
    if question_id is None:
        return list(rows)
    matched = [
        r
        for r in rows
        if r.get("question_id") is not None
        and int(r["question_id"]) == int(question_id)
    ]
    if not matched:
        raise ValueError(
            f"question_id {question_id} not found in annotations"
        )
    return matched


def diagnose_and_reclassify(
    row: Dict[str, Any],
    classifier: QuestionClassifier,
) -> Dict[str, Any]:
    """Fast-path diagnostics + classify_one for one annotation."""
    question = str(row.get("question") or "").strip()
    prior_label = row.get("label")
    prior_source = row.get("visual_filter_source")

    fast_path = is_fast_path_visual(question)
    suspect = is_non_visual_suspect(question)
    would_skip_llm = bool(fast_path)

    # Always call the LLM so the report tests the classifier itself;
    # would_skip_llm shows what the production Fast Path gate would do.
    new_label, detail = classifier.classify_one(question)

    flipped = False
    if (
        prior_label is not None
        and new_label is not None
        and str(prior_label).strip().upper() != str(new_label).strip().upper()
    ):
        flipped = True

    return {
        "question_id": row.get("question_id"),
        "image_id": row.get("image_id"),
        "question": question,
        "answer": row.get("answer"),
        "prior_label": prior_label,
        "prior_visual_filter_source": prior_source,
        "fast_path_match": fast_path,
        "suspect_match": suspect,
        "would_skip_llm": would_skip_llm,
        "new_label": new_label,
        "detail": detail,
        "flipped": flipped,
    }


def run_reclassify(
    rows: Sequence[Dict[str, Any]],
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Reclassify each row; log progress."""
    classifier = QuestionClassifier(host=host, model=model)
    records: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        print(f"  reclassify: {i}/{total}", flush=True)
        records.append(diagnose_and_reclassify(row, classifier))
    return records


def output_path_for(source: Path, question_id: Optional[int]) -> Path:
    """Build outputs/ path for all-items or single-qid report."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if question_id is None:
        name = f"{source.stem}_reclassify.json"
    else:
        name = f"{source.stem}_reclassify_qid{question_id}.json"
    return OUTPUTS_DIR / name


def write_report(
    out_path: Path,
    *,
    source: Path,
    records: List[Dict[str, Any]],
    num_input: int,
    question_id: Optional[int],
    model: str,
    host: str,
) -> Dict[str, Any]:
    """Write reclassify JSON and return the payload."""
    new_counts = Counter(
        str(r.get("new_label") or "None") for r in records
    )
    flip_count = sum(1 for r in records if r.get("flipped"))
    payload = {
        "info": {
            "source": str(source.resolve()),
            "question_id_filter": question_id,
            "model": model,
            "host": host,
            "num_input": num_input,
            "num_reclassified": len(records),
            "new_label_counts": dict(new_counts),
            "flip_count": flip_count,
        },
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "not_directly_visual_path",
        type=Path,
        help=(
            "Path to *_not_directly_visual.json "
            "(info + annotations with question_id, question, label, …)"
        ),
    )
    parser.add_argument(
        "question_id",
        nargs="?",
        default=None,
        type=int,
        help="Optional: reclassify only this question_id (default: all)",
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source: Path = args.not_directly_visual_path
    if not source.is_file():
        raise FileNotFoundError(f"Sidecar file not found: {source}")

    rows = load_sidecar(source)
    selected = select_rows(rows, args.question_id)
    if not selected:
        raise ValueError("No annotations to reclassify")

    qid_msg = (
        f"question_id={args.question_id}"
        if args.question_id is not None
        else "all items"
    )
    print(
        f"Classifier retest: source={source} {qid_msg} "
        f"n={len(selected)} model={args.model}",
        flush=True,
    )

    records = run_reclassify(
        selected,
        host=args.host,
        model=args.model,
    )
    out_path = output_path_for(source, args.question_id)
    payload = write_report(
        out_path,
        source=source,
        records=records,
        num_input=len(rows),
        question_id=args.question_id,
        model=args.model,
        host=args.host,
    )
    info = payload["info"]
    print(f"Wrote {len(records)} reclassify records -> {out_path}")
    print(
        f"  new_label_counts={info['new_label_counts']} "
        f"flip_count={info['flip_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
