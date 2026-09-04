"""Re-run the blacklist-gate question classifier on a not_directly_visual sidecar.

Usage (from QuestionDependentCaptionGenerator/):

    python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json
    python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json 9002
    python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json --batch-size 10
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
    is_non_visual_candidate,
)

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
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


def _base_record(row: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Shared fields for one reclassify record (before label fill)."""
    return {
        "question_id": row.get("question_id"),
        "image_id": row.get("image_id"),
        "question": question,
        "answer": row.get("answer"),
        "prior_label": row.get("label"),
        "prior_visual_filter_source": row.get("visual_filter_source"),
    }


def _with_flip(
    record: Dict[str, Any],
    *,
    new_label: Optional[str],
    detail: str,
    non_visual_reason: Optional[str],
) -> Dict[str, Any]:
    """Attach new_label / detail / reason / flipped onto a partial record."""
    prior_label = record.get("prior_label")
    flipped = False
    if (
        prior_label is not None
        and new_label is not None
        and str(prior_label).strip().upper() != str(new_label).strip().upper()
    ):
        flipped = True
    record["new_label"] = new_label
    record["non_visual_reason"] = non_visual_reason
    record["detail"] = detail
    record["flipped"] = flipped
    return record


def diagnose_gate(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decide gate without calling the LLM.

    For ``llm_confirm`` rows, ``new_label`` is left unset until a batch
    confirm fills it.
    """
    question = str(row.get("question") or "").strip()
    record = _base_record(row, question)

    exempt_match = is_fast_path_visual(question)
    blacklist_match = is_non_visual_candidate(question)
    record["exempt_match"] = exempt_match
    record["blacklist_match"] = blacklist_match

    if exempt_match:
        record["gate"] = "fast_path"
        return _with_flip(
            record,
            new_label="DIRECTLY_VISUAL",
            detail="fast_path",
            non_visual_reason=None,
        )
    if not blacklist_match:
        record["gate"] = "default_visual"
        return _with_flip(
            record,
            new_label="DIRECTLY_VISUAL",
            detail="default_visual",
            non_visual_reason=None,
        )

    record["gate"] = "llm_confirm"
    # Label filled later by batched classify.
    record["new_label"] = None
    record["non_visual_reason"] = None
    record["detail"] = "pending"
    record["flipped"] = False
    return record


def _apply_llm_result(
    record: Dict[str, Any],
    *,
    label: Optional[str],
    detail: str,
    non_visual_reason: Optional[str],
) -> Dict[str, Any]:
    """Fill an llm_confirm record after classify_batch / salvage."""
    if label is None:
        # Fail-closed, same as production filter.
        return _with_flip(
            record,
            new_label="NOT_DIRECTLY_VISUAL",
            detail=detail or "parse_fail",
            non_visual_reason=None,
        )
    return _with_flip(
        record,
        new_label=label,
        detail=detail,
        non_visual_reason=non_visual_reason,
    )


def run_reclassify(
    rows: Sequence[Dict[str, Any]],
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Reclassify rows; LLM confirms are packed into batches."""
    classifier = QuestionClassifier(host=host, model=model)
    batch_n = max(1, int(batch_size))

    records: List[Dict[str, Any]] = [diagnose_gate(row) for row in rows]
    llm_indices = [
        i for i, rec in enumerate(records) if rec.get("gate") == "llm_confirm"
    ]
    n_llm = len(llm_indices)
    print(
        f"  gates done: {len(records)} rows, "
        f"{n_llm} llm_confirm (batch_size={batch_n})",
        flush=True,
    )
    if not llm_indices:
        return records

    for start in range(0, n_llm, batch_n):
        chunk_idxs = llm_indices[start : start + batch_n]
        questions = [str(records[i]["question"] or "") for i in chunk_idxs]
        end = start + len(chunk_idxs)
        print(
            f"  llm_confirm batch: {start + 1}-{end}/{n_llm} "
            f"(size={len(chunk_idxs)})",
            flush=True,
        )
        results, detail = classifier.classify_batch(questions)
        if results is None:
            # Salvage: one classify_one per item in this batch.
            for i, q in zip(chunk_idxs, questions):
                label, one_detail, reason = classifier.classify_one(q)
                records[i] = _apply_llm_result(
                    records[i],
                    label=label,
                    detail=one_detail or detail,
                    non_visual_reason=reason,
                )
        else:
            for i, (label, reason) in zip(chunk_idxs, results):
                records[i] = _apply_llm_result(
                    records[i],
                    label=label,
                    detail=detail,
                    non_visual_reason=reason,
                )
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
    batch_size: int,
) -> Dict[str, Any]:
    """Write reclassify JSON and return the payload."""
    new_counts = Counter(
        str(r.get("new_label") or "None") for r in records
    )
    gate_counts = Counter(str(r.get("gate") or "None") for r in records)
    flip_count = sum(1 for r in records if r.get("flipped"))
    payload = {
        "info": {
            "source": str(source.resolve()),
            "question_id_filter": question_id,
            "model": model,
            "host": host,
            "batch_size": batch_size,
            "num_input": num_input,
            "num_reclassified": len(records),
            "new_label_counts": dict(new_counts),
            "gate_counts": dict(gate_counts),
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"llm_confirm items per Ollama call (default {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source: Path = args.not_directly_visual_path
    if not source.is_file():
        raise FileNotFoundError(f"Sidecar file not found: {source}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

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
        f"n={len(selected)} model={args.model} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    records = run_reclassify(
        selected,
        host=args.host,
        model=args.model,
        batch_size=args.batch_size,
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
        batch_size=args.batch_size,
    )
    info = payload["info"]
    print(f"Wrote {len(records)} reclassify records -> {out_path}")
    print(
        f"  new_label_counts={info['new_label_counts']} "
        f"gate_counts={info['gate_counts']} "
        f"flip_count={info['flip_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
