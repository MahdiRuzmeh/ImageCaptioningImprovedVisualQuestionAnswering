"""CLI baraye generate kardan-e question-dependent captions az VQA v2.

Run az in folder:

    python generate.py --split train
    python generate.py --split val
    python generate.py --split train --max-items 1000   # smoke test
    python generate.py --split val --llm --batch-size 10 \\
        --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 50

Output default: ./outputs/vqa_v2_question_dependent_captions_{train,val}2014.json

Har run classifier-e binary (DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL) ro
ejra mikone — Ollama baraye classifier lazem ast hata bedoon ``--llm``.

Resume (Ctrl+C safe):
    hamoon command ro dobare bezan — classifier + LLM az checkpoint edame mide.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from caption_rules import answer_mode_stats, generate_caption, is_ocr_question
from llm_client import (
    ItemOutcome,
    OllamaClient,
    run_batches_concurrent,
)
from llm_prompts import PROMPT_VERSION
from question_classifier import (
    CLASSIFIER_PROMPT_VERSION,
    QuestionClassifier,
    delete_classifier_checkpoint,
    filter_non_visual_questions,
    load_classifier_checkpoint,
)
from validation import (
    VALIDATOR_VERSION,
    ValidationConfig,
    ValidationLogWriter,
    _VALIDATION_FAIL_REASONS,
    fast_validate,
    validation_log_path,
)
from validation.fast_validator import FastVerdict
from validation.pipeline import validate_rows, ValidationStats

PROJECT_ROOT = Path(__file__).resolve().parent
# VQA raw data az ../dataset; caption JSON inja save mishe
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Default paths — input az dataset/, output dakhele in folder
# ---------------------------------------------------------------------------

SPLIT_PATHS: Dict[str, Dict[str, Path]] = {
    "train": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_train2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_train2014_annotations.json",
        "output": OUTPUT_ROOT / "vqa_v2_question_dependent_captions_train2014.json",
        "classification_result": (
            OUTPUT_ROOT / "vqa_v2_question_classification_result.json"
        ),
    },
    "val": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_val2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_val2014_annotations.json",
        "output": OUTPUT_ROOT / "vqa_v2_question_dependent_captions_val2014.json",
        "classification_result": (
            OUTPUT_ROOT / "vqa_v2_question_classification_result_val2014.json"
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chunked(
    items: Sequence[Any],
    size: int,
) -> List[List[Any]]:
    """List ro be batch haye size N chop mikone."""
    n = max(1, int(size))
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def now_iso() -> str:
    """Local timezone ISO-8601 timestamp (seconds precision)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def duration_seconds(started_at: str, ended_at: str) -> float:
    """Elapsed seconds between two ISO timestamps; 0.0 if unparsable."""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(ended_at)
    except ValueError:
        return 0.0
    return max(0.0, round((end - start).total_seconds(), 3))


def recount_rules(rows: List[Dict[str, Any]]) -> Counter:
    """Az rows, Counter rule ha ro dobare hesab kon."""
    return Counter(str(r.get("rule", "unknown")) for r in rows)


def load_output_payload(output_path: Path) -> Optional[Dict[str, Any]]:
    """Output JSON ro load kon; age corrupt bashe None."""
    if not output_path.is_file():
        return None
    try:
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_existing_llm_map(output_path: Path) -> Dict[int, Dict[str, Any]]:
    """Age output ghablan hast, llm_fallback ha ro baraye resume load kon.

    Returns:
        map question_id -> row (faghat rule=llm_fallback)
    """
    data = load_output_payload(output_path)
    if not data:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in data.get("annotations") or []:
        if str(row.get("rule")) != "llm_fallback":
            continue
        try:
            qid = int(row["question_id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[qid] = row
    return out


def count_vqa_overlap(
    questions_json: Path,
    annotations_json: Path,
    max_items: Optional[int] = None,
) -> int:
    """Tedad sample moshtarak questions∩annotations (ba max_items)."""
    with questions_json.open("r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with annotations_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    qids = set(int(x["question_id"]) for x in questions) & set(
        int(x["question_id"]) for x in annotations
    )
    n = len(qids)
    if max_items is not None and max_items > 0:
        n = min(n, max_items)
    return n


def classification_result_path(split: str) -> Path:
    """Persistent question-classification result (also used as resume checkpoint).

    Train: ``vqa_v2_question_classification_result.json``
    Val: ``vqa_v2_question_classification_result_val2014.json``
    """
    return SPLIT_PATHS[split]["classification_result"]


def resolve_post_filter_row_count(
    output_path: Path,
    classification_path: Optional[Path] = None,
) -> Optional[int]:
    """Expected annotation count after OCR/dedup/classifier filters."""
    if classification_path is not None:
        ckpt = load_classifier_checkpoint(classification_path)
        if ckpt:
            info = ckpt.get("info") or {}
            if info.get("status") == "complete":
                pf = info.get("post_filter_count")
                if pf is not None:
                    return int(pf)
                kept = ckpt.get("kept")
                if isinstance(kept, list):
                    return len(kept)
    data = load_output_payload(output_path)
    if not data:
        return None
    info = data.get("info") or {}
    for key in ("post_filter_count", "directly_visual_count", "num_samples"):
        val = info.get(key)
        if val is not None:
            return int(val)
    rows = data.get("annotations")
    if isinstance(rows, list) and rows:
        return len(rows)
    return None


def try_load_checkpoint_rows(
    output_path: Path,
    classification_path: Optional[Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Load output rows when count matches post-filter expectation (resume fast-path)."""
    data = load_output_payload(output_path)
    if not data:
        return None
    rows = data.get("annotations")
    if not isinstance(rows, list) or not rows:
        return None
    expected = resolve_post_filter_row_count(
        output_path,
        classification_path=classification_path,
    )
    if expected is not None and len(rows) != expected:
        return None
    for row in rows[:3]:
        if "question_id" not in row or "rule" not in row:
            return None
    return list(rows)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def load_vqa_pairs(
    questions_json: Path,
    annotations_json: Path,
    max_items: Optional[int] = None,
    min_consensus: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Counter, int, int, int, List[Dict[str, Any]], int]:
    """Soal va javab haye VQA v2 ro load kon va ba rule caption besaz.

    Filter order is fixed: OCR → answer consensus → dedup. Consensus runs
    before dedup so a dropped low-consensus pair never claims a dedup slot.

    Rule captions are validated with the same high-precision checks used for
    LLM captions (Comments8: validation must not be LLM-only). A rule caption
    that fails a hard check becomes ``needs_llm`` so the SLM can rewrite it
    instead of shipping a broken template.

    Args:
        questions_json: path be v2_OpenEnded_*_questions.json
        annotations_json: path be v2_mscoco_*_annotations.json
        max_items: age set shode, faghat N sample aval (smoke test)
        min_consensus: drop pairs whose ``answer_consensus`` is below this
            (0.0 = off). Annotator agreement, not model confidence.

    Returns:
        (rows, rule_counts, ocr_excluded_count, duplicate_count, input_count,
        low_consensus_rows, rule_validation_reject_count)
    """
    print(f"Loading VQA JSON: {questions_json.name} + {annotations_json.name} ...")
    with questions_json.open("r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with annotations_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    print(
        f"Loaded {len(questions)} questions, {len(annotations)} annotations "
        "(building rows + rules) ..."
    )

    qmap = {int(x["question_id"]): x for x in questions}
    amap = {int(x["question_id"]): x for x in annotations}
    qids = sorted(set(qmap.keys()) & set(amap.keys()))
    if max_items is not None and max_items > 0:
        qids = qids[:max_items]

    rows: List[Dict[str, Any]] = []
    rule_counts: Counter = Counter()
    seen_per_image: Dict[int, Set[Tuple[str, str]]] = {}
    low_consensus_rows: List[Dict[str, Any]] = []
    duplicates_dropped = 0
    ocr_excluded = 0
    rule_validation_rejects = 0
    input_count = len(qids)
    total_qids = len(qids)
    progress_every = max(500, total_qids // 20) if total_qids else 500

    for i, qid in enumerate(qids, start=1):
        q = qmap[qid]
        ann = amap[qid]
        image_id = int(q["image_id"])
        question_type = str(ann.get("question_type") or "")

        if is_ocr_question(q["question"], question_type):
            ocr_excluded += 1
            continue

        answers = [x["answer"] for x in ann["answers"]]
        ans, answer_count, answer_consensus = answer_mode_stats(answers)

        if min_consensus > 0.0 and answer_consensus < min_consensus:
            low_consensus_rows.append(
                {
                    "question_id": qid,
                    "image_id": image_id,
                    "question": q["question"],
                    "answer": ans,
                    "answer_count": answer_count,
                    "answer_consensus": answer_consensus,
                    "detail": f"consensus<{min_consensus}",
                }
            )
            continue

        dedup_key = (q["question"].strip().lower(), ans)
        seen = seen_per_image.setdefault(image_id, set())
        if dedup_key in seen:
            duplicates_dropped += 1
            continue
        seen.add(dedup_key)

        caption, rule = generate_caption(q["question"], ans)
        if caption:
            fast = fast_validate(q["question"], ans, caption)
            if fast.verdict == FastVerdict.FAIL:
                rule_validation_rejects += 1
                caption, rule = "", "needs_llm"
        rule_counts[rule] += 1

        rows.append(
            {
                "question_id": qid,
                "image_id": image_id,
                "question": q["question"],
                "answer": ans,
                "answer_count": answer_count,
                "answer_consensus": answer_consensus,
                "caption": caption,
                "rule": rule,
            }
        )
        if i % progress_every == 0 or i == total_qids:
            print(
                f"  rules progress: {i}/{total_qids} scanned "
                f"({len(rows)} kept, {ocr_excluded} OCR dropped)",
                flush=True,
            )

    if ocr_excluded:
        print(
            f"OCR filter: {ocr_excluded} OCR-dependent question/answer pairs "
            "excluded (is_ocr_question) — captioner cannot read rendered text."
        )
    if low_consensus_rows:
        print(
            f"Consensus filter: {len(low_consensus_rows)} pairs dropped with "
            f"answer_consensus < {min_consensus} — annotators disagreed, so the "
            "caption target is unreliable."
        )
    if duplicates_dropped:
        print(
            f"Dedup: {duplicates_dropped} duplicate (image_id, question, answer) "
            "rows dropped — kept the first occurrence of each."
        )
    if rule_validation_rejects:
        print(
            f"Rule validation: {rule_validation_rejects} rule captions failed a "
            "hard check and were routed to the LLM instead."
        )
    print(
        f"Rule pass done: {len(rows)} rows "
        f"({rule_counts.get('needs_llm', 0)} needs_llm)",
        flush=True,
    )

    return (
        rows,
        rule_counts,
        ocr_excluded,
        duplicates_dropped,
        input_count,
        low_consensus_rows,
        rule_validation_rejects,
    )


def not_directly_visual_path(output_path: Path) -> Path:
    """Sidecar JSON for questions dropped as NOT_DIRECTLY_VISUAL."""
    return output_path.with_name(output_path.stem + "_not_directly_visual.json")


def low_consensus_path(output_path: Path) -> Path:
    """Sidecar JSON for pairs dropped by the answer-consensus threshold."""
    return output_path.with_name(output_path.stem + "_low_consensus.json")


def _write_sidecar(
    side: Path,
    output_path: Path,
    description: str,
    dropped_rows: List[Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically write a dropped-rows sidecar next to the captions JSON."""
    info: Dict[str, Any] = {
        "description": description,
        "source_output": str(output_path.resolve()),
        "num_samples": len(dropped_rows),
    }
    if extra_info:
        info.update(extra_info)
    payload = {"info": info, "annotations": dropped_rows}
    side.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=side.stem + "_",
        suffix=".tmp.json",
        dir=str(side.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, side)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return side


def write_not_directly_visual_sidecar(
    output_path: Path,
    dropped_rows: List[Dict[str, Any]],
) -> Path:
    """Persist classifier-dropped questions for later VISUAL vs non-VISUAL analysis."""
    return _write_sidecar(
        not_directly_visual_path(output_path),
        output_path,
        "Questions dropped as NOT_DIRECTLY_VISUAL "
        "(captioner-training filter only; do not alter raw VQA2 eval)",
        dropped_rows,
    )


def write_low_consensus_sidecar(
    output_path: Path,
    dropped_rows: List[Dict[str, Any]],
    min_consensus: float,
) -> Path:
    """Persist pairs dropped below the answer-consensus threshold."""
    return _write_sidecar(
        low_consensus_path(output_path),
        output_path,
        "Question/answer pairs dropped because annotator agreement was below "
        "--min-consensus (captioner-training filter only; "
        "do not alter raw VQA2 eval)",
        dropped_rows,
        extra_info={"min_consensus": min_consensus},
    )


def count_validation_stats(
    outcomes: Sequence[ItemOutcome],
) -> Tuple[int, int]:
    """Count validation regenerations and final validation failures."""
    retries = 0
    failures = 0
    for o in outcomes:
        retries += sum(1 for a in o.attempts if a.startswith("single#"))
        if o.caption is None and o.reason in _VALIDATION_FAIL_REASONS:
            failures += 1
    return retries, failures


def merge_llm_resume(
    rows: List[Dict[str, Any]],
    resume_map: Dict[int, Dict[str, Any]],
) -> int:
    """llm_fallback haye save-shode ro roye rows restore kon.

    Returns:
        tedad row ke restore shod.
    """
    restored = 0
    for row in rows:
        if row["rule"] != "needs_llm":
            continue
        prev = resume_map.get(int(row["question_id"]))
        if prev is None:
            continue
        cap = str(prev.get("caption") or "").strip()
        if not cap:
            continue
        row["caption"] = cap
        row["rule"] = "llm_fallback"
        restored += 1
    return restored


def llm_failure_log_path(output_path: Path) -> Path:
    """Sidecar log path next to captions JSON (``*.json.llm_failures``)."""
    return output_path.with_suffix(output_path.suffix + ".llm_failures")


def retry_audit_path(output_path: Path) -> Path:
    """Sidecar JSONL with one record per retried item."""
    return output_path.with_name(output_path.stem + "_validation_audit.jsonl")


class RetryAuditLogger:
    """JSONL log of every retried item, whether or not the retry succeeded.

    Comments8 item 8: the run must show which samples were regenerated and
    what happened to them. ``validation_retry_count`` alone hides that, and
    the failure log only ever recorded rows that stayed ``needs_llm``.

    One record per retry event::

        {"question_id": 1, "retry_kind": "validator",
         "first_caption": "...", "failure_reason": "answer_mismatch",
         "retry_caption": "...", "final_result": "accepted"}

    ``retry_kind`` is ``validator`` when a caption was produced but rejected,
    and ``generation`` when the model failed to produce a parsable caption
    (``parse_*``, ``timeout``, ``empty_response``, connection error).
    """

    def __init__(self, path: Path) -> None:
        """Open (overwrite) the audit log."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.count = 0
        self.kind_counts: Counter = Counter()
        self.result_counts: Counter = Counter()

    @staticmethod
    def should_log(outcome: ItemOutcome) -> bool:
        """True when the item went through at least one retry."""
        if outcome.first_caption is not None:
            return True
        return any(a.startswith("single#") for a in outcome.attempts)

    def log_retry(
        self,
        *,
        question_id: int,
        question: str,
        answer: str,
        stage: str,
        outcome: ItemOutcome,
    ) -> None:
        """Append one retry record."""
        accepted = outcome.caption is not None
        kind = outcome.retry_kind or "generation"
        record: Dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "stage": stage,
            "retry_kind": kind,
            "first_caption": outcome.first_caption,
            "failure_reason": outcome.first_reason or outcome.reason,
            "retry_caption": outcome.caption,
            "final_result": "accepted" if accepted else "dropped",
            "final_reason": outcome.reason,
            "attempts": list(outcome.attempts),
            "validation_flags": list(outcome.flags),
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1
        self.kind_counts[kind] += 1
        self.result_counts[record["final_result"]] += 1

    def summary(self) -> str:
        """One-line summary for the console."""
        return (
            f"{self.count} retry records "
            f"(validator={self.kind_counts.get('validator', 0)}, "
            f"generation={self.kind_counts.get('generation', 0)}; "
            f"accepted={self.result_counts.get('accepted', 0)}, "
            f"dropped={self.result_counts.get('dropped', 0)})"
        )

    def close(self) -> None:
        """Close the underlying file handle."""
        try:
            self._fh.close()
        except Exception:
            pass


class LlmFailureLogger:
    """Append-only log file that explains why LLM could not resolve ``needs_llm``.

    Written next to the captions JSON so connection / parse / answer-mismatch
    reasons are visible without silent leftovers.
    """

    def __init__(self, path: Path) -> None:
        """Open (overwrite) the failure log and write a header."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.count = 0
        self.reason_counts: Counter = Counter()
        self._fh.write(
            "# LLM fallback failure log\n"
            "# Each block = one Q+A that stayed rule=needs_llm after LLM attempts.\n"
            f"# log_path={self.path.resolve()}\n"
            f"{'=' * 72}\n"
        )
        self._fh.flush()

    def log_failure(
        self,
        *,
        question_id: int,
        question: str,
        answer: str,
        template_caption: str,
        outcome: ItemOutcome,
    ) -> None:
        """Record one failed LLM upgrade with reason + attempt trail."""
        self.count += 1
        self.reason_counts[outcome.reason] += 1
        attempts = " -> ".join(outcome.attempts) if outcome.attempts else "(none)"
        self._fh.write(
            f"\n[{self.count}] question_id={question_id}\n"
            f"  reason:   {outcome.reason}\n"
            f"  detail:   {outcome.detail}\n"
            f"  attempts: {attempts}\n"
            f"  Q: {question}\n"
            f"  A: {answer}\n"
            f"  template_caption: {template_caption}\n"
        )
        self._fh.flush()

    def write_summary(self, still_needs_llm: int) -> None:
        """Append reason histogram + leftover count."""
        self._fh.write(f"\n{'=' * 72}\n")
        self._fh.write("SUMMARY\n")
        self._fh.write(f"  failures_logged: {self.count}\n")
        self._fh.write(f"  still_rule_needs_llm: {still_needs_llm}\n")
        if self.reason_counts:
            self._fh.write("  by_reason:\n")
            for reason, n in self.reason_counts.most_common():
                self._fh.write(f"    {reason}: {n}\n")
        self._fh.write(
            "\nIf still_rule_needs_llm > 0, the system did NOT fully succeed.\n"
            "Fix the top reason (Ollama down, parse_length_mismatch, "
            "answer_mismatch, …) and re-run the same --llm command.\n"
        )
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        try:
            self._fh.close()
        except Exception:
            pass


def apply_llm_fallbacks(
    rows: List[Dict[str, Any]],
    *,
    client: OllamaClient,
    batch_size: int,
    workers: int,
    checkpoint_every: int,
    output_path: Path,
    questions_json: Path,
    annotations_json: Path,
    llm_meta: Dict[str, Any],
    resume_map: Optional[Dict[int, Dict[str, Any]]] = None,
    failure_log: Optional[LlmFailureLogger] = None,
    retry_audit: Optional[RetryAuditLogger] = None,
    single_retries: int = 1,
    final_retries: int = 1,
    ocr_excluded_count: int = 0,
    dropped_empty_count: int = 0,
    duplicate_count: int = 0,
    input_count: int = 0,
    directly_visual_count: int = 0,
    not_directly_visual_count: int = 0,
    validation_retry_count: int = 0,
    validation_failure_count: int = 0,
    classifier_meta: Optional[Dict[str, Any]] = None,
    low_consensus_excluded_count: int = 0,
    min_consensus: float = 0.0,
    rule_validation_reject_count: int = 0,
    process_started_at: Optional[str] = None,
) -> Tuple[Counter, int, int]:
    """Row haye rule=needs_llm ro ba packed LLM caption update mikone.

    Returns:
        (rule_counts, validation_retry_count, validation_failure_count)
    """
    resume_map = resume_map or {}
    restored = merge_llm_resume(rows, resume_map)
    if restored:
        print(f"Resume: {restored} llm_fallback az checkpoint restore shod")

    pending_idx: List[int] = [
        i for i, r in enumerate(rows) if r["rule"] == "needs_llm"
    ]
    already = sum(1 for r in rows if r["rule"] == "llm_fallback")
    print(
        f"LLM fallback: {len(pending_idx)} pending, {already} already done "
        f"(batch-size={batch_size}, workers={workers}, "
        f"checkpoint-every={checkpoint_every}, num_ctx={client.num_ctx})",
        flush=True,
    )

    if not pending_idx:
        print("Hichi pending nist — LLM pass skip.")
        return recount_rules(rows), validation_retry_count, validation_failure_count

    pairs = [(rows[i]["question"], rows[i]["answer"]) for i in pending_idx]
    indexed = list(zip(pending_idx, pairs))
    batches_idx = chunked(indexed, batch_size)
    batches_pairs: List[List[Tuple[str, str]]] = [
        [(q, a) for _, (q, a) in batch] for batch in batches_idx
    ]

    last_outcome: Dict[int, ItemOutcome] = {}
    done_batches = 0
    total_batches = len(batches_pairs)
    print(
        f"Starting LLM captioning: {total_batches} batches "
        f"(first batch may take a while if Ollama is loading the model)...",
        flush=True,
    )

    def _checkpoint() -> None:
        counts = recount_rules(rows)
        write_output_json(
            output_path,
            rows,
            counts,
            questions_json,
            annotations_json,
            llm_meta=llm_meta,
            ocr_excluded_count=ocr_excluded_count,
            dropped_empty_count=dropped_empty_count,
            duplicate_count=duplicate_count,
            input_count=input_count,
            directly_visual_count=directly_visual_count,
            not_directly_visual_count=not_directly_visual_count,
            validation_retry_count=validation_retry_count,
            validation_failure_count=validation_failure_count,
            classifier_meta=classifier_meta,
            low_consensus_excluded_count=low_consensus_excluded_count,
            min_consensus=min_consensus,
            rule_validation_reject_count=rule_validation_reject_count,
            process_started_at=process_started_at,
        )

    def _record_outcome(row_i: int, outcome: ItemOutcome, stage: str) -> None:
        """Apply one outcome to its row and audit-log any retry it went through."""
        last_outcome[row_i] = outcome
        if outcome.caption is not None:
            rows[row_i]["caption"] = outcome.caption
            rows[row_i]["rule"] = "llm_fallback"
            if outcome.flags:
                rows[row_i]["validation_flags"] = list(outcome.flags)
            else:
                rows[row_i].pop("validation_flags", None)
        if retry_audit is not None and RetryAuditLogger.should_log(outcome):
            retry_audit.log_retry(
                question_id=int(rows[row_i]["question_id"]),
                question=str(rows[row_i]["question"]),
                answer=str(rows[row_i]["answer"]),
                stage=stage,
                outcome=outcome,
            )

    def _on_batch_start(batch_i: int, batch_len: int) -> None:
        print(
            f"  LLM batch {batch_i + 1}/{total_batches} calling Ollama "
            f"({batch_len} Q+A)...",
            flush=True,
        )

    def _on_batch(batch_i: int, outcomes: List[ItemOutcome]) -> None:
        nonlocal done_batches, validation_retry_count, validation_failure_count
        ok_n = 0
        r, f = count_validation_stats(outcomes)
        validation_retry_count += r
        # failures counted at end from leftovers; retries accumulate here
        for j, outcome in enumerate(outcomes):
            row_i = batches_idx[batch_i][j][0]
            _record_outcome(row_i, outcome, "main")
            if outcome.caption is not None:
                ok_n += 1
        done_batches += 1
        still = sum(1 for r in rows if r["rule"] == "needs_llm")
        print(
            f"  LLM batch {batch_i + 1}/{total_batches} done "
            f"(accepted {ok_n}/{len(outcomes)}, "
            f"{still} needs_llm left)",
            flush=True,
        )
        if checkpoint_every > 0 and done_batches % checkpoint_every == 0:
            _checkpoint()
            print(f"  checkpoint saved -> {output_path}", flush=True)

    run_batches_concurrent(
        client,
        batches_pairs,
        workers=workers,
        on_batch_done=_on_batch,
        on_batch_start=_on_batch_start,
        single_retries=single_retries,
    )

    leftover = [i for i, r in enumerate(rows) if r["rule"] == "needs_llm"]
    if leftover and final_retries > 0:
        for round_i in range(1, final_retries + 1):
            leftover = [i for i, r in enumerate(rows) if r["rule"] == "needs_llm"]
            if not leftover:
                break
            salvage_indexed = [
                (row_i, (rows[row_i]["question"], rows[row_i]["answer"]))
                for row_i in leftover
            ]
            salvage_batches_idx = chunked(salvage_indexed, batch_size)
            salvage_pairs: List[List[Tuple[str, str]]] = [
                [(q, a) for _, (q, a) in batch] for batch in salvage_batches_idx
            ]
            n_salvage = len(salvage_pairs)
            print(
                f"Final salvage round {round_i}/{final_retries}: "
                f"{len(leftover)} leftovers in {n_salvage} packed batches "
                f"(batch-size={batch_size}, one single-item retry per leftover)",
                flush=True,
            )

            def _make_salvage_callbacks(
                batches_idx_local: List[List[Any]],
                n_local: int,
            ) -> Tuple[
                Callable[[int, int], None],
                Callable[[int, List[ItemOutcome]], None],
            ]:
                def _start(batch_i: int, batch_len: int) -> None:
                    print(
                        f"  salvage batch {batch_i + 1}/{n_local} calling Ollama "
                        f"({batch_len} Q+A)...",
                        flush=True,
                    )

                def _done(batch_i: int, outcomes: List[ItemOutcome]) -> None:
                    nonlocal validation_retry_count
                    ok_n = 0
                    r, _f = count_validation_stats(outcomes)
                    validation_retry_count += r
                    for j, outcome in enumerate(outcomes):
                        row_i = batches_idx_local[batch_i][j][0]
                        _record_outcome(row_i, outcome, "salvage")
                        if outcome.caption is not None:
                            ok_n += 1
                    still = sum(1 for r in rows if r["rule"] == "needs_llm")
                    print(
                        f"  salvage batch {batch_i + 1}/{n_local} done "
                        f"(accepted {ok_n}/{len(outcomes)}, "
                        f"{still} needs_llm left)",
                        flush=True,
                    )

                return _start, _done

            s_start, s_done = _make_salvage_callbacks(salvage_batches_idx, n_salvage)
            # Comments8 item 9: never drop a parse failure without trying the
            # item on its own first — a packed batch that failed to parse says
            # nothing about the individual Q+A.
            run_batches_concurrent(
                client,
                salvage_pairs,
                workers=workers,
                on_batch_done=s_done,
                on_batch_start=s_start,
                single_retries=1,
            )

    if failure_log is not None:
        for row_i, row in enumerate(rows):
            if row["rule"] != "needs_llm":
                continue
            outcome = last_outcome.get(
                row_i,
                ItemOutcome(
                    reason="unknown",
                    detail="no LLM outcome recorded for this row",
                ),
            )
            failure_log.log_failure(
                question_id=int(row["question_id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                template_caption=str(row["caption"]),
                outcome=outcome,
            )
        still_logged = sum(1 for r in rows if r["rule"] == "needs_llm")
        failure_log.write_summary(still_logged)

    # Final validation failure count from leftovers with validation reject reasons
    validation_failure_count = 0
    for row_i, row in enumerate(rows):
        if row["rule"] != "needs_llm":
            continue
        outcome = last_outcome.get(row_i)
        if outcome is not None and outcome.reason in _VALIDATION_FAIL_REASONS:
            validation_failure_count += 1
        elif outcome is None:
            validation_failure_count += 1

    still = sum(1 for r in rows if r["rule"] == "needs_llm")
    if still:
        log_hint = (
            f" — see {failure_log.path}" if failure_log is not None else ""
        )
        print(
            f"WARNING: {still} rows still rule=needs_llm after LLM"
            f"{log_hint}. System did not fully succeed."
        )
    else:
        print("LLM fallback: all pending rows upgraded to llm_fallback.")

    return recount_rules(rows), validation_retry_count, validation_failure_count


def final_validation_pass(
    rows: List[Dict[str, Any]],
    *,
    client: Optional[OllamaClient] = None,
    config: Optional[ValidationConfig] = None,
    log_writer: Optional[ValidationLogWriter] = None,
    use_llm: bool = True,
) -> Tuple[List[Dict[str, Any]], int, int, List[Dict[str, Any]], ValidationStats]:
    """Validate every caption once more via fast + batched LLM judge.

    Returns:
        (kept_rows, dropped_count, flagged_count, failed_rows, stats)
    """
    cfg = config or ValidationConfig()
    kept, failed, stats = validate_rows(
        rows,
        config=cfg,
        client=client,
        use_llm=use_llm and client is not None,
        log_writer=log_writer,
    )
    flagged = sum(1 for r in kept if r.get("validation_flags"))
    return kept, len(failed), flagged, failed, stats


def drop_empty_or_short_captions(
    rows: List[Dict[str, Any]],
    *,
    min_words: int = 3,
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove rows with empty, whitespace-only, or short captions.

    Also drops leftover ``needs_llm`` rows so they never reach a DataLoader.
    Returns (kept_rows, dropped_count).
    """
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        cap = str(row.get("caption") or "").strip()
        rule = str(row.get("rule") or "")
        if rule == "needs_llm" or not cap or len(cap.split()) < min_words:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def write_output_json(
    output_path: Path,
    rows: List[Dict[str, Any]],
    rule_counts: Counter,
    questions_json: Path,
    annotations_json: Path,
    llm_meta: Optional[Dict[str, Any]] = None,
    ocr_excluded_count: int = 0,
    dropped_empty_count: int = 0,
    duplicate_count: int = 0,
    input_count: int = 0,
    directly_visual_count: int = 0,
    not_directly_visual_count: int = 0,
    validation_retry_count: int = 0,
    validation_failure_count: int = 0,
    classifier_meta: Optional[Dict[str, Any]] = None,
    low_consensus_excluded_count: int = 0,
    min_consensus: float = 0.0,
    validation_flagged_count: int = 0,
    rule_validation_reject_count: int = 0,
    validation_meta: Optional[Dict[str, Any]] = None,
    process_started_at: Optional[str] = None,
) -> None:
    """Natije ro atomic be JSON file save kon (crash-safe)."""
    ended_at = now_iso()
    started_at = process_started_at or ended_at
    info: Dict[str, Any] = {
        "description": "VQA v2 question-dependent captions (rule-based Q+A → statement)",
        "source_questions": str(questions_json),
        "source_annotations": str(annotations_json),
        "num_samples": len(rows),
        "post_filter_count": directly_visual_count or len(rows),
        "input_count": input_count,
        "directly_visual_count": directly_visual_count,
        "not_directly_visual_count": not_directly_visual_count,
        "ocr_excluded_count": ocr_excluded_count,
        "duplicate_count": duplicate_count,
        "min_consensus": min_consensus,
        "low_consensus_excluded_count": low_consensus_excluded_count,
        "dropped_empty_count": dropped_empty_count,
        "validation_retry_count": validation_retry_count,
        "validation_failure_count": validation_failure_count,
        "validation_flagged_count": validation_flagged_count,
        "rule_validation_reject_count": rule_validation_reject_count,
        "process_started_at": started_at,
        "process_ended_at": ended_at,
        "process_duration_seconds": duration_seconds(started_at, ended_at),
        "rule_counts": dict(rule_counts),
    }
    if classifier_meta:
        info["question_classifier"] = classifier_meta
    if llm_meta:
        info["description"] = (
            "VQA v2 question-dependent captions (rules + optional LLM fallback)"
        )
        info["llm"] = llm_meta
    if validation_meta:
        info["validation"] = validation_meta

    payload = {
        "info": info,
        "annotations": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_path.stem + "_",
        suffix=".tmp.json",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, output_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def print_stats(
    rows: List[Dict[str, Any]],
    rule_counts: Counter,
    output_path: Path,
    ocr_excluded_count: int = 0,
    duplicate_count: int = 0,
    not_directly_visual_count: int = 0,
    low_consensus_excluded_count: int = 0,
    min_consensus: float = 0.0,
    process_started_at: Optional[str] = None,
) -> None:
    """Statistik rule ha ro chap kon ta befahmim cheghadr needs_llm darim."""
    total = len(rows)
    print(f"Wrote {total} captions -> {output_path}")
    if process_started_at:
        ended_at = now_iso()
        secs = duration_seconds(process_started_at, ended_at)
        print(
            f"  process: {process_started_at} -> {ended_at} "
            f"({secs:.1f}s)"
        )
    if ocr_excluded_count:
        print(f"  (excluded {ocr_excluded_count} OCR-dependent question/answer pairs)")
    if low_consensus_excluded_count:
        print(
            f"  (excluded {low_consensus_excluded_count} pairs with "
            f"answer_consensus < {min_consensus})"
        )
    if duplicate_count:
        print(f"  (excluded {duplicate_count} duplicate rows)")
    if not_directly_visual_count:
        print(
            f"  (excluded {not_directly_visual_count} NOT_DIRECTLY_VISUAL questions)"
        )
    for rule, count in rule_counts.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {rule}: {count} ({pct:.1f}%)")

    print("\nSample captions:")
    for row in rows[:5]:
        print(f"  Q: {row['question']}")
        print(f"  A: {row['answer']}")
        print(f"  C: {row['caption']}  [{row['rule']}]")
        print()

    llm_rows = [r for r in rows if r["rule"] == "llm_fallback"][:3]
    if llm_rows:
        print("Sample llm_fallback captions:")
        for row in llm_rows:
            print(f"  Q: {row['question']}")
            print(f"  A: {row['answer']}")
            print(f"  C: {row['caption']}  [{row['rule']}]")
            print()

    still = sum(1 for r in rows if r["rule"] == "needs_llm")
    if still:
        log_p = llm_failure_log_path(output_path)
        print(
            f"NOTE: {still} rule=needs_llm remain. "
            f"With --llm, see failure reasons in:\n  {log_p}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon (rule + optional LLM)."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate question-dependent captions from VQA v2 "
            "(rules + optional LLM). Ctrl+C safe: checkpoint + resume."
        )
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Kodom split VQA v2 (train ya val)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Override path be questions JSON",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Override path be annotations JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override path be output JSON",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Faghat N sample aval (baraye smoke test)",
    )
    parser.add_argument(
        "--min-consensus",
        type=float,
        default=0.0,
        help=(
            "Drop Q/A pairs whose answer_consensus (share of the 10 VQA "
            "annotators giving the mode answer) is below this, e.g. 0.4. "
            "Default 0.0 = keep everything. Dropped pairs go to "
            "*_low_consensus.json"
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Baraye rule=needs_llm az Ollama/Mistral caption begir",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Chand Q+A toye yek LLM request (default 10; prefer <=10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Esm model Ollama (default mistral)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API request be hamoon Ollama (default 1, 8GB safe)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Base URL Ollama",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Har chand LLM batch output JSON save beshe "
            "(1=har batch, 50 ya 100=kamtar I/O; default 1)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore classifier + LLM checkpoints and start fresh",
    )
    parser.add_argument(
        "--classifier-checkpoint-every",
        type=int,
        default=50,
        help=(
            "Save classifier progress every N classified questions "
            "(default 50; enables resume after interrupt). Final result is "
            "kept as vqa_v2_question_classification_result[.json]. "
            "The binary DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL classifier "
            "always runs."
        ),
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help="Ollama model for question classifier (default: same as --model)",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=10,
        help=(
            "Pack N questions into one classifier Ollama call "
            "(JSON array of labels; default 10)"
        ),
    )
    parser.add_argument(
        "--no-fast-path",
        action="store_true",
        help=(
            "Disable the Fast Path whitelist: send every question to the LLM "
            "classifier (slower; use to measure Fast Path false positives)"
        ),
    )
    parser.add_argument(
        "--drop-subjective-candidates",
        action="store_true",
        help=(
            "Deprecated no-op: the binary classifier always runs. "
            "Previously dropped regex non-visual candidates without Qwen."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — classifier (always) + rule caption + optional LLM fallback + resume."""
    args = parse_args()
    process_started_at = now_iso()
    paths = SPLIT_PATHS[args.split]

    questions_json = Path(args.questions) if args.questions else paths["questions"]
    annotations_json = (
        Path(args.annotations) if args.annotations else paths["annotations"]
    )
    output_path = Path(args.output) if args.output else paths["output"]
    clf_result_path = classification_result_path(args.split)

    if not questions_json.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_json}")
    if not annotations_json.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_json}")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 10:
        print(
            f"WARNING: --batch-size={args.batch_size} > 10 increases "
            "cross-item contamination risk; prefer <=10."
        )
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    if args.classifier_checkpoint_every < 1:
        raise ValueError("--classifier-checkpoint-every must be >= 1")
    if args.classifier_batch_size < 1:
        raise ValueError("--classifier-batch-size must be >= 1")
    if not 0.0 <= args.min_consensus <= 1.0:
        raise ValueError("--min-consensus must be between 0.0 and 1.0")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.no_resume:
        delete_classifier_checkpoint(clf_result_path)

    rows: Optional[List[Dict[str, Any]]] = None
    rule_counts: Counter
    ocr_excluded_count = 0
    duplicate_count = 0
    input_count = 0
    directly_visual_count = 0
    not_directly_visual_count = 0
    validation_retry_count = 0
    validation_failure_count = 0
    classifier_meta: Optional[Dict[str, Any]] = None
    dropped_empty_count = 0
    dropped_not_visual: List[Dict[str, Any]] = []
    low_consensus_excluded_count = 0
    validation_flagged_count = 0
    rule_validation_reject_count = 0
    validation_meta: Optional[Dict[str, Any]] = None

    # Resume fast-path: skip load/rules/classifier when output JSON matches.
    # A different --min-consensus means the previous rows were filtered with
    # another threshold, so they must be rebuilt instead of reused.
    if args.llm and not args.no_resume:
        rows = try_load_checkpoint_rows(
            output_path,
            classification_path=clf_result_path,
        )
        prev_payload = load_output_payload(output_path) or {}
        prev_min = float((prev_payload.get("info") or {}).get("min_consensus", 0.0))
        if rows is not None and prev_min != args.min_consensus:
            print(
                f"Ignoring checkpoint: it was built with "
                f"--min-consensus {prev_min}, this run uses "
                f"{args.min_consensus} — rebuilding rows."
            )
            rows = None
        if rows is not None:
            rule_counts = recount_rules(rows)
            prev_info = prev_payload.get("info") or {}
            prev_started = prev_info.get("process_started_at")
            if isinstance(prev_started, str) and prev_started.strip():
                process_started_at = prev_started.strip()
            low_consensus_excluded_count = int(
                prev_info.get("low_consensus_excluded_count", 0)
            )
            ocr_excluded_count = int(prev_info.get("ocr_excluded_count", 0))
            duplicate_count = int(prev_info.get("duplicate_count", 0))
            input_count = int(prev_info.get("input_count", 0))
            directly_visual_count = int(prev_info.get("directly_visual_count", 0))
            not_directly_visual_count = int(
                prev_info.get("not_directly_visual_count", 0)
                or prev_info.get("subjective_excluded_count", 0)
            )
            validation_retry_count = int(prev_info.get("validation_retry_count", 0))
            validation_failure_count = int(
                prev_info.get("validation_failure_count", 0)
            )
            classifier_meta = prev_info.get("question_classifier")
            dropped_empty_count = int(prev_info.get("dropped_empty_count", 0))
            rule_validation_reject_count = int(
                prev_info.get("rule_validation_reject_count", 0)
            )
            print(
                f"Loaded checkpoint ({len(rows)} rows) az {output_path} "
                f"— rules skip, LLM az ja-monde edame."
            )

    if rows is None:
        (
            rows,
            rule_counts,
            ocr_excluded_count,
            duplicate_count,
            input_count,
            low_consensus_rows,
            rule_validation_reject_count,
        ) = load_vqa_pairs(
            questions_json,
            annotations_json,
            max_items=args.max_items,
            min_consensus=args.min_consensus,
        )
        low_consensus_excluded_count = len(low_consensus_rows)
        if low_consensus_rows:
            side = write_low_consensus_sidecar(
                output_path,
                low_consensus_rows,
                args.min_consensus,
            )
            print(f"Low-consensus sidecar -> {side}")

        if args.drop_subjective_candidates:
            print(
                "WARNING: --drop-subjective-candidates is ignored; "
                "the DIRECTLY_VISUAL classifier always runs."
            )
        clf_model = args.classifier_model or args.model
        clf = QuestionClassifier(
            host=args.ollama_host,
            model=clf_model,
        )
        classifier_meta = clf.metadata()
        classifier_meta["fast_path_enabled"] = not args.no_fast_path
        classifier_meta["batch_size"] = args.classifier_batch_size
        print(
            f"Question classifier: model={clf_model} "
            f"prompt={CLASSIFIER_PROMPT_VERSION} "
            f"batch-size={args.classifier_batch_size} "
            f"fast_path={'off' if args.no_fast_path else 'on'}"
        )
        try:
            rows, dropped_not_visual, lab_counts = filter_non_visual_questions(
                rows,
                clf,
                offline_drop_candidates=False,
                checkpoint_path=clf_result_path,
                checkpoint_every=args.classifier_checkpoint_every,
                resume=not args.no_resume,
                classifier_meta=classifier_meta,
                input_count=input_count,
                fast_path=not args.no_fast_path,
                batch_size=args.classifier_batch_size,
            )
        except KeyboardInterrupt:
            ckpt = load_classifier_checkpoint(clf_result_path)
            done = 0
            total = len(rows)
            if ckpt:
                info = ckpt.get("info") or {}
                done = int(info.get("classified_count", 0))
                total = int(info.get("total_to_classify", total))
            print(
                f"\nInterrupted during classification — "
                f"checkpoint saved -> {clf_result_path} "
                f"({done}/{total} done). Rerun the same command to continue."
            )
            raise SystemExit(130) from None
        not_directly_visual_count = len(dropped_not_visual)
        directly_visual_count = len(rows)
        rule_counts = recount_rules(rows)
        classifier_meta = dict(classifier_meta)
        classifier_meta["label_counts"] = dict(lab_counts)
        side = write_not_directly_visual_sidecar(output_path, dropped_not_visual)
        print(
            f"Classifier filter: kept {directly_visual_count} DIRECTLY_VISUAL, "
            f"dropped {not_directly_visual_count} NOT_DIRECTLY_VISUAL; "
            f"sidecar -> {side}; "
            f"classification result -> {clf_result_path}; "
            f"label_counts={dict(lab_counts)}"
        )

    llm_meta: Optional[Dict[str, Any]] = None
    failure_log: Optional[LlmFailureLogger] = None
    retry_audit: Optional[RetryAuditLogger] = None
    validation_config = ValidationConfig()
    ollama_client: Optional[OllamaClient] = None
    if args.llm:
        log_path = llm_failure_log_path(output_path)
        failure_log = LlmFailureLogger(log_path)
        audit_path = retry_audit_path(output_path)
        retry_audit = RetryAuditLogger(audit_path)
        print(f"LLM failure log -> {log_path}")
        print(f"Retry audit log -> {audit_path}")
        llm_meta = {
            "model": args.model,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "host": args.ollama_host,
            "prompt_version": PROMPT_VERSION,
            "num_ctx": 4096,
            "failure_log": str(log_path.resolve()),
            "retry_audit_log": str(audit_path.resolve()),
            "validation": {
                "single_retries": 1,
                "salvage_single_retries": 1,
                "tier": "fast_three_class+batch_llm_judge",
                "validator_version": VALIDATOR_VERSION,
                "overlap_fail_threshold": validation_config.overlap_fail_threshold,
                "overlap_pass_threshold": validation_config.overlap_pass_threshold,
                "min_words": validation_config.min_words,
                "max_words": validation_config.max_words,
            },
        }
        if args.no_resume:
            print("--no-resume: checkpoint llm_fallback merge nemishe")
            resume_map: Dict[int, Dict[str, Any]] = {}
        else:
            resume_map = load_existing_llm_map(output_path)

        ollama_client = OllamaClient(
            host=args.ollama_host,
            model=args.model,
            num_ctx=4096,
            validation_config=validation_config,
        )
        try:
            rule_counts, validation_retry_count, validation_failure_count = (
                apply_llm_fallbacks(
                    rows,
                    client=ollama_client,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    checkpoint_every=args.checkpoint_every,
                    output_path=output_path,
                    questions_json=questions_json,
                    annotations_json=annotations_json,
                    llm_meta=llm_meta,
                    resume_map=resume_map,
                    failure_log=failure_log,
                    retry_audit=retry_audit,
                    single_retries=1,
                    ocr_excluded_count=ocr_excluded_count,
                    dropped_empty_count=dropped_empty_count,
                    duplicate_count=duplicate_count,
                    input_count=input_count,
                    directly_visual_count=directly_visual_count,
                    not_directly_visual_count=not_directly_visual_count,
                    validation_retry_count=validation_retry_count,
                    validation_failure_count=validation_failure_count,
                    classifier_meta=classifier_meta,
                    low_consensus_excluded_count=low_consensus_excluded_count,
                    min_consensus=args.min_consensus,
                    rule_validation_reject_count=rule_validation_reject_count,
                    process_started_at=process_started_at,
                )
            )
        except KeyboardInterrupt:
            print("\nInterrupted — saving checkpoint...")
            rule_counts = recount_rules(rows)
            write_output_json(
                output_path,
                rows,
                rule_counts,
                questions_json,
                annotations_json,
                llm_meta=llm_meta,
                ocr_excluded_count=ocr_excluded_count,
                dropped_empty_count=dropped_empty_count,
                duplicate_count=duplicate_count,
                input_count=input_count,
                directly_visual_count=directly_visual_count,
                not_directly_visual_count=not_directly_visual_count,
                validation_retry_count=validation_retry_count,
                validation_failure_count=validation_failure_count,
                classifier_meta=classifier_meta,
                low_consensus_excluded_count=low_consensus_excluded_count,
                min_consensus=args.min_consensus,
                rule_validation_reject_count=rule_validation_reject_count,
                process_started_at=process_started_at,
            )
            still = sum(1 for r in rows if r["rule"] == "needs_llm")
            if failure_log is not None:
                failure_log.write_summary(still)
                failure_log.close()
            print(
                f"Checkpoint saved -> {output_path} "
                f"({still} needs_llm left). Dobare hamoon command ro bezan."
            )
            raise SystemExit(130) from None
        finally:
            if failure_log is not None:
                failure_log.close()
            if retry_audit is not None:
                print(f"Retry audit: {retry_audit.summary()} -> {retry_audit.path}")
                retry_audit.close()

    # Never ship empty / needs_llm leftovers into the written dataset.
    # Keep validation_failure_count separate from dropped_empty_count so
    # accounting does not double-count.
    rows, n_dropped = drop_empty_or_short_captions(rows)
    other_empty = max(0, n_dropped - validation_failure_count)
    dropped_empty_count += other_empty
    if n_dropped:
        print(
            f"Dropped {n_dropped} rows with empty/short/needs_llm captions "
            f"(validation_failure={validation_failure_count}, "
            f"other_empty={other_empty}; "
            f"total dropped_empty_count={dropped_empty_count})."
        )

    # Same validator for every caption, rule-based or LLM (Comments8 item 7).
    val_log_path = validation_log_path(output_path)
    val_log = ValidationLogWriter(val_log_path)
    print(f"Validation log -> {val_log_path}")

    final_client = ollama_client
    if final_client is None and args.llm:
        final_client = OllamaClient(
            host=args.ollama_host,
            model=args.model,
            num_ctx=4096,
            validation_config=validation_config,
        )

    rows, n_final_rejects, validation_flagged_count, _failed_rows, val_stats = (
        final_validation_pass(
            rows,
            client=final_client,
            config=validation_config,
            log_writer=val_log,
            use_llm=args.llm,
        )
    )
    val_log.close()
    failed_sidecar = val_log.write_failed_sidecar(output_path)
    if failed_sidecar:
        print(f"Validation failed sidecar -> {failed_sidecar}")

    validation_meta = {
        "validator_version": VALIDATOR_VERSION,
        **validation_config.__dict__,
        **val_stats.to_dict(),
        "validation_log": str(val_log_path.resolve()),
    }
    if failed_sidecar:
        validation_meta["validation_failed_sidecar"] = str(failed_sidecar.resolve())

    if n_final_rejects:
        validation_failure_count += n_final_rejects
        print(
            f"Final validation: dropped {n_final_rejects} captions that failed "
            "a hard check."
        )
    if validation_flagged_count:
        print(
            f"Final validation: {validation_flagged_count} captions kept with "
            "validation_flags for review."
        )
    rule_counts = recount_rules(rows)

    write_output_json(
        output_path,
        rows,
        rule_counts,
        questions_json,
        annotations_json,
        llm_meta=llm_meta,
        ocr_excluded_count=ocr_excluded_count,
        dropped_empty_count=dropped_empty_count,
        duplicate_count=duplicate_count,
        input_count=input_count,
        directly_visual_count=directly_visual_count,
        not_directly_visual_count=not_directly_visual_count,
        validation_retry_count=validation_retry_count,
        validation_failure_count=validation_failure_count,
        classifier_meta=classifier_meta,
        low_consensus_excluded_count=low_consensus_excluded_count,
        min_consensus=args.min_consensus,
        validation_flagged_count=validation_flagged_count,
        rule_validation_reject_count=rule_validation_reject_count,
        validation_meta=validation_meta,
        process_started_at=process_started_at,
    )
    print_stats(
        rows,
        rule_counts,
        output_path,
        ocr_excluded_count=ocr_excluded_count,
        duplicate_count=duplicate_count,
        not_directly_visual_count=not_directly_visual_count,
        low_consensus_excluded_count=low_consensus_excluded_count,
        min_consensus=args.min_consensus,
        process_started_at=process_started_at,
    )
    if args.llm:
        still = sum(1 for r in rows if r["rule"] == "needs_llm")
        if still:
            print(
                f"ERROR: {still} needs_llm remain after --llm. "
                f"Inspect {llm_failure_log_path(output_path)}"
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
