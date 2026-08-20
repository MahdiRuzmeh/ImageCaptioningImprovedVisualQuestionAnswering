"""CLI baraye generate kardan-e question-dependent captions az VQA v2.

Run az in folder:

    python generate.py --split train
    python generate.py --split val
    python generate.py --split train --max-items 1000   # smoke test
    python generate.py --split val --llm --batch-size 10 \\
        --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 50

Output default: ./outputs/v2_question_dependent_captions_{train,val}2014.json

Resume (Ctrl+C safe):
    hamoon command ro dobare bezan — classifier + LLM az checkpoint edame mide.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from caption_rules import answer_mode_stats, generate_caption, is_ocr_question
from llm_client import (
    ItemOutcome,
    OllamaClient,
    run_batches_concurrent,
    _VALIDATION_FAIL_REASONS,
)
from llm_prompts import PROMPT_VERSION
from question_classifier import (
    CLASSIFIER_PROMPT_VERSION,
    QuestionClassifier,
    delete_classifier_checkpoint,
    filter_non_visual_questions,
    load_classifier_checkpoint,
)

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
        "output": OUTPUT_ROOT / "v2_question_dependent_captions_train2014.json",
    },
    "val": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_val2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_val2014_annotations.json",
        "output": OUTPUT_ROOT / "v2_question_dependent_captions_val2014.json",
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


def classifier_checkpoint_path(output_path: Path) -> Path:
    """Sidecar JSON for incremental classifier resume."""
    return output_path.with_name(output_path.stem + "_classifier_checkpoint.json")


def resolve_post_filter_row_count(output_path: Path) -> Optional[int]:
    """Expected annotation count after OCR/dedup/classifier filters."""
    ckpt = load_classifier_checkpoint(classifier_checkpoint_path(output_path))
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
) -> Optional[List[Dict[str, Any]]]:
    """Load output rows when count matches post-filter expectation (resume fast-path)."""
    data = load_output_payload(output_path)
    if not data:
        return None
    rows = data.get("annotations")
    if not isinstance(rows, list) or not rows:
        return None
    expected = resolve_post_filter_row_count(output_path)
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
) -> Tuple[List[Dict[str, Any]], Counter, int, int, int]:
    """Soal va javab haye VQA v2 ro load kon va ba rule caption besaz.

    Args:
        questions_json: path be v2_OpenEnded_*_questions.json
        annotations_json: path be v2_mscoco_*_annotations.json
        max_items: age set shode, faghat N sample aval (smoke test)

    Returns:
        (rows, rule_counts, ocr_excluded_count, duplicate_count, input_count)
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
    duplicates_dropped = 0
    ocr_excluded = 0
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

        dedup_key = (q["question"].strip().lower(), ans)
        seen = seen_per_image.setdefault(image_id, set())
        if dedup_key in seen:
            duplicates_dropped += 1
            continue
        seen.add(dedup_key)

        caption, rule = generate_caption(q["question"], ans)
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
    if duplicates_dropped:
        print(
            f"Dedup: {duplicates_dropped} duplicate (image_id, question, answer) "
            "rows dropped — kept the first occurrence of each."
        )
    print(
        f"Rule pass done: {len(rows)} rows "
        f"({rule_counts.get('needs_llm', 0)} needs_llm)",
        flush=True,
    )

    return rows, rule_counts, ocr_excluded, duplicates_dropped, input_count


def not_directly_visual_path(output_path: Path) -> Path:
    """Sidecar JSON for questions dropped as NOT_DIRECTLY_VISUAL."""
    return output_path.with_name(output_path.stem + "_not_directly_visual.json")


def write_not_directly_visual_sidecar(
    output_path: Path,
    dropped_rows: List[Dict[str, Any]],
) -> Path:
    """Persist classifier-dropped questions for later VISUAL vs non-VISUAL analysis."""
    side = not_directly_visual_path(output_path)
    payload = {
        "info": {
            "description": (
                "Questions dropped as NOT_DIRECTLY_VISUAL "
                "(captioner-training filter only; do not alter raw VQA2 eval)"
            ),
            "source_output": str(output_path.resolve()),
            "num_samples": len(dropped_rows),
        },
        "annotations": dropped_rows,
    }
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
    """Sidecar log path next to captions JSON (``*.json.llm_failures.log``)."""
    return output_path.with_suffix(output_path.suffix + ".llm_failures.log")


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
            last_outcome[row_i] = outcome
            if outcome.caption is None:
                continue
            rows[row_i]["caption"] = outcome.caption
            rows[row_i]["rule"] = "llm_fallback"
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
                f"(batch-size={batch_size}, no per-item retry)",
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
                        last_outcome[row_i] = outcome
                        if outcome.caption is None:
                            continue
                        rows[row_i]["caption"] = outcome.caption
                        rows[row_i]["rule"] = "llm_fallback"
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
            run_batches_concurrent(
                client,
                salvage_pairs,
                workers=workers,
                on_batch_done=s_done,
                on_batch_start=s_start,
                single_retries=0,
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


def drop_empty_or_short_captions(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove rows with empty, whitespace-only, or <2-word captions.

    Also drops leftover ``needs_llm`` rows so they never reach a DataLoader.
    Returns (kept_rows, dropped_count).
    """
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        cap = str(row.get("caption") or "").strip()
        rule = str(row.get("rule") or "")
        if rule == "needs_llm" or not cap or len(cap.split()) < 2:
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
) -> None:
    """Natije ro atomic be JSON file save kon (crash-safe)."""
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
        "dropped_empty_count": dropped_empty_count,
        "validation_retry_count": validation_retry_count,
        "validation_failure_count": validation_failure_count,
        "rule_counts": dict(rule_counts),
    }
    if classifier_meta:
        info["question_classifier"] = classifier_meta
    if llm_meta:
        info["description"] = (
            "VQA v2 question-dependent captions (rules + optional LLM fallback)"
        )
        info["llm"] = llm_meta

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
) -> None:
    """Statistik rule ha ro chap kon ta befahmim cheghadr needs_llm darim."""
    total = len(rows)
    print(f"Wrote {total} captions -> {output_path}")
    if ocr_excluded_count:
        print(f"  (excluded {ocr_excluded_count} OCR-dependent question/answer pairs)")
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
            "(default 50; enables resume after interrupt)"
        ),
    )
    parser.add_argument(
        "--classify-questions",
        action="store_true",
        help=(
            "Run binary DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL classifier "
            "on every question and drop NOT_DIRECTLY_VISUAL "
            "(writes *_not_directly_visual.json sidecar)"
        ),
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help="Ollama model for question classifier (default: same as --model)",
    )
    parser.add_argument(
        "--drop-subjective-candidates",
        action="store_true",
        help=(
            "Without calling the classifier, drop regex non-visual "
            "candidates (offline conservative mode)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — rule caption + optional LLM fallback + resume."""
    args = parse_args()
    paths = SPLIT_PATHS[args.split]

    questions_json = Path(args.questions) if args.questions else paths["questions"]
    annotations_json = (
        Path(args.annotations) if args.annotations else paths["annotations"]
    )
    output_path = Path(args.output) if args.output else paths["output"]

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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    ckpt_path = classifier_checkpoint_path(output_path)
    if args.no_resume:
        delete_classifier_checkpoint(ckpt_path)

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

    # Resume fast-path: skip load/rules/classifier when output JSON matches.
    if args.llm and not args.no_resume:
        rows = try_load_checkpoint_rows(output_path)
        if rows is not None:
            rule_counts = recount_rules(rows)
            prev_payload = load_output_payload(output_path) or {}
            prev_info = prev_payload.get("info") or {}
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
        ) = load_vqa_pairs(
            questions_json,
            annotations_json,
            max_items=args.max_items,
        )

        if args.classify_questions or args.drop_subjective_candidates:
            clf: Optional[QuestionClassifier] = None
            if args.classify_questions:
                clf_model = args.classifier_model or args.model
                clf = QuestionClassifier(
                    host=args.ollama_host,
                    model=clf_model,
                )
                classifier_meta = clf.metadata()
                print(
                    f"Question classifier: model={clf_model} "
                    f"prompt={CLASSIFIER_PROMPT_VERSION}"
                )
            try:
                rows, dropped_not_visual, lab_counts = filter_non_visual_questions(
                    rows,
                    clf,
                    offline_drop_candidates=args.drop_subjective_candidates
                    and not args.classify_questions,
                    checkpoint_path=ckpt_path if args.classify_questions else None,
                    checkpoint_every=args.classifier_checkpoint_every,
                    resume=not args.no_resume,
                    classifier_meta=classifier_meta,
                    input_count=input_count,
                )
            except KeyboardInterrupt:
                ckpt = load_classifier_checkpoint(ckpt_path)
                done = 0
                total = len(rows)
                if ckpt:
                    info = ckpt.get("info") or {}
                    done = int(info.get("classified_count", 0))
                    total = int(info.get("total_to_classify", total))
                print(
                    f"\nInterrupted during classification — "
                    f"checkpoint saved -> {ckpt_path} "
                    f"({done}/{total} done). Rerun the same command to continue."
                )
                raise SystemExit(130) from None
            not_directly_visual_count = len(dropped_not_visual)
            directly_visual_count = len(rows)
            rule_counts = recount_rules(rows)
            if classifier_meta is None:
                classifier_meta = {
                    "prompt_version": CLASSIFIER_PROMPT_VERSION,
                    "mode": "offline_candidates",
                }
            classifier_meta = dict(classifier_meta)
            classifier_meta["label_counts"] = dict(lab_counts)
            side = write_not_directly_visual_sidecar(output_path, dropped_not_visual)
            print(
                f"Classifier filter: kept {directly_visual_count} DIRECTLY_VISUAL, "
                f"dropped {not_directly_visual_count} NOT_DIRECTLY_VISUAL; "
                f"sidecar -> {side}; label_counts={dict(lab_counts)}"
            )
        else:
            directly_visual_count = len(rows)
            not_directly_visual_count = 0

    llm_meta: Optional[Dict[str, Any]] = None
    failure_log: Optional[LlmFailureLogger] = None
    if args.llm:
        log_path = llm_failure_log_path(output_path)
        failure_log = LlmFailureLogger(log_path)
        print(f"LLM failure log -> {log_path}")
        llm_meta = {
            "model": args.model,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "host": args.ollama_host,
            "prompt_version": PROMPT_VERSION,
            "num_ctx": 4096,
            "failure_log": str(log_path.resolve()),
            "validation": {
                "single_retries": 1,
                "tier": "lexical+semantic_judge",
            },
        }
        if args.no_resume:
            print("--no-resume: checkpoint llm_fallback merge nemishe")
            resume_map: Dict[int, Dict[str, Any]] = {}
        else:
            resume_map = load_existing_llm_map(output_path)

        client = OllamaClient(
            host=args.ollama_host,
            model=args.model,
            num_ctx=4096,
        )
        try:
            rule_counts, validation_retry_count, validation_failure_count = (
                apply_llm_fallbacks(
                    rows,
                    client=client,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    checkpoint_every=args.checkpoint_every,
                    output_path=output_path,
                    questions_json=questions_json,
                    annotations_json=annotations_json,
                    llm_meta=llm_meta,
                    resume_map=resume_map,
                    failure_log=failure_log,
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
    )
    delete_classifier_checkpoint(ckpt_path)
    print_stats(
        rows,
        rule_counts,
        output_path,
        ocr_excluded_count=ocr_excluded_count,
        duplicate_count=duplicate_count,
        not_directly_visual_count=not_directly_visual_count,
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
