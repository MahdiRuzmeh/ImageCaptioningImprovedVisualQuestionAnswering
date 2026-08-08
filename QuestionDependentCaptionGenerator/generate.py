"""CLI baraye generate kardan-e question-dependent captions az VQA v2.

Run az in folder:

    python generate.py --split train
    python generate.py --split val
    python generate.py --split train --max-items 1000   # smoke test
    python generate.py --split val --llm --batch-size 10 \\
        --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 50

Output default: ./outputs/v2_question_dependent_captions_{train,val}2014.json

Resume (Ctrl+C bad):
    hamoon command ro dobare bezan — az checkpoint edame mide.
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
from llm_client import ItemOutcome, OllamaClient, run_batches_concurrent
from llm_prompts import PROMPT_VERSION
from question_classifier import (
    CLASSIFIER_PROMPT_VERSION,
    QuestionClassifier,
    filter_non_visual_questions,
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


def try_load_checkpoint_rows(
    output_path: Path,
    expected_n: int,
) -> Optional[List[Dict[str, Any]]]:
    """Age checkpoint size == expected, rows ro az file bardar (resume sari)."""
    data = load_output_payload(output_path)
    if not data:
        return None
    rows = data.get("annotations")
    if not isinstance(rows, list) or len(rows) != expected_n:
        return None
    # Minimal field check
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
) -> Tuple[List[Dict[str, Any]], Counter, int]:
    """Soal va javab haye VQA v2 ro load kon va ba rule caption besaz.

    Args:
        questions_json: path be v2_OpenEnded_*_questions.json
        annotations_json: path be v2_mscoco_*_annotations.json
        max_items: age set shode, faghat N sample aval (smoke test)

    Returns:
        (rows, rule_counts, ocr_excluded_count) — rows + statistik rule ha +
        chand ta item OCR-dependent bood ke kollan hazf shod (nemiyad toye
        rows, chon SimpleImageCaptioner OCR nadare — see ``is_ocr_question``).

    Skips duplicate rows: VQA v2 sometimes has two distinct question_ids for
    the same image with the exact same question text and answer (independent
    annotators asked the same thing) — only the first (lowest question_id)
    occurrence per image is kept.

    Skips OCR-dependent rows entirely (``is_ocr_question``): questions that
    can only be answered by reading rendered text/digits (signs, logos,
    brands, plates, jersey numbers, clock faces) are dropped before caption
    generation, since the downstream captioner has no way to learn them.
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

    return rows, rule_counts, ocr_excluded


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
    single_retries: int = 3,
    final_retries: int = 1,
    ocr_excluded_count: int = 0,
    dropped_empty_count: int = 0,
    subjective_excluded_count: int = 0,
    classifier_ocr_excluded_count: int = 0,
    classifier_meta: Optional[Dict[str, Any]] = None,
) -> Counter:
    """Row haye rule=needs_llm ro ba packed LLM caption update mikone.

    Resume: age question_id toye resume_map bashe, LLM call nemikone.
    Checkpoint: har N batch (default 1), JSON ro atomic save mikone.
    Failures: written to ``failure_log`` (why LLM could not be used).
    Final pass: leftover needs_llm rows get **batched** salvage (default 1 round),
    not slow per-item retries.

    Args:
        rows: annotation rows (mutated in place)
        client: Ollama client
        batch_size: Q+A per packed prompt
        workers: concurrent HTTP workers
        checkpoint_every: save JSON every N batches
        output_path: captions JSON path
        questions_json: source questions path (metadata)
        annotations_json: source annotations path (metadata)
        llm_meta: stored under info.llm
        resume_map: prior llm_fallback captions by question_id
        failure_log: optional logger for rejected / failed items
        single_retries: per-item retries inside each main-pass batch
        final_retries: number of **batched** salvage rounds for leftovers
            (default 1; each round uses packed batches with no per-item retry)
        ocr_excluded_count: forwarded to checkpoint writes (info.ocr_excluded_count)
        dropped_empty_count: forwarded to checkpoint writes
        subjective_excluded_count: forwarded to checkpoint writes
        classifier_ocr_excluded_count: forwarded to checkpoint writes
        classifier_meta: forwarded to checkpoint writes
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
        return recount_rules(rows)

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
            subjective_excluded_count=subjective_excluded_count,
            classifier_ocr_excluded_count=classifier_ocr_excluded_count,
            classifier_meta=classifier_meta,
        )

    def _on_batch_start(batch_i: int, batch_len: int) -> None:
        print(
            f"  LLM batch {batch_i + 1}/{total_batches} calling Ollama "
            f"({batch_len} Q+A)...",
            flush=True,
        )

    def _on_batch(batch_i: int, outcomes: List[ItemOutcome]) -> None:
        nonlocal done_batches
        ok_n = 0
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
        # Batched salvage: one packed attempt per leftover batch (no slow
        # per-item single retries). final_retries = number of salvage rounds.
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
                    ok_n = 0
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
                # One packed call only — no per-item single retries
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

    return recount_rules(rows)


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
    subjective_excluded_count: int = 0,
    classifier_ocr_excluded_count: int = 0,
    classifier_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Natije ro atomic be JSON file save kon (crash-safe).

    Args:
        ocr_excluded_count: chand ta OCR-dependent Q/A pair kollan hazf shod
            ghabl az caption generation (see ``is_ocr_question``); stored
            under ``info.ocr_excluded_count`` baraye visibility.
        dropped_empty_count: rows removed for empty/short/needs_llm captions.
        subjective_excluded_count: non-visual questions dropped by classifier.
        classifier_ocr_excluded_count: OCR labeled by subjective classifier stage.
        classifier_meta: optional reproducibility info for the Qwen classifier.
    """
    info: Dict[str, Any] = {
        "description": "VQA v2 question-dependent captions (rule-based Q+A → statement)",
        "source_questions": str(questions_json),
        "source_annotations": str(annotations_json),
        "num_samples": len(rows),
        "ocr_excluded_count": ocr_excluded_count,
        "dropped_empty_count": dropped_empty_count,
        "subjective_excluded_count": subjective_excluded_count,
        "classifier_ocr_excluded_count": classifier_ocr_excluded_count,
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
    # Atomic write: tmp file bad rename — file nime-kar corrupt nashe
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
) -> None:
    """Statistik rule ha ro chap kon ta befahmim cheghadr needs_llm darim."""
    total = len(rows)
    print(f"Wrote {total} captions -> {output_path}")
    if ocr_excluded_count:
        print(f"  (excluded {ocr_excluded_count} OCR-dependent question/answer pairs)")
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
        help="Checkpoint ghabli ro ignore kon (az aval LLM)",
    )
    parser.add_argument(
        "--classify-questions",
        action="store_true",
        help=(
            "Run two-stage subjective/OCR classifier on keyword candidates "
            "and drop non-VISUAL questions"
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
            "Without calling the classifier, drop all regex subjective "
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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    expected_n = count_vqa_overlap(
        questions_json, annotations_json, max_items=args.max_items
    )

    rows: Optional[List[Dict[str, Any]]] = None
    rule_counts: Counter
    ocr_excluded_count = 0
    subjective_excluded_count = 0
    classifier_ocr_excluded_count = 0
    classifier_meta: Optional[Dict[str, Any]] = None
    dropped_empty_count = 0

    # Resume sari: age output size match bashe, dobare rule run nakon
    if args.llm and not args.no_resume:
        rows = try_load_checkpoint_rows(output_path, expected_n)
        if rows is not None:
            rule_counts = recount_rules(rows)
            prev_payload = load_output_payload(output_path) or {}
            prev_info = prev_payload.get("info") or {}
            ocr_excluded_count = int(prev_info.get("ocr_excluded_count", 0))
            subjective_excluded_count = int(
                prev_info.get("subjective_excluded_count", 0)
            )
            classifier_ocr_excluded_count = int(
                prev_info.get("classifier_ocr_excluded_count", 0)
            )
            classifier_meta = prev_info.get("question_classifier")
            dropped_empty_count = int(prev_info.get("dropped_empty_count", 0))
            print(
                f"Loaded checkpoint ({len(rows)} rows) az {output_path} "
                f"— rules skip, LLM az ja-monde edame."
            )

    if rows is None:
        rows, rule_counts, ocr_excluded_count = load_vqa_pairs(
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
            rows, subjective_excluded_count, classifier_ocr_excluded_count, lab_counts = (
                filter_non_visual_questions(
                    rows,
                    clf,
                    offline_drop_candidates=args.drop_subjective_candidates
                    and not args.classify_questions,
                )
            )
            rule_counts = recount_rules(rows)
            print(
                f"Subjective filter: dropped {subjective_excluded_count} "
                f"non-visual + {classifier_ocr_excluded_count} classifier-OCR; "
                f"label_counts={dict(lab_counts)}"
            )

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
        }
        # Merge llm_fallback az file (age rules-rebuild shode bashe).
        # Age rows mostaghim az checkpoint load shode, merge no-op safe hast.
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
            rule_counts = apply_llm_fallbacks(
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
                ocr_excluded_count=ocr_excluded_count,
                dropped_empty_count=dropped_empty_count,
                subjective_excluded_count=subjective_excluded_count,
                classifier_ocr_excluded_count=classifier_ocr_excluded_count,
                classifier_meta=classifier_meta,
            )
        except KeyboardInterrupt:
            # Ctrl+C: last state ro save kon ta resume beshe
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
                subjective_excluded_count=subjective_excluded_count,
                classifier_ocr_excluded_count=classifier_ocr_excluded_count,
                classifier_meta=classifier_meta,
            )
            still = sum(1 for r in rows if r["rule"] == "needs_llm")
            if failure_log is not None:
                # Log whatever we know so far (outcomes may be partial)
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

    # Never ship empty / needs_llm leftovers into the written dataset
    rows, n_dropped = drop_empty_or_short_captions(rows)
    dropped_empty_count += n_dropped
    if n_dropped:
        print(
            f"Dropped {n_dropped} rows with empty/short/needs_llm captions "
            f"(total dropped_empty_count={dropped_empty_count})."
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
        subjective_excluded_count=subjective_excluded_count,
        classifier_ocr_excluded_count=classifier_ocr_excluded_count,
        classifier_meta=classifier_meta,
    )
    print_stats(rows, rule_counts, output_path, ocr_excluded_count=ocr_excluded_count)
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
