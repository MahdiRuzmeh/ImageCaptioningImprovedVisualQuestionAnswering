"""Re-run the LLM question classifier on a finished question-dependent captions JSON.

Why: the first classification run kept every non-blacklisted question as
DIRECTLY_VISUAL because Ollama could not be reached (every call failed and
fell into PARSE_FAIL_KEEP). This script asks the LLM again, with the SAME
prompt and few-shots (``question_classifier._build_messages``), for every row
already in the captions file, and splits the file:

  * DIRECTLY_VISUAL      -> stays in the captions JSON (caption untouched)
  * NOT_DIRECTLY_VISUAL  -> removed, written to ``*_not_directly_visual_reclassified.json``

Unlike the first run, a failed request is NEVER counted as a label:

  * connection / HTTP / timeout errors are retried; if they still fail the
    row stays unclassified and is retried on the next run (resume);
  * the run checks Ollama before it starts and stops if requests keep
    failing (so a dead server can't silently "classify" everything);
  * only a real reply the parser can't read counts as PARSE_FAIL_KEEP (kept
    as visual, same policy as before), and it is counted separately.

Progress is appended to a small JSONL file after every batch, so Ctrl+C (or
the Kaggle runner's time limit) loses at most one batch. Re-run the same
command to continue. The output JSONs are written only once every row is
classified.

When ``--classification-result`` is given, the classifier checkpoint
(``vqa_v2_question_classification_result.json``) is updated too: removed rows
move from ``kept`` to ``dropped`` and ``post_filter_count`` is lowered. Without
that, a later ``generate.py --llm`` would see a row-count mismatch and rebuild
every row from scratch.

Examples::

    python reclassify_captions.py \
        --input outputs/vqa_v2_question_dependent_captions_train2014.json \
        --classification-result outputs/vqa_v2_question_classification_result.json \
        --output-dir outputs_reclassified --llm-parallel 4 --batch-size 10

    # quick check on the first 50 rows, nothing final written:
    python reclassify_captions.py --input ... --output-dir /tmp/x --max-items 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from question_classifier import (
    CLASSIFIER_PROMPT_VERSION,
    VISUAL_FILTER_LLM,
    QuestionClassifier,
    _drop_record,
)

RECLASSIFIER_VERSION = "r1"
NDV_SUFFIX = "_not_directly_visual_reclassified.json"
PROGRESS_SUFFIX = "_reclassify_progress.jsonl"

# Details from QuestionClassifier._chat that mean "no reply from the model".
TRANSPORT_PREFIXES = ("connection_error", "http_error", "timeout", "json_error",
                      "exception")

# Status of one classified row in the progress file.
OK = "ok"
PARSE_FAIL = "parse_fail"          # model replied, parser couldn't read it


# ---------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp.json",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_transport_error(detail: str) -> bool:
    return detail.startswith(TRANSPORT_PREFIXES)


def chunks(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------- progress
def load_progress(path: Path) -> Dict[int, Dict[str, Any]]:
    """question_id -> record. A torn last line (killed mid-write) is skipped."""
    done: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[int(rec["question_id"])] = rec
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def check_progress_header(path: Path, meta: Dict[str, Any]) -> None:
    """The first line records model + prompt; refuse to mix two setups."""
    if not path.is_file() or path.stat().st_size == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"header": meta}, ensure_ascii=False) + "\n")
        return
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    try:
        old = json.loads(first).get("header") or {}
    except (json.JSONDecodeError, AttributeError):
        old = {}
    for key in ("model", "prompt_version", "input_name"):
        if old.get(key) != meta.get(key):
            raise SystemExit(
                f"Progress file {path} was made with {key}={old.get(key)!r}, "
                f"this run uses {meta.get(key)!r}. Use another --output-dir "
                f"or delete the progress file to start over.")


# ---------------------------------------------------------------- classify
def classify_with_retry(
    clf: QuestionClassifier, question: str, retries: int, backoff_s: float,
) -> Tuple[Optional[str], Optional[str], str, str]:
    """Returns (label, non_visual_reason, status, detail).

    status is OK / PARSE_FAIL, or "transport" when the model never answered
    (label None: the caller must NOT record that row).
    """
    detail = ""
    for attempt in range(retries + 1):
        try:
            label, detail, reason = clf.classify_one(question)
        except Exception as exc:   # e.g. connection reset while reading
            label, detail, reason = None, f"exception:{type(exc).__name__}:{exc}", None
        if label is not None:
            return label, reason, OK, detail
        if not is_transport_error(detail):
            return "DIRECTLY_VISUAL", None, PARSE_FAIL, detail
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt))
    return None, None, "transport", detail


def preflight(clf: QuestionClassifier) -> None:
    """Stop at once if Ollama is unreachable or answers nonsense."""
    probes = [("What color is the bus?", "DIRECTLY_VISUAL"),
              ("What does the sign say?", "NOT_DIRECTLY_VISUAL")]
    for q, want in probes:
        label, _reason, status, detail = classify_with_retry(clf, q, 3, 2.0)
        if status == "transport":
            raise SystemExit(f"Preflight: Ollama at {clf.host} not reachable "
                             f"({detail}). Start it and re-run.")
        if status == PARSE_FAIL:
            raise SystemExit(f"Preflight: model reply could not be parsed "
                             f"({detail}). Check --model.")
        print(f"Preflight: {q!r} -> {label} (expected {want})", flush=True)


def run_classification(
    rows: List[Dict[str, Any]], clf: QuestionClassifier,
    progress_path: Path, done: Dict[int, Dict[str, Any]], args: argparse.Namespace,
) -> int:
    """Classify rows not yet in ``done``; append to progress. Returns #pending left."""
    todo = [r for r in rows if int(r["question_id"]) not in done]
    total, n_todo = len(rows), len(todo)
    print(f"Rows: {total}, already classified: {total - n_todo}, "
          f"to classify now: {n_todo} (batch-size={args.batch_size}, "
          f"llm-parallel={args.llm_parallel})", flush=True)
    if not todo:
        return 0

    counts: Counter = Counter()
    transport_left = 0
    fail_streak = 0                 # consecutive batches with only transport failures
    started = time.perf_counter()
    classified_now = 0
    last_print = 0.0

    # Each question is its own Ollama request (as in classify_batch). A window
    # of 2*parallel requests stays in flight so both GPUs are always busy;
    # results are taken in input order and committed batch-size at a time.
    window = max(1, 2 * args.llm_parallel)
    pool = ThreadPoolExecutor(max_workers=args.llm_parallel)
    inflight: Deque[Tuple[Dict[str, Any], Future]] = deque()
    it = iter(todo)

    def submit_more() -> None:
        while len(inflight) < window:
            row = next(it, None)
            if row is None:
                return
            inflight.append((row, pool.submit(
                classify_with_retry, clf, str(row.get("question") or ""),
                args.retries, args.retry_backoff)))

    progress = open(progress_path, "a", encoding="utf-8")
    try:
        submit_more()
        while inflight:
            batch: List[Tuple[Dict[str, Any], Tuple]] = []
            while inflight and len(batch) < args.batch_size:
                row, fut = inflight.popleft()
                batch.append((row, fut.result()))
                submit_more()

            lines, batch_transport = [], 0
            for row, (label, reason, status, detail) in batch:
                if status == "transport":
                    batch_transport += 1
                    continue
                rec = {"question_id": int(row["question_id"]), "label": label,
                       "non_visual_reason": reason, "status": status}
                if status == PARSE_FAIL:
                    rec["detail"] = detail[:200]
                lines.append(json.dumps(rec, ensure_ascii=False))
                done[rec["question_id"]] = rec
                counts[reason or label] += 1
                if status == PARSE_FAIL:
                    counts["PARSE_FAIL_KEEP"] += 1
            if lines:
                progress.write("\n".join(lines) + "\n")
                progress.flush()
            classified_now += len(lines)
            transport_left += batch_transport

            if batch_transport == len(batch):
                fail_streak += 1
                if fail_streak >= args.max_failed_batches:
                    raise SystemExit(
                        f"Stopping: {fail_streak} batches in a row got no reply "
                        f"from Ollama (last error: {batch[-1][1][3]}). Progress "
                        f"is saved; fix Ollama and re-run.")
            else:
                fail_streak = 0

            el = time.perf_counter() - started
            if el - last_print >= args.log_every_s or not inflight:
                last_print = el
                rate = classified_now / el if el > 0 else 0.0
                left = n_todo - classified_now - transport_left
                eta = left / rate / 3600 if rate > 0 else float("inf")
                print(f"  {total - n_todo + classified_now}/{total} classified "
                      f"| {rate:.2f} q/s | ETA {eta:.2f} h | no-reply {transport_left} "
                      f"| {dict(counts)}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: progress saved up to the last finished batch. "
              "Re-run the same command to continue.", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        progress.close()
        raise SystemExit(130) from None
    finally:
        if not progress.closed:
            progress.close()
    pool.shutdown(wait=True)
    pending = sum(1 for r in rows if int(r["question_id"]) not in done)
    print(f"Classified this run: {classified_now} in "
          f"{(time.perf_counter() - started) / 3600:.2f} h; unclassified "
          f"(no reply, retry next run): {pending}", flush=True)
    return pending


# ---------------------------------------------------------------- outputs
def write_outputs(
    data: Dict[str, Any], done: Dict[int, Dict[str, Any]], args: argparse.Namespace,
    meta: Dict[str, Any], started_at: str,
) -> None:
    rows: List[Dict[str, Any]] = data["annotations"]
    kept, dropped = [], []
    label_counts: Counter = Counter()
    for row in rows:
        rec = done[int(row["question_id"])]
        label_counts[rec["label"]] += 1
        if rec.get("non_visual_reason"):
            label_counts[rec["non_visual_reason"]] += 1
        if rec.get("status") == PARSE_FAIL:
            label_counts["PARSE_FAIL_KEEP"] += 1
        if rec["label"] == "DIRECTLY_VISUAL":
            kept.append(row)
        else:
            d = _drop_record(row, "NOT_DIRECTLY_VISUAL", "llm_reclassify",
                             VISUAL_FILTER_LLM,
                             non_visual_reason=rec.get("non_visual_reason"))
            d["caption"] = row.get("caption")
            d["rule"] = row.get("rule")
            dropped.append(d)

    out_dir = Path(args.output_dir)
    in_path = Path(args.input)
    captions_out = out_dir / in_path.name
    ndv_out = out_dir / (in_path.stem + NDV_SUFFIX)
    reclass_meta = dict(meta, started_at=started_at, ended_at=now_iso(),
                        rows_before=len(rows), rows_after=len(kept),
                        label_counts=dict(label_counts),
                        not_directly_visual_file=ndv_out.name)

    # ---- captions JSON (same structure, fewer rows)
    info = dict(data.get("info") or {})
    old_ndv = int(info.get("not_directly_visual_count", 0))
    info.update(num_samples=len(kept), post_filter_count=len(kept),
                directly_visual_count=len(kept),
                not_directly_visual_count=old_ndv + len(dropped),
                rule_counts=dict(Counter(r.get("rule") for r in kept)),
                question_reclassifier=reclass_meta)
    if captions_out.resolve() == in_path.resolve():
        backup = in_path.with_name(in_path.stem + ".before_reclassify.json")
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"Backup of the input -> {backup}", flush=True)
    write_json_atomic(captions_out, {**data, "info": info, "annotations": kept})
    print(f"Captions (DIRECTLY_VISUAL): {len(kept)} rows -> {captions_out}", flush=True)

    # ---- NOT_DIRECTLY_VISUAL items
    write_json_atomic(ndv_out, {
        "info": {
            "description": "Questions removed as NOT_DIRECTLY_VISUAL by the LLM "
                           "re-classification of the captions file (captioner-"
                           "training filter only; do not alter raw VQA2 eval)",
            "source_output": str(in_path),
            "num_samples": len(dropped),
            "reason_counts": dict(Counter(d.get("non_visual_reason") for d in dropped)),
            "question_reclassifier": reclass_meta,
        },
        "annotations": dropped,
    })
    print(f"NOT_DIRECTLY_VISUAL: {len(dropped)} rows -> {ndv_out}", flush=True)

    # ---- classifier checkpoint, so generate.py keeps resuming from the captions
    if args.classification_result:
        cr_in = Path(args.classification_result)
        cr = load_json(cr_in)
        drop_ids = {int(d["question_id"]) for d in dropped}
        cr_kept = [r for r in cr.get("kept") or [] if int(r["question_id"]) not in drop_ids]
        cr_info = dict(cr.get("info") or {})
        lc = dict(cr_info.get("label_counts") or {})
        lc["DIRECTLY_VISUAL"] = len(cr_kept)
        lc["NOT_DIRECTLY_VISUAL"] = int(lc.get("NOT_DIRECTLY_VISUAL", 0)) + len(dropped)
        lc["LLM_RECLASSIFY_NDV"] = len(dropped)
        cr_info.update(post_filter_count=len(cr_kept), label_counts=lc,
                       question_reclassifier=reclass_meta)
        cr_out = out_dir / cr_in.name
        write_json_atomic(cr_out, {**cr, "info": cr_info, "kept": cr_kept,
                                   "dropped": list(cr.get("dropped") or []) + dropped})
        print(f"Classifier checkpoint: {len(cr_kept)} kept -> {cr_out}", flush=True)
        if len(cr_kept) != len(kept):
            print(f"WARNING: checkpoint kept ({len(cr_kept)}) != captions rows "
                  f"({len(kept)}); generate.py would not resume from this file.",
                  flush=True)
    print(f"Label counts: {dict(label_counts)}", flush=True)


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True,
                    help="vqa_v2_question_dependent_captions_<split>2014.json")
    ap.add_argument("--classification-result", default=None,
                    help="vqa_v2_question_classification_result.json to update "
                         "(recommended, keeps generate.py resume working)")
    ap.add_argument("--output-dir", required=True,
                    help="where the new files and the progress JSONL go "
                         "(may equal the input folder: the input is backed up)")
    ap.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--llm-parallel", type=int, default=4,
                    help="concurrent Ollama requests")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="questions committed to the progress file at once")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="seconds per Ollama request")
    ap.add_argument("--retries", type=int, default=3,
                    help="retries per question on connection/HTTP/timeout errors")
    ap.add_argument("--retry-backoff", type=float, default=2.0,
                    help="seconds before the first retry (doubles each time)")
    ap.add_argument("--max-failed-batches", type=int, default=5,
                    help="stop after this many batches in a row with no reply")
    ap.add_argument("--log-every-s", type=float, default=60.0)
    ap.add_argument("--max-items", type=int, default=None,
                    help="only the first N rows (smoke test; outputs not written)")
    ap.add_argument("--no-preflight", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started_at = now_iso()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / (in_path.stem + PROGRESS_SUFFIX)

    print(f"Loading {in_path} ...", flush=True)
    data = load_json(in_path)
    rows: List[Dict[str, Any]] = data["annotations"]
    if args.max_items:
        rows = rows[:args.max_items]

    clf = QuestionClassifier(host=args.ollama_host, model=args.model,
                             timeout_s=args.timeout, parallel=args.llm_parallel)
    meta = {"version": RECLASSIFIER_VERSION, "model": args.model,
            "host": args.ollama_host, "prompt_version": CLASSIFIER_PROMPT_VERSION,
            "parallel": args.llm_parallel, "batch_size": args.batch_size,
            "input_name": in_path.name, "blacklist_drop_enabled": False}
    check_progress_header(progress_path, meta)
    done = load_progress(progress_path)
    print(f"Progress file: {progress_path} ({len(done)} rows already done)", flush=True)

    if any(int(r["question_id"]) not in done for r in rows):
        if not args.no_preflight:
            preflight(clf)
        pending = run_classification(rows, clf, progress_path, done, args)
        if pending:
            print(f"{pending} rows still unclassified -> outputs NOT written. "
                  f"Re-run to continue.", flush=True)
            sys.exit(3)

    if args.max_items:
        c = Counter(done[int(r["question_id"])]["non_visual_reason"]
                    or done[int(r["question_id"])]["label"] for r in rows)
        print(f"--max-items: done, outputs not written. Counts: {dict(c)}", flush=True)
        return

    # Finishing the files takes a minute; don't let a late Ctrl+C cut it short
    # (writes are atomic anyway, and progress is complete, so a re-run redoes it).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    write_outputs(data, done, args, meta, started_at)
    print("Reclassification complete.", flush=True)


if __name__ == "__main__":
    main()
