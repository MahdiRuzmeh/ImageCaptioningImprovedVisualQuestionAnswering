"""Find WRONG captions and regenerate only those (resumable, multi-session).

For every row of a question-dependent captions JSON:

1. the LLM judge (``caption_errors.judge``) gives ``caption_quality_verdict``
   ``ok`` / ``GRAMMAR_ISSUE_ONLY`` / ``WRONG`` and a ``verdict_reason``;
2. a WRONG row is regenerated right away (``caption_errors.regenerate``),
   in packs of ``--regen-batch-size`` (default 5), each new caption checked
   against its pack (``regenerate.check_caption``: fast_validate hard
   rejects incl. batch contamination, plus spurious negation; rejects are
   redone one by one) and judged again,
   up to ``--regen-rounds`` times. ``ok`` / ``GRAMMAR_ISSUE_ONLY`` rows are
   never changed.

Progress is appended to ``<stem>_caption_check_progress.jsonl`` after every
batch, so Ctrl+C (or the Kaggle runner's time limit) loses at most the
batches in flight. Re-run the same command (same --output-dir) to continue.

On every exit (done, Ctrl+C, too many failed requests) the script writes:

* ``<output-dir>/<input name>``: the captions JSON with the new fields on
  every judged row (rows not judged yet: ``caption_quality_verdict: null``)
  and ``info.caption_error_fix.progress.complete`` true/false;
* ``<stem>_suspicious.json``: rows whose FINAL caption may still be wrong
  (``caption_errors.suspicious``: words not in question + answer, or a
  flipped yes/no), for a later pass with a bigger model. Every row of the
  captions JSON also gets ``suspicious`` / ``suspicious_reasons``;
* ``<stem>_caption_check_state.json``: small progress summary.

Exit code: 0 all rows of --scope judged, 3 some rows still unjudged, 4 stopped because
the model did not reply, 5 time limit (--deadline-unix) reached, 130 interrupted.

Examples::

    # local smoke test with the small model, first 200 rows only
    python -m caption_errors.judge_captions \
        --input outputs/vqa_v2_question_dependent_captions_train2014.json \
        --output-dir outputs_caption_check --model qwen2.5:3b-instruct-q4_K_M \
        --max-items 200

    # rebuild the captions JSON from the progress file only (no LLM calls)
    python -m caption_errors.judge_captions --input ... --output-dir ... --write-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

GEN_DIR = Path(__file__).resolve().parent.parent
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from reclassify_captions import (  # noqa: E402  (shared helpers)
    load_json, load_progress, now_iso, write_json_atomic,
)
from validation.checks import CAPTION_STATUS_MANUAL, CAPTION_STATUS_READY  # noqa: E402

from caption_errors.judge import (  # noqa: E402
    GRAMMAR, JUDGE_PROMPT_VERSION, OK, WRONG, OllamaChat, TransportError,
    judge_items,
)
from caption_errors.regenerate import REGEN_PROMPT_VERSION, regenerate_checked  # noqa: E402
from caption_errors.suspicious import SUSPICIOUS_VERSION, suspicious_reasons  # noqa: E402

CHECKER_VERSION = "c1"
PROGRESS_SUFFIX = "_caption_check_progress.jsonl"   # keep in sync with the Kaggle runner
STATE_SUFFIX = "_caption_check_state.json"
HEADER_KEYS = ("model", "judge_prompt_version", "regen_prompt_version",
               "regen_rounds", "input_name")
GOOD = (OK, GRAMMAR)


# ---------------------------------------------------------------- progress
def check_progress_header(path: Path, meta: Dict[str, Any]) -> None:
    """First line stores the setup; refuse to mix two setups in one file."""
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
    for key in HEADER_KEYS:
        if old.get(key) != meta.get(key):
            raise SystemExit(
                f"Progress file {path} was made with {key}={old.get(key)!r}, "
                f"this run uses {meta.get(key)!r}. Use the same settings, "
                f"another --output-dir, or delete the progress file to start over.")


# ---------------------------------------------------------------- one batch
def _chunks(seq: Sequence[Any], n: int):
    for i in range(0, len(seq), max(1, n)):
        yield seq[i:i + n]


def process_batch(rows: List[Dict[str, Any]], client: OllamaChat,
                  args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Judge a batch, regenerate its WRONG rows. Returns finished records.

    Rows whose verdict could not be read are left out (judged again later).
    Raises TransportError if the model stops replying: nothing is recorded
    for the batch and it is redone later.
    """
    kw = dict(retries=args.retries, backoff_s=args.retry_backoff)
    items = [(str(r["question"]), str(r["answer"]), str(r.get("caption") or ""))
             for r in rows]
    verdicts = judge_items(client, items, **kw)

    records: List[Dict[str, Any]] = []
    wrong: List[Dict[str, Any]] = []
    for row, res in zip(rows, verdicts):
        if res is None:
            continue
        v, reason = res
        rec = {"question_id": int(row["question_id"]), "verdict": v,
               "reason": reason, "original_verdict": v, "original_reason": reason}
        if v == WRONG and args.regen_rounds > 0:
            rec.update(caption=None, regen_rounds=0, history=[],
                       _q=str(row["question"]), _a=str(row["answer"]),
                       _last=str(row.get("caption") or ""), _reason=reason)
            wrong.append(rec)
        records.append(rec)

    for rnd in range(1, args.regen_rounds + 1):
        todo = [r for r in wrong if r["verdict"] == WRONG]
        if not todo:
            break
        new = regenerate_checked(
            client, [(r["_q"], r["_a"], r["_last"], r["_reason"]) for r in todo],
            batch_size=args.regen_batch_size,
            single_fallback=not args.no_regen_single_fallback,
            temperature=args.regen_temperature, **kw)
        to_judge: List[Tuple[Dict[str, Any], str]] = []
        for rec, (cap, why) in zip(todo, new):
            rec["regen_rounds"] = rnd
            if why is not None:          # unreadable / hard reject / contamination
                rec["history"].append({"caption": cap, "verdict": WRONG,
                                       "reason": "fast_validator", "detail": why})
                rec["_reason"] = "fast_validator"
                continue
            to_judge.append((rec, cap))
        if to_judge:
            res2 = judge_items(client, [(r["_q"], r["_a"], c) for r, c in to_judge],
                               **kw)
            for (rec, cap), res in zip(to_judge, res2):
                v, reason = res if res is not None else (WRONG, "unjudged")
                rec["history"].append({"caption": cap, "verdict": v,
                                       "reason": reason})
                rec["caption"] = cap        # last caption that passed fast checks
                rec["_last"] = cap
                if v in GOOD:
                    rec["verdict"], rec["reason"] = v, None
                else:
                    rec["reason"] = reason if reason != "unjudged" else rec["reason"]
                    rec["_reason"] = rec["reason"]

    for rec in records:
        for k in ("_q", "_a", "_last", "_reason"):
            rec.pop(k, None)
    return records


# ---------------------------------------------------------------- outputs
SCOPES = ("all", "manual")       # manual = only rows with caption_status
                                  # "Need to Manual validate" in the INPUT file


def in_scope(row: Dict[str, Any], scope: str) -> bool:
    return scope == "all" or row.get("caption_status") == CAPTION_STATUS_MANUAL


def summarize(rows: Sequence[Dict[str, Any]], done: Dict[int, Dict[str, Any]],
              scope: str = "all") -> Dict[str, Any]:
    judged = [done[int(r["question_id"])] for r in rows
              if int(r["question_id"]) in done]
    regen = [d for d in judged if d.get("regen_rounds")]
    scoped = [r for r in rows if in_scope(r, scope)]
    scoped_left = sum(1 for r in scoped if int(r["question_id"]) not in done)
    return {
        "scope": scope,
        "rows_total": len(rows),
        "rows_in_scope": len(scoped),
        "rows_judged": len(judged),
        "rows_left": scoped_left,
        "rows_left_all": len(rows) - len(judged),
        "original_verdict_counts": dict(Counter(d["original_verdict"] for d in judged)),
        "original_wrong_reasons": dict(Counter(d["original_reason"] for d in judged
                                               if d["original_verdict"] == WRONG)),
        "regenerated": len(regen),
        "fixed": sum(1 for d in regen if d["verdict"] in GOOD),
        "still_wrong": sum(1 for d in judged if d["verdict"] == WRONG),
        "final_verdict_counts": dict(Counter(d["verdict"] for d in judged)),
        "complete": scoped_left == 0,                 # every row of the scope judged
        "complete_all": len(judged) == len(rows),     # every row of the file judged
    }


def build_row(row: Dict[str, Any], rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        out[k] = v
        if k == "caption":
            out["caption_quality_verdict"] = rec["verdict"] if rec else None
            out["verdict_reason"] = rec["reason"] if rec else None
            out["suspicious"] = False            # filled in below (final caption)
            out["suspicious_reasons"] = []
    if rec is not None:
        if rec.get("regen_rounds"):
            out["caption_original"] = row.get("caption")
            out["caption_original_verdict_reason"] = rec["original_reason"]
            out["regen_rounds"] = rec["regen_rounds"]
            if rec.get("caption"):
                out["caption"] = rec["caption"]
        out["caption_status"] = (CAPTION_STATUS_READY if rec["verdict"] in GOOD
                                 else CAPTION_STATUS_MANUAL)
    # Step 3: code-only check of the FINAL caption, for a later bigger model.
    why = suspicious_reasons(str(out.get("question") or ""),
                             str(out.get("answer") or ""),
                             str(out.get("caption") or ""))
    out["suspicious"], out["suspicious_reasons"] = bool(why), why
    return out


SUSPICIOUS_SUFFIX = "_suspicious.json"
_SUSPICIOUS_KEYS = ("question_id", "image_id", "question", "answer", "caption",
                    "caption_quality_verdict", "verdict_reason", "caption_original",
                    "caption_status", "suspicious_reasons")


def write_outputs(data: Dict[str, Any], done: Dict[int, Dict[str, Any]],
                  in_path: Path, out_dir: Path, meta: Dict[str, Any],
                  started_at: str, extra: Optional[Dict[str, Any]] = None,
                  scope: str = "all") -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = data["annotations"]
    progress = summarize(rows, done, scope)
    new_rows = [build_row(r, done.get(int(r["question_id"]))) for r in rows]
    sus = [r for r in new_rows if r["suspicious"]]
    sus_judged = sum(1 for r in sus if r["caption_quality_verdict"] is not None)
    progress.update(
        suspicious=len(sus), suspicious_judged_rows=sus_judged,
        suspicious_reason_counts=dict(Counter(k.split(":")[0] for r in sus
                                              for k in r["suspicious_reasons"])))
    info = dict(data.get("info") or {})
    versions = dict(info.get("versions") or {})
    versions.update(caption_judge_prompt=JUDGE_PROMPT_VERSION,
                    caption_regen_prompt=REGEN_PROMPT_VERSION,
                    caption_suspicious_check=SUSPICIOUS_VERSION)
    info["versions"] = versions
    info["caption_error_fix"] = dict(meta, session_started_at=started_at,
                                     written_at=now_iso(), progress=progress,
                                     **(extra or {}))
    out_path = out_dir / in_path.name
    if out_path.resolve() == in_path.resolve():
        backup = in_path.with_name(in_path.stem + ".before_caption_check.json")
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"Backup of the input -> {backup}", flush=True)
    t = time.perf_counter()
    write_json_atomic(out_path, {**data, "info": info, "annotations": new_rows})
    write_json_atomic(out_dir / (in_path.stem + SUSPICIOUS_SUFFIX), {
        "info": {
            "description": "Rows whose FINAL caption may still be wrong (code check: "
                           "words not in question+answer, or flipped yes/no). "
                           "For a later pass with a bigger model.",
            "source_output": out_path.name,
            "suspicious_version": SUSPICIOUS_VERSION,
            "num_samples": len(sus),
            "of_rows": len(new_rows),
            "progress_complete": progress["complete"],
            "reason_counts": progress["suspicious_reason_counts"],
        },
        "annotations": [{k: r.get(k) for k in _SUSPICIOUS_KEYS} for r in sus],
    })
    write_json_atomic(out_dir / (in_path.stem + STATE_SUFFIX),
                      dict(progress, written_at=now_iso(), **(extra or {})))
    print(f"Captions JSON (scope {scope}: {progress['rows_in_scope'] - progress['rows_left']}"
          f"/{progress['rows_in_scope']} judged, complete={progress['complete']}; "
          f"whole file {progress['rows_judged']}/{progress['rows_total']}; suspicious "
          f"{progress['suspicious']}) -> {out_path} "
          f"[{time.perf_counter() - t:.0f}s]", flush=True)
    return progress


# ---------------------------------------------------------------- main loop
def preflight(client: OllamaChat) -> None:
    try:
        res = judge_items(client, [("How many dogs are there?", "2",
                                    "There are two dogs.")], retries=3, backoff_s=2)
    except TransportError as exc:
        raise SystemExit(f"Preflight: Ollama at {client.host} not reachable "
                         f"({exc}). Start it and re-run.")
    if res[0] is None:
        raise SystemExit("Preflight: judge reply could not be parsed. Check --model.")
    print(f"Preflight: model replied {res[0]}", flush=True)


def run_pass(todo: List[Dict[str, Any]], client: OllamaChat, progress_path: Path,
             done: Dict[int, Dict[str, Any]], args: argparse.Namespace,
             total: int) -> int:
    """One pass over ``todo``. Returns rows recorded in this pass."""
    batches = list(_chunks(todo, args.batch_size))
    started, recorded, last_print = time.perf_counter(), 0, 0.0
    fail_streak = 0
    counts: Counter = Counter()
    pool = ThreadPoolExecutor(max_workers=args.llm_parallel)
    inflight: Set[Future] = set()
    it = iter(batches)
    progress = open(progress_path, "a", encoding="utf-8")

    def submit_more() -> None:
        if args.deadline_unix and time.time() >= args.deadline_unix:
            return          # time limit: finish what is in flight, start nothing new
        while len(inflight) < args.llm_parallel:
            b = next(it, None)
            if b is None:
                return
            inflight.add(pool.submit(process_batch, b, client, args))

    try:
        submit_more()
        while inflight:
            if args.deadline_unix and time.time() >= args.deadline_unix + args.deadline_grace_s:
                raise _Deadline()
            finished, _ = wait(inflight, timeout=5, return_when=FIRST_COMPLETED)
            for fut in finished:
                inflight.discard(fut)
                try:
                    recs = fut.result()
                    fail_streak = 0
                except TransportError as exc:
                    fail_streak += 1
                    if fail_streak >= args.max_failed_batches:
                        raise _ModelDown(str(exc))
                    continue
                if recs:
                    progress.write("\n".join(json.dumps(r, ensure_ascii=False)
                                             for r in recs) + "\n")
                    progress.flush()
                for r in recs:
                    done[r["question_id"]] = r
                    counts[r["original_verdict"]] += 1
                    if r.get("regen_rounds"):
                        counts["regenerated"] += 1
                        counts["fixed" if r["verdict"] in GOOD else "still_wrong"] += 1
                recorded += len(recs)
            submit_more()
            el = time.perf_counter() - started
            if el - last_print >= args.log_every_s or not inflight:
                last_print = el
                rate = recorded / el if el > 0 else 0.0
                left = len(todo) - recorded
                eta = left / rate / 3600 if rate > 0 else float("inf")
                print(f"  {total - len(todo) + recorded}/{total} judged | {rate:.2f} rows/s | "
                      f"ETA {eta:.2f} h | this session: {dict(counts)}", flush=True)
    finally:
        progress.close()
        pool.shutdown(wait=False, cancel_futures=True)
    return recorded


class _ModelDown(Exception):
    pass


class _Deadline(Exception):
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="captions JSON to check")
    ap.add_argument("--output-dir", required=True,
                    help="progress files + the new captions JSON go here")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--llm-parallel", type=int, default=4,
                    help="batches processed at the same time (= requests in flight)")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="captions per judge request")
    ap.add_argument("--regen-batch-size", type=int, default=5,
                    help="WRONG captions per regeneration request (packed like "
                         "generation; rejects are redone one by one)")
    ap.add_argument("--no-regen-single-fallback", action="store_true",
                    help="do not redo rejected captions of a pack one by one")
    ap.add_argument("--regen-rounds", type=int, default=2)
    ap.add_argument("--regen-temperature", type=float, default=0.3)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds per Ollama request")
    ap.add_argument("--retries", type=int, default=3,
                    help="retries per request on connection/HTTP/timeout errors")
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument("--max-failed-batches", type=int, default=8,
                    help="stop after this many batches in a row with no reply")
    ap.add_argument("--max-passes", type=int, default=3,
                    help="passes over rows whose verdict could not be read")
    ap.add_argument("--log-every-s", type=float, default=60.0)
    ap.add_argument("--scope", choices=SCOPES, default="all",
                    help="all rows, or only rows whose INPUT caption_status is "
                         "'Need to Manual validate'. Progress is shared: a later "
                         "--scope all run skips rows already judged")
    ap.add_argument("--max-items", type=int, default=None,
                    help="only judge the first N rows (smoke test); the JSON "
                         "still holds every row")
    ap.add_argument("--write-only", action="store_true",
                    help="no LLM calls: rebuild the captions JSON from progress")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--deadline-unix", type=float, default=None,
                    help="stop starting new batches at this Unix time, then save "
                         "and exit (the Kaggle runner sets it from run.hours)")
    ap.add_argument("--deadline-grace-s", type=float, default=120.0,
                    help="after the deadline, wait at most this long for batches "
                         "in flight before saving")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started_at, t0 = now_iso(), time.perf_counter()
    in_path, out_dir = Path(args.input), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / (in_path.stem + PROGRESS_SUFFIX)

    print(f"Loading {in_path} ...", flush=True)
    data = load_json(in_path)
    rows: List[Dict[str, Any]] = data["annotations"]
    work_rows = [r for r in rows if in_scope(r, args.scope)]
    if args.max_items:
        work_rows = work_rows[:args.max_items]
    print(f"Scope: {args.scope} -> {len(work_rows)} of {len(rows)} rows", flush=True)

    meta = {"version": CHECKER_VERSION, "model": args.model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "regen_prompt_version": REGEN_PROMPT_VERSION,
            "regen_rounds": args.regen_rounds, "input_name": in_path.name,
            "batch_size": args.batch_size,
            "regen_batch_size": args.regen_batch_size,
            "regen_single_fallback": not args.no_regen_single_fallback,
            "regen_temperature": args.regen_temperature,
            "llm_parallel": args.llm_parallel}
    check_progress_header(progress_path, meta)
    done = load_progress(progress_path)
    print(f"Progress file: {progress_path} ({len(done)} rows already judged)",
          flush=True)

    def finish(code: int, why: str) -> None:
        # Ignore further Ctrl+C while the JSON is written (writes are atomic).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(why, flush=True)
        prog = write_outputs(data, done, in_path, out_dir, meta, started_at,
                             {"session_hours": round((time.perf_counter() - t0) / 3600, 3)},
                             scope=args.scope)
        if not prog["complete"]:
            print(f"NOT finished: {prog['rows_left']} of {prog['rows_in_scope']} rows "
                  f"(scope {args.scope}) left. Re-run with the same output dir "
                  f"(Kaggle: inputs.resume_dir) to continue.", flush=True)
        elif not prog["complete_all"]:
            print(f"Scope {args.scope} done. {prog['rows_left_all']} other rows are "
                  f"not judged; run with --scope all (same output dir) to do them "
                  f"too, nothing is judged twice.", flush=True)
        sys.stdout.flush()
        os._exit(code)          # don't wait for in-flight HTTP threads

    if args.write_only:
        finish(0, "--write-only: rebuilding outputs from the progress file.")

    client = OllamaChat(args.ollama_host, args.model, num_ctx=args.num_ctx,
                        timeout_s=args.timeout)
    todo = [r for r in work_rows if int(r["question_id"]) not in done]
    print(f"Rows: {len(rows)}, to judge in this run: {len(todo)} "
          f"(batch-size={args.batch_size}, llm-parallel={args.llm_parallel}, "
          f"regen-rounds={args.regen_rounds})", flush=True)
    try:
        if todo and not args.no_preflight:
            preflight(client)
        for p in range(1, args.max_passes + 1):
            todo = [r for r in work_rows if int(r["question_id"]) not in done]
            if not todo:
                break
            if p > 1:
                print(f"Pass {p}: {len(todo)} rows whose verdict could not be "
                      f"read earlier", flush=True)
            if run_pass(todo, client, progress_path, done, args, len(work_rows)) == 0:
                break
            if args.deadline_unix and time.time() >= args.deadline_unix:
                raise _Deadline()
    except _Deadline:
        finish(5, "Time limit reached: progress saved.")
    except KeyboardInterrupt:
        finish(130, "\nInterrupted: progress saved up to the last finished batch.")
    except _ModelDown as exc:
        finish(4, f"Stopping: {args.max_failed_batches} batches in a row got no "
                  f"reply from Ollama (last error: {exc}).")
    left = sum(1 for r in work_rows if int(r["question_id"]) not in done)
    finish(0 if left == 0 else 3,
           f"Run finished in {(time.perf_counter() - t0) / 3600:.2f} h.")


if __name__ == "__main__":
    main()
