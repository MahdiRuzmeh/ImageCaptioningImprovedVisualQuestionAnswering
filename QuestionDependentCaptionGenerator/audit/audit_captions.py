"""LLM caption auditor: sample k captions and judge them in Ollama batches.

Usage (from QuestionDependentCaptionGenerator/):

    python audit/audit_captions.py outputs/vqa_v2_question_dependent_captions_train2014.json 50
    python audit/audit_captions.py outputs/vqa_v2_question_dependent_captions_train2014.json 50 --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_SEED = 42
DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT_S = 300.0
AUDIT_DIR = Path(__file__).resolve().parent
ERROR_LOG_PATH = AUDIT_DIR / "llm_caption_audit_errors.jsonl"

LABELS = frozenset({"PASS", "FAIL"})
ERROR_TYPES = frozenset(
    {"NONE", "GRAMMAR_ERROR", "HALLUCINATION", "WRONG_CAPTION"}
)

_AUDITOR_SYSTEM_PROMPT = (
    "You are a strict VQA caption auditor.\n"
    "\n"
    "Reply with ONLY a JSON array. "
    "Use the same id numbers shown in each --- Item <id> --- header "
    "(0-based). "
    'Each element must be {"id": <number>, "label": "PASS" or "FAIL", '
    '"error_type": "NONE" or "GRAMMAR_ERROR" or "HALLUCINATION" or '
    '"WRONG_CAPTION", "reason": "<short explanation>"}.'
)

_AUDITOR_RULES_AND_FEW_SHOTS = (
    "Your task is to evaluate whether each generated caption is correct "
    "based ONLY on its question and answer.\n"
    "\n"
    "Evaluation rules:\n"
    "\n"
    "PASS:\n"
    "- Caption is grammatically correct.\n"
    "- Caption naturally expresses the answer.\n"
    "- Caption does not add any information beyond the question and answer.\n"
    "- Caption meaning matches the answer.\n"
    "- Restating the answer in a full sentence is PASS (this is expected).\n"
    "- Answer words/phrases in the caption are allowed and required when they "
    "express the answer.\n"
    "- Reusing question words plus grammar (a/an/the, is/are, looking at, "
    "used for) is not extra information.\n"
    "\n"
    "FAIL — choose one error_type:\n"
    "\n"
    "GRAMMAR_ERROR:\n"
    "The caption has grammatical mistakes or unnatural language.\n"
    "\n"
    "HALLUCINATION:\n"
    "Only when the caption adds facts not present in the question or the "
    "answer (e.g. colors, counts, locations not in Q/A). "
    "Do not mark HALLUCINATION for including the answer text.\n"
    "\n"
    "WRONG_CAPTION:\n"
    "The caption does not correctly represent the answer or contradicts it.\n"
    "\n"
    "For PASS use error_type NONE.\n"
    "\n"
    "Examples:\n"
    "\n"
    "Example 1\n"
    "Question: How many tracks are in the snow?\n"
    "Answer: 3\n"
    "Caption: There are three tracks.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption correctly expresses the answer."}\n'
    "\n"
    "Example 2\n"
    "Question: Is the train moving?\n"
    "Answer: no\n"
    "Caption: The train is not moving.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption correctly represents the negative answer."}\n'
    "\n"
    "Example 3\n"
    "Question: What game is being played?\n"
    "Answer: soccer\n"
    "Caption: Soccer is being played.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption directly expresses the answer."}\n'
    "\n"
    "Example 4\n"
    "Question: What are the animals doing?\n"
    "Answer: eating\n"
    "Caption: The animals are eating.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption matches the answer without extra information."}\n'
    "\n"
    "Example 5\n"
    "Question: What color is the bus?\n"
    "Answer: yellow\n"
    "Caption: The bus is yellow.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption correctly describes the answer."}\n'
    "\n"
    "Example 6\n"
    "Question: Is that a stove?\n"
    "Answer: yes\n"
    "Caption: That is a stove.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption affirms the yes answer as a statement."}\n'
    "\n"
    "Example 7\n"
    "Question: Are they at a zoo?\n"
    "Answer: yes\n"
    "Caption: They are at a zoo.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption affirms the yes answer as a statement."}\n'
    "\n"
    "Example 8\n"
    "Question: What is the color scheme of the photo?\n"
    "Answer: black and white\n"
    "Caption: The color scheme of the photo is black and white.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption restates the answer without extras."}\n'
    "\n"
    "Example 9\n"
    "Question: What color plate is this?\n"
    "Answer: white\n"
    "Caption: A white plate is shown.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption paraphrases the answer naturally."}\n'
    "\n"
    "Example 10\n"
    "Question: What is leaning against the wall?\n"
    "Answer: skateboard\n"
    "Caption: Skateboard is leaning against the wall.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption restates the answer with the question context."}\n'
    "\n"
    "Example 11\n"
    "Question: What is the object in the water used for?\n"
    "Answer: entertainment\n"
    "Caption: The object in the water is used for entertainment.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption restates the answer; answer words are allowed."}\n'
    "\n"
    "Example 12\n"
    "Question: What color is the disk?\n"
    "Answer: yellow and blue\n"
    "Caption: The disk is yellow and blue.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption correctly restates the multi-word answer."}\n'
    "\n"
    "Example 13\n"
    "Question: What is the giraffe looking at?\n"
    "Answer: camera\n"
    "Caption: The giraffe is looking at a camera.\n"
    'Output: {"label": "PASS", "error_type": "NONE", '
    '"reason": "Caption matches the answer; article a is fine."}\n'
    "\n"
    "Example 14\n"
    "Question: What kind of weather it is?\n"
    "Answer: sunny\n"
    "Caption: The weather it is is a sunny weather it.\n"
    'Output: {"label": "FAIL", "error_type": "GRAMMAR_ERROR", '
    '"reason": "Caption has incorrect grammar and unnatural wording."}\n'
    "\n"
    "Example 15\n"
    "Question: What game is being played?\n"
    "Answer: soccer\n"
    "Caption: Two children are playing soccer.\n"
    'Output: {"label": "FAIL", "error_type": "HALLUCINATION", '
    '"reason": "The number of children is not provided in the question '
    'or answer."}\n'
    "\n"
    "Example 16\n"
    "Question: Is the dog sleeping?\n"
    "Answer: yes\n"
    "Caption: The brown dog is sleeping on the couch.\n"
    'Output: {"label": "FAIL", "error_type": "HALLUCINATION", '
    '"reason": "Color and location information are not provided."}\n'
    "\n"
    "Example 17\n"
    "Question: How many birds are flying?\n"
    "Answer: 4\n"
    "Caption: Three birds are flying.\n"
    'Output: {"label": "FAIL", "error_type": "WRONG_CAPTION", '
    '"reason": "Caption gives an incorrect number."}\n'
    "\n"
    "Example 18\n"
    "Question: What is the woman holding?\n"
    "Answer: umbrella\n"
    "Caption: The woman is standing.\n"
    'Output: {"label": "FAIL", "error_type": "WRONG_CAPTION", '
    '"reason": "Caption does not contain the answer."}'
)


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.I)
    if m:
        return m.group(1).strip()
    return t


def build_auditor_prompt(
    items: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    """System + user messages for one batch of audit items."""
    lines: List[str] = [
        _AUDITOR_RULES_AND_FEW_SHOTS,
        "",
        "Now audit the items below.",
        "",
    ]
    for item in items:
        idx = int(item["id"])
        lines.append(f"--- Item {idx} ---")
        lines.append(f"QUESTION: {item['question']}")
        lines.append(f"ANSWER: {item['answer']}")
        lines.append(f"CAPTION: {item['caption']}")
        lines.append("")
    lines.append(
        f'Return a JSON array of exactly {len(items)} objects with keys '
        f'"id", "label", "error_type", and "reason". '
        f"Keep each reason under 12 words."
    )
    return _AUDITOR_SYSTEM_PROMPT, "\n".join(lines)


def _normalize_verdict(
    *,
    label_raw: str,
    error_type_raw: str,
    reason_raw: str,
    parse_detail: str = "",
) -> Tuple[str, str, str]:
    """Normalize label / error_type / reason; fail-closed on bad label."""
    label = (label_raw or "").strip().upper().replace("-", "_")
    error_type = (error_type_raw or "").strip().upper().replace("-", "_")
    reason = (reason_raw or "").strip()

    if label not in LABELS:
        return (
            "FAIL",
            "NONE",
            parse_detail or f"parse_invalid_label:{label_raw!r}",
        )

    if label == "PASS":
        error_type = "NONE"
    elif error_type not in ERROR_TYPES or error_type == "NONE":
        error_type = "WRONG_CAPTION"

    if not reason:
        reason = parse_detail or "no reason provided"
    return label, error_type, reason


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Pull complete ``{...}`` JSON objects from possibly truncated text."""
    objects: List[Dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        start = i
        j = i
        closed = False
        while j < n:
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : j + 1]
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict):
                            objects.append(obj)
                        i = j + 1
                        closed = True
                        break
            j += 1
        if not closed:
            break
    return objects


def _entries_from_response(raw: str) -> Tuple[List[Any], str]:
    """Return (list_entries, error_detail). error_detail empty when usable."""
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data, ""
            return [], "parse_not_a_list"
        except json.JSONDecodeError as exc:
            objects = _extract_json_objects(text)
            if objects:
                return objects, ""
            return [], f"parse_json_error:{exc}"

    objects = _extract_json_objects(text)
    if objects:
        return objects, ""
    return [], "parse_no_json_array"


_PARSE_FAILURE_PREFIXES = (
    "parse_",
    "missing_id_in_response",
    "empty_auditor_response",
    "llm_auditor_error:",
)


def _is_parse_failure(reason: str) -> bool:
    r = reason or ""
    return any(r.startswith(p) for p in _PARSE_FAILURE_PREFIXES)


def parse_auditor_response(
    raw: str,
    expected_ids: Sequence[int],
) -> Dict[int, Dict[str, str]]:
    """Parse JSON array into id -> {label, error_type, reason}. Fail-closed.

    Robust to common small-model quirks:
    - truncated JSON arrays (salvage complete objects)
    - missing ``id`` (fill by array order)
    - 1-based ids when expected ids are 0-based
    - extra/unknown ids ignored
    """
    id_list = list(expected_ids)
    id_set = set(id_list)
    n = len(id_list)

    def fail_all(detail: str) -> Dict[int, Dict[str, str]]:
        return {
            i: {
                "label": "FAIL",
                "error_type": "NONE",
                "reason": detail,
            }
            for i in id_list
        }

    data, err = _entries_from_response(raw)
    if not data:
        return fail_all(err or "parse_no_json_array")

    parsed: List[Tuple[Optional[int], Dict[str, str]]] = []
    raw_ids: List[int] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if not any(k in entry for k in ("label", "verdict", "error_type", "reason")):
            continue
        label_raw = entry.get("label", entry.get("verdict", ""))
        label, error_type, reason = _normalize_verdict(
            label_raw=str(label_raw),
            error_type_raw=str(entry.get("error_type", "")),
            reason_raw=str(entry.get("reason", "")),
        )
        verdict = {
            "label": label,
            "error_type": error_type,
            "reason": reason,
        }
        idx = entry.get("id", entry.get("index", entry.get("item")))
        idx_int: Optional[int] = None
        if idx is not None:
            try:
                idx_int = int(idx)
                raw_ids.append(idx_int)
            except (TypeError, ValueError):
                idx_int = None
        parsed.append((idx_int, verdict))

    if not parsed:
        return fail_all(err or "parse_no_verdict_objects")

    one_based = False
    if (
        n > 0
        and id_list == list(range(n))
        and raw_ids
        and 0 not in raw_ids
        and all(1 <= i <= n for i in raw_ids)
    ):
        one_based = True

    results: Dict[int, Dict[str, str]] = {}
    unmatched_entries: List[Dict[str, str]] = []

    for idx_int, verdict in parsed:
        if idx_int is None:
            unmatched_entries.append(verdict)
            continue
        mapped = idx_int - 1 if one_based else idx_int
        if mapped in id_set and mapped not in results:
            results[mapped] = verdict
        else:
            unmatched_entries.append(verdict)

    next_slot = 0
    for verdict in unmatched_entries:
        while next_slot < n and id_list[next_slot] in results:
            next_slot += 1
        if next_slot >= n:
            break
        results[id_list[next_slot]] = verdict
        next_slot += 1

    out: Dict[int, Dict[str, str]] = {}
    for i in id_list:
        if i in results:
            out[i] = results[i]
        else:
            out[i] = {
                "label": "FAIL",
                "error_type": "NONE",
                "reason": "missing_id_in_response",
            }
    return out


def ollama_chat(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    num_predict: int,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Call Ollama /api/chat; return message content or raise."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": num_predict,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    base = host.rstrip("/")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    msg = raw.get("message") or {}
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return ""


def sample_rows(
    rows: Sequence[Dict[str, Any]],
    test_item_count: int,
    *,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Random sample of rows with non-empty captions."""
    pool = [r for r in rows if str(r.get("caption") or "").strip()]
    if not pool:
        return []
    k = min(int(test_item_count), len(pool))
    rng = random.Random(seed)
    return rng.sample(pool, k)


def _call_batch_verdicts(
    items: Sequence[Dict[str, Any]],
    *,
    host: str,
    model: str,
) -> Dict[int, Dict[str, str]]:
    """One Ollama batch call → id -> verdict (ids must be 0..n-1)."""
    expected_ids = [int(it["id"]) for it in items]
    if not items:
        return {}
    system, user = build_auditor_prompt(items)
    num_predict = max(512, len(items) * 160 + 256)
    try:
        content = ollama_chat(
            host=host,
            model=model,
            system=system,
            user=user,
            num_predict=num_predict,
        )
    except Exception as exc:
        return {
            i: {
                "label": "FAIL",
                "error_type": "NONE",
                "reason": f"llm_auditor_error:{exc}",
            }
            for i in expected_ids
        }
    if not content.strip():
        return {
            i: {
                "label": "FAIL",
                "error_type": "NONE",
                "reason": "empty_auditor_response",
            }
            for i in expected_ids
        }
    return parse_auditor_response(content, expected_ids)


def _row_to_record(
    row: Dict[str, Any],
    verdict: Dict[str, str],
) -> Dict[str, Any]:
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    caption = str(row.get("caption") or "")
    rec: Dict[str, Any] = {
        "question": question,
        "answer": answer,
        "caption": caption,
        "label": verdict["label"],
        "reason": verdict["reason"],
        "error_type": verdict["error_type"],
    }
    if "question_id" in row:
        rec["question_id"] = row["question_id"]
    if "image_id" in row:
        rec["image_id"] = row["image_id"]
    if "rule" in row:
        rec["rule"] = row["rule"]
    return rec


def audit_batch(
    batch_rows: Sequence[Dict[str, Any]],
    *,
    host: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Audit one batch; return records with label/error_type/reason.

    Item ids sent to the LLM are always batch-local ``0 .. n-1``.
    """
    items = [
        {
            "id": offset,
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "caption": str(row.get("caption") or ""),
        }
        for offset, row in enumerate(batch_rows)
    ]
    verdicts = _call_batch_verdicts(items, host=host, model=model)
    return [
        _row_to_record(row, verdicts[offset])
        for offset, row in enumerate(batch_rows)
    ]


def _retry_parse_failures_batched(
    records: List[Dict[str, Any]],
    *,
    batch_size: int,
    host: str,
    model: str,
) -> List[Dict[str, Any]]:
    """One batched retry for parse/missing failures; return still-failed records."""
    fail_idxs = [
        i
        for i, rec in enumerate(records)
        if _is_parse_failure(str(rec.get("reason") or ""))
    ]
    if not fail_idxs:
        return []

    print(
        f"  retry batch: {len(fail_idxs)} parse/missing failures "
        f"(batch_size={batch_size})",
        flush=True,
    )
    still_failed: List[Dict[str, Any]] = []
    batch_n = max(1, int(batch_size))
    for start in range(0, len(fail_idxs), batch_n):
        chunk_idxs = fail_idxs[start : start + batch_n]
        items = [
            {
                "id": j,
                "question": str(records[idx].get("question") or ""),
                "answer": str(records[idx].get("answer") or ""),
                "caption": str(records[idx].get("caption") or ""),
            }
            for j, idx in enumerate(chunk_idxs)
        ]
        verdicts = _call_batch_verdicts(items, host=host, model=model)
        for j, idx in enumerate(chunk_idxs):
            # Preserve ids from the original record while refreshing verdict.
            base = {
                k: records[idx][k]
                for k in ("question_id", "image_id", "rule")
                if k in records[idx]
            }
            base.update(
                {
                    "question": records[idx]["question"],
                    "answer": records[idx]["answer"],
                    "caption": records[idx]["caption"],
                }
            )
            updated = _row_to_record(base, verdicts[j])
            records[idx] = updated
            if _is_parse_failure(str(updated.get("reason") or "")):
                still_failed.append(dict(updated))
    return still_failed


def log_audit_errors(
    error_path: Path,
    failures: Sequence[Dict[str, Any]],
    *,
    source: Path,
) -> None:
    """Append remaining parse/LLM failures to a JSONL log under audit/."""
    if not failures:
        return
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open("a", encoding="utf-8") as f:
        for rec in failures:
            entry = {
                "source": str(source.resolve()),
                "question_id": rec.get("question_id"),
                "image_id": rec.get("image_id"),
                "question": rec.get("question"),
                "answer": rec.get("answer"),
                "caption": rec.get("caption"),
                "rule": rec.get("rule"),
                "label": rec.get("label"),
                "error_type": rec.get("error_type"),
                "reason": rec.get("reason"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(
        f"  logged {len(failures)} remaining parse/LLM errors -> {error_path}",
        flush=True,
    )


def run_audit(
    rows: Sequence[Dict[str, Any]],
    test_item_count: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    error_log_path: Optional[Path] = None,
    source: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Sample and audit; return (records, pool_size).

    Parse/missing failures are retried once as a batch; leftovers are logged.
    """
    pool = [r for r in rows if str(r.get("caption") or "").strip()]
    sampled = sample_rows(rows, test_item_count, seed=seed)
    if not sampled:
        return [], len(pool)

    batch_n = max(1, int(batch_size))
    all_records: List[Dict[str, Any]] = []
    total = len(sampled)
    for start in range(0, total, batch_n):
        chunk = sampled[start : start + batch_n]
        end = start + len(chunk)
        print(
            f"  auditor batch: items {start + 1}-{end}/{total} "
            f"(batch_size={len(chunk)})",
            flush=True,
        )
        all_records.extend(audit_batch(chunk, host=host, model=model))

    still_failed = _retry_parse_failures_batched(
        all_records,
        batch_size=batch_n,
        host=host,
        model=model,
    )
    if still_failed and error_log_path is not None:
        log_audit_errors(
            error_log_path,
            still_failed,
            source=source or Path("."),
        )
    return all_records, len(pool)


def write_report(
    output_path: Path,
    *,
    source: Path,
    records: List[Dict[str, Any]],
    test_item_count: int,
    pool_size: int,
    batch_size: int,
    seed: int,
    model: str,
    host: str,
) -> Dict[str, Any]:
    """Write audit JSON and return the payload."""
    label_counts = Counter(str(r.get("label")) for r in records)
    error_counts = Counter(str(r.get("error_type")) for r in records)

    payload = {
        "info": {
            "source": str(source.resolve()),
            "test_item_count": test_item_count,
            "seed": seed,
            "batch_size": batch_size,
            "model": model,
            "host": host,
            "pool_size": pool_size,
            "num_audited": len(records),
            "label_counts": dict(label_counts),
            "error_type_counts": dict(error_counts),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "caption_file_path",
        type=Path,
        help="Path to captions dataset JSON (info + annotations)",
    )
    parser.add_argument(
        "test_item_count",
        type=int,
        help="Number of captions to sample and audit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Items per Ollama call (default {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    caption_path: Path = args.caption_file_path
    if not caption_path.is_file():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")
    if args.test_item_count < 1:
        raise ValueError("test_item_count must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    with caption_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("annotations") or []
    if not isinstance(rows, list):
        raise ValueError("annotations must be a list")

    print(
        f"LLM caption auditor: source={caption_path} "
        f"test_item_count={args.test_item_count} "
        f"batch_size={args.batch_size} model={DEFAULT_MODEL}",
        flush=True,
    )
    records, pool_size = run_audit(
        rows,
        args.test_item_count,
        batch_size=args.batch_size,
        seed=DEFAULT_SEED,
        host=DEFAULT_HOST,
        model=DEFAULT_MODEL,
        error_log_path=ERROR_LOG_PATH,
        source=caption_path,
    )
    k = len(records)
    out_path = caption_path.with_name(
        f"{caption_path.stem}_llm_caption_audit_k{k}_seed{DEFAULT_SEED}.json"
    )
    payload = write_report(
        out_path,
        source=caption_path,
        records=records,
        test_item_count=args.test_item_count,
        pool_size=pool_size,
        batch_size=args.batch_size,
        seed=DEFAULT_SEED,
        model=DEFAULT_MODEL,
        host=DEFAULT_HOST,
    )
    info = payload["info"]
    print(f"Wrote {k} audit records -> {out_path}")
    print(f"  pool_size={pool_size}")
    print(f"  label_counts={info['label_counts']}")
    print(f"  error_type_counts={info['error_type_counts']}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
