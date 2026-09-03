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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_SEED = 42
DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT_S = 300.0

LABELS = frozenset({"PASS", "FAIL"})
ERROR_TYPES = frozenset(
    {"NONE", "GRAMMAR_ERROR", "HALLUCINATION", "WRONG_CAPTION"}
)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "this", "that",
    "these", "those", "there", "here", "it", "its", "do", "does", "did",
    "can", "could", "will", "would", "have", "has", "had", "you", "your",
    "what", "which", "who", "where", "when", "why", "how", "many", "much",
    "any", "some", "from", "by", "as", "if", "than", "then", "so", "too",
    "very", "just", "about", "into", "over", "after", "before", "between",
    "out", "up", "down", "off", "again", "further", "once", "all", "both",
    "each", "few", "more", "most", "other", "such", "only", "own", "same",
    "i", "me", "my", "we", "our", "he", "she", "they", "them", "his", "her",
    "their", "no", "not", "nor", "never", "none",
}

_AUDITOR_SYSTEM_PROMPT = (
    "You are a strict VQA caption auditor.\n"
    "\n"
    "Reply with ONLY a JSON array. "
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
    "\n"
    "FAIL — choose one error_type:\n"
    "\n"
    "GRAMMAR_ERROR:\n"
    "The caption has grammatical mistakes or unnatural language.\n"
    "\n"
    "HALLUCINATION:\n"
    "The caption contains information not supported by the question and "
    "answer.\n"
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
    "Question: What kind of weather it is?\n"
    "Answer: sunny\n"
    "Caption: The weather it is is a sunny weather it.\n"
    'Output: {"label": "FAIL", "error_type": "GRAMMAR_ERROR", '
    '"reason": "Caption has incorrect grammar and unnatural wording."}\n'
    "\n"
    "Example 7\n"
    "Question: What game is being played?\n"
    "Answer: soccer\n"
    "Caption: Two children are playing soccer.\n"
    'Output: {"label": "FAIL", "error_type": "HALLUCINATION", '
    '"reason": "The number of children is not provided in the question '
    'or answer."}\n'
    "\n"
    "Example 8\n"
    "Question: Is the dog sleeping?\n"
    "Answer: yes\n"
    "Caption: The brown dog is sleeping on the couch.\n"
    'Output: {"label": "FAIL", "error_type": "HALLUCINATION", '
    '"reason": "Color and location information are not provided."}\n'
    "\n"
    "Example 9\n"
    "Question: How many birds are flying?\n"
    "Answer: 4\n"
    "Caption: Three birds are flying.\n"
    'Output: {"label": "FAIL", "error_type": "WRONG_CAPTION", '
    '"reason": "Caption gives an incorrect number."}\n'
    "\n"
    "Example 10\n"
    "Question: What is the woman holding?\n"
    "Answer: umbrella\n"
    "Caption: The woman is standing.\n"
    'Output: {"label": "FAIL", "error_type": "WRONG_CAPTION", '
    '"reason": "Caption does not contain the answer."}'
)


def content_words(text: str) -> Set[str]:
    """Lowercase alphanumeric tokens minus a small stopword list."""
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if t and t not in _STOPWORDS}


def caption_precision_recall(
    question: str, answer: str, caption: str
) -> Tuple[float, float]:
    """Q+A grounding precision and answer recall for one caption."""
    c_words = content_words(caption)
    a_words = content_words(answer)
    s_words = content_words(f"{question} {answer}")

    if not c_words:
        precision = 0.0
    else:
        precision = len(c_words & s_words) / len(c_words)

    if not a_words:
        recall = 1.0 if (caption or "").strip() else 0.0
    else:
        recall = len(c_words & a_words) / len(a_words)

    return precision, recall


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
        f'"id", "label", "error_type", and "reason".'
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


def parse_auditor_response(
    raw: str,
    expected_ids: Sequence[int],
) -> Dict[int, Dict[str, str]]:
    """Parse JSON array into id -> {label, error_type, reason}. Fail-closed."""
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")
    id_set = set(expected_ids)
    results: Dict[int, Dict[str, str]] = {}

    def fail_all(detail: str) -> Dict[int, Dict[str, str]]:
        return {
            i: {
                "label": "FAIL",
                "error_type": "NONE",
                "reason": detail,
            }
            for i in expected_ids
        }

    if start < 0 or end <= start:
        return fail_all("parse_no_json_array")

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return fail_all(f"parse_json_error:{exc}")

    if not isinstance(data, list):
        return fail_all("parse_not_a_list")

    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if idx_int not in id_set:
            continue
        label, error_type, reason = _normalize_verdict(
            label_raw=str(entry.get("label", "")),
            error_type_raw=str(entry.get("error_type", "")),
            reason_raw=str(entry.get("reason", "")),
        )
        results[idx_int] = {
            "label": label,
            "error_type": error_type,
            "reason": reason,
        }

    out: Dict[int, Dict[str, str]] = {}
    for i in expected_ids:
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
            "num_ctx": 4096,
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


def audit_batch(
    batch_rows: Sequence[Dict[str, Any]],
    *,
    host: str,
    model: str,
    start_id: int = 0,
) -> List[Dict[str, Any]]:
    """Audit one batch; return records with label/error_type/reason/P/R."""
    items = []
    for offset, row in enumerate(batch_rows):
        items.append(
            {
                "id": start_id + offset,
                "question": str(row.get("question") or ""),
                "answer": str(row.get("answer") or ""),
                "caption": str(row.get("caption") or ""),
            }
        )

    system, user = build_auditor_prompt(items)
    expected_ids = [int(it["id"]) for it in items]
    num_predict = max(128, len(items) * 48 + 64)

    try:
        content = ollama_chat(
            host=host,
            model=model,
            system=system,
            user=user,
            num_predict=num_predict,
        )
    except Exception as exc:
        verdicts = {
            i: {
                "label": "FAIL",
                "error_type": "NONE",
                "reason": f"llm_auditor_error:{exc}",
            }
            for i in expected_ids
        }
    else:
        if not content.strip():
            verdicts = {
                i: {
                    "label": "FAIL",
                    "error_type": "NONE",
                    "reason": "empty_auditor_response",
                }
                for i in expected_ids
            }
        else:
            verdicts = parse_auditor_response(content, expected_ids)

    records: List[Dict[str, Any]] = []
    for offset, row in enumerate(batch_rows):
        idx = start_id + offset
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        caption = str(row.get("caption") or "")
        precision, recall = caption_precision_recall(question, answer, caption)
        verdict = verdicts[idx]
        rec: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "caption": caption,
            "label": verdict["label"],
            "reason": verdict["reason"],
            "error_type": verdict["error_type"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
        if "question_id" in row:
            rec["question_id"] = row["question_id"]
        if "image_id" in row:
            rec["image_id"] = row["image_id"]
        if "rule" in row:
            rec["rule"] = row["rule"]
        records.append(rec)
    return records


def run_audit(
    rows: Sequence[Dict[str, Any]],
    test_item_count: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
) -> Tuple[List[Dict[str, Any]], int]:
    """Sample and audit; return (records, pool_size)."""
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
        all_records.extend(
            audit_batch(chunk, host=host, model=model, start_id=start)
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
    if records:
        mean_precision = sum(float(r["precision"]) for r in records) / len(
            records
        )
        mean_recall = sum(float(r["recall"]) for r in records) / len(records)
    else:
        mean_precision = 0.0
        mean_recall = 0.0

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
            "mean_precision": round(mean_precision, 4),
            "mean_recall": round(mean_recall, 4),
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
    print(
        f"  mean_precision={info['mean_precision']} "
        f"mean_recall={info['mean_recall']}"
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
