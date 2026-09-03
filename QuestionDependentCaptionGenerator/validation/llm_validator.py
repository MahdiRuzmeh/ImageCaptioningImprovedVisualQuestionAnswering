"""Batched LLM PASS/FAIL judge for captions the fast validator marked UNKNOWN."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from validation.config import ValidationConfig


class LlmVerdict(str, Enum):
    """Binary outcome from the LLM judge."""

    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class JudgeItem:
    """One item sent to the batched LLM judge."""

    index: int
    question: str
    answer: str
    caption: str


@dataclass
class JudgeResult:
    """Per-item LLM judge outcome."""

    index: int
    verdict: LlmVerdict
    detail: str = ""


def _preview(text: str, limit: int = 400) -> str:
  flat = " ".join(text.split())
  if len(flat) <= limit:
    return flat
  return flat[: limit - 3] + "..."


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.I)
    if m:
        return m.group(1).strip()
    return t


_JUDGE_SYSTEM_PROMPT = (
    "You are validating captions generated from a Visual Question Answering "
    "(VQA) dataset.\n"
    "\n"
    "Reply with ONLY a JSON array. "
    'Each element must be {"id": <number>, "verdict": "PASS" or "FAIL"}.'
)

_JUDGE_RULES_AND_FEW_SHOTS = (
    "Input for each item:\n"
    "- Question\n"
    "- Answer\n"
    "- Caption\n"
    "\n"
    "Label each item PASS or FAIL.\n"
    "\n"
    "PASS if ALL of the following are true:\n"
    "1. The caption is grammatically correct and natural.\n"
    "2. The caption clearly expresses the answer.\n"
    "3. Every piece of information in the caption can be inferred ONLY from "
    "the question and answer.\n"
    "4. The caption does not add extra facts, attributes, objects, actions, "
    "colors, locations, numbers, or relationships that are not present in "
    "the question and answer.\n"
    "\n"
    "Natural paraphrases are PASS when they express the answer and add no "
    "extra facts (for example: omitting a location phrase already in the "
    "question, using an antonym for a no-answer such as \"closed\" for "
    "\"not open\", or rewriting \"What is X doing? / eating\" as "
    "\"X is eating.\").\n"
    "\n"
    "FAIL if ANY of the following occur:\n"
    "- Grammar errors.\n"
    "- Awkward or unnatural wording.\n"
    "- Missing or incorrect answer.\n"
    "- Hallucinated information not supported by the question and answer.\n"
    "- Changed meaning.\n"
    "- Contradiction with the answer.\n"
    "- Unnecessary extra details.\n"
    "\n"
    "Examples:\n"
    "\n"
    "Example 1\n"
    "Question: How many tracks are in the snow?\n"
    "Answer: 3\n"
    "Caption: There are three tracks.\n"
    "Label: PASS\n"
    "\n"
    "Example 2\n"
    "Question: Is the train moving?\n"
    "Answer: no\n"
    "Caption: The train is not moving.\n"
    "Label: PASS\n"
    "\n"
    "Example 3\n"
    "Question: What game is being played?\n"
    "Answer: soccer\n"
    "Caption: Soccer is being played.\n"
    "Label: PASS\n"
    "\n"
    "Example 4\n"
    "Question: What are the animals doing?\n"
    "Answer: eating\n"
    "Caption: The animals are eating.\n"
    "Label: PASS\n"
    "\n"
    "Example 5\n"
    "Question: What color is the bus?\n"
    "Answer: yellow\n"
    "Caption: The bus is yellow.\n"
    "Label: PASS\n"
    "\n"
    "Example 6\n"
    "Question: What game is being played?\n"
    "Answer: soccer\n"
    "Caption: Two children are playing soccer.\n"
    "Label: FAIL\n"
    "\n"
    "Example 7\n"
    "Question: Is the dog sleeping?\n"
    "Answer: yes\n"
    "Caption: The brown dog is sleeping on the couch.\n"
    "Label: FAIL\n"
    "\n"
    "Example 8\n"
    "Question: How many people are there?\n"
    "Answer: 2\n"
    "Caption: Two people are smiling.\n"
    "Label: FAIL\n"
    "\n"
    "Example 9\n"
    "Question: What kind of weather it is?\n"
    "Answer: sunny\n"
    "Caption: The weather it is is a sunny weather it.\n"
    "Label: FAIL\n"
    "\n"
    "Example 10\n"
    "Question: Is there grass?\n"
    "Answer: yes\n"
    "Caption: The there is grass.\n"
    "Label: FAIL"
)


def build_judge_prompt(items: Sequence[JudgeItem]) -> Tuple[str, str]:
    """Build system + user messages for a batched judge call.

    Each item is numbered independently; no cross-item context is shared.
    """
    lines: List[str] = [
        _JUDGE_RULES_AND_FEW_SHOTS,
        "",
        "Now classify the items below.",
        "",
    ]
    for item in items:
        lines.append(f"--- Item {item.index} ---")
        lines.append(f"QUESTION: {item.question}")
        lines.append(f"ANSWER: {item.answer}")
        lines.append(f"CAPTION: {item.caption}")
        lines.append("")
    lines.append(
        f'Return a JSON array of exactly {len(items)} objects with keys "id" and "verdict".'
    )
    return _JUDGE_SYSTEM_PROMPT, "\n".join(lines)


def parse_judge_response(
    raw: str,
    items: Sequence[JudgeItem],
) -> List[JudgeResult]:
    """Parse model JSON array into per-item :class:`JudgeResult`.

    Fail-closed: parse errors or missing ids → FAIL for affected items.
    """
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")
    expected_ids = {item.index for item in items}
    results: Dict[int, JudgeResult] = {}

    if start < 0 or end <= start:
        return [
            JudgeResult(index=i, verdict=LlmVerdict.FAIL, detail="parse_no_json_array")
            for i in sorted(expected_ids)
        ]

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return [
            JudgeResult(
                index=i,
                verdict=LlmVerdict.FAIL,
                detail=f"parse_json_error:{exc}",
            )
            for i in sorted(expected_ids)
        ]

    if not isinstance(data, list):
        return [
            JudgeResult(index=i, verdict=LlmVerdict.FAIL, detail="parse_not_a_list")
            for i in sorted(expected_ids)
        ]

    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        verdict_raw = str(entry.get("verdict", "")).strip().upper()
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if idx_int not in expected_ids:
            continue
        if verdict_raw == "PASS":
            results[idx_int] = JudgeResult(index=idx_int, verdict=LlmVerdict.PASS)
        else:
            results[idx_int] = JudgeResult(
                index=idx_int,
                verdict=LlmVerdict.FAIL,
                detail=f"llm_verdict:{verdict_raw or 'FAIL'}",
            )

    out: List[JudgeResult] = []
    for i in sorted(expected_ids):
        if i in results:
            out.append(results[i])
        else:
            out.append(
                JudgeResult(
                    index=i,
                    verdict=LlmVerdict.FAIL,
                    detail="missing_id_in_response",
                )
            )
    return out


def llm_validate_batch(
    client: Any,
    items: Sequence[JudgeItem],
    *,
    config: Optional[ValidationConfig] = None,
) -> List[JudgeResult]:
    """Run the batched LLM judge on UNKNOWN captions.

    Args:
        client: Object with ``host``, ``model``, ``num_ctx``, ``timeout_s`` attrs
            (typically :class:`llm_client.OllamaClient`).
        items: Items to judge (each with a unique ``index``).
        config: Unused today; reserved for future prompt tuning.

    Returns:
        One :class:`JudgeResult` per input item, in index order.
    """
    del config
    if not items:
        return []

    system, user = build_judge_prompt(items)
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": min(getattr(client, "num_ctx", 4096), 4096),
            "num_predict": max(64, len(items) * 12 + 32),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    host = str(getattr(client, "host", "http://localhost:11434")).rstrip("/")
    timeout_s = float(getattr(client, "timeout_s", 300.0))
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return [
            JudgeResult(
                index=item.index,
                verdict=LlmVerdict.FAIL,
                detail=f"llm_judge_error:{exc}",
            )
            for item in items
        ]

    content = ""
    msg = raw.get("message") or {}
    if isinstance(msg, dict):
        content = str(msg.get("content") or "")
    if not content:
        return [
            JudgeResult(
                index=item.index,
                verdict=LlmVerdict.FAIL,
                detail="empty_judge_response",
            )
            for item in items
        ]

    return parse_judge_response(content, items)
