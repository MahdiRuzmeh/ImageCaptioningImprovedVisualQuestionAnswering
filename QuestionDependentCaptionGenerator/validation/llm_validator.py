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


def build_judge_prompt(items: Sequence[JudgeItem]) -> Tuple[str, str]:
    """Build system + user messages for a batched judge call.

    Each item is numbered independently; no cross-item context is shared.
    """
    system = (
        "You are a strict caption validator. Reply with ONLY a JSON array. "
        "Each element must be {\"id\": <number>, \"verdict\": \"PASS\" or \"FAIL\"}."
    )
    lines = [
        "For each numbered item below, return PASS only if the CAPTION correctly "
        "expresses the ANSWER to the QUESTION and adds no unsupported facts. "
        "Otherwise return FAIL.\n"
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
    return system, "\n".join(lines)


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
