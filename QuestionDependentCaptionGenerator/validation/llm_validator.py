"""Batched LLM PASS/SUSPICIOUS judge for captions the fast validator marked UNKNOWN."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from validation.config import ValidationConfig
from validation.checks import (
    lexical_caption_looks_faithful,
    neither_stronger_than_no_trap,
    solitude_quantifier_trap,
)

class LlmVerdict(str, Enum):
    """Outcome from the LLM judge (never hard-drops a caption)."""

    PASS = "PASS"
    SUSPICIOUS = "SUSPICIOUS"


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
    'Each element must be {"id": <number>, "verdict": "PASS" or "SUSPICIOUS"}.'
)

_JUDGE_RULES_AND_FEW_SHOTS = (
    "Input for each item:\n"
    "- Question\n"
    "- Answer\n"
    "- Caption\n"
    "\n"
    "Label each item PASS or SUSPICIOUS.\n"
    "\n"
    "These captions were produced by a caption-generation model. Judge them "
    "with the SAME standards used for generation:\n"
    "- One short natural declarative sentence.\n"
    "- Express exactly the fact in the question and answer.\n"
    "- No invented objects, attributes, numbers, colors, or locations.\n"
    "- Natural paraphrases are allowed.\n"
    "- Numbers may appear as digits or words (2 / Two / seven for 7 / Forty for 40).\n"
    "- Answer 1 may appear as \"one\", \"a\", or \"an\".\n"
    "- Yes/no quantifiers: \"Are all …? / no\" → \"Not all …\"; "
    "\"Are both …? / no\" → \"Not both …\" (or \"One of … is not …\").\n"
    "- Kind/type questions become natural noun phrases "
    "(\"What kind of X? / Y\" → \"This is a Y X.\").\n"
    "\n"
    "Default to PASS when the caption is a faithful, grammatical restatement "
    "of the question+answer. Only mark SUSPICIOUS when there is a clear error "
    "(wrong/missing answer, hallucination, contradiction, or broken grammar).\n"
    "\n"
    "PASS if ALL of the following are true:\n"
    "1. The caption is grammatically correct and natural.\n"
    "2. The caption clearly expresses the answer.\n"
    "3. Every piece of information in the caption can be inferred ONLY from "
    "the question and answer.\n"
    "4. The caption does not invent new objects, attributes, or counts.\n"
    "\n"
    "Important PASS clarifications (do NOT mark these SUSPICIOUS):\n"
    "- Near-literal restatements of the question with the answer filled in "
    "are PASS (e.g. \"What color are the headlights? / yellow\" → "
    "\"The headlights are yellow.\").\n"
    "- Existence / any questions (\"Are there any X? / yes\" → "
    "\"There are X.\") are PASS.\n"
    "- Omitting a question noun already implied by context (for example "
    "omitting \"cake\" from \"How many layers are in this cake?\") is PASS "
    "when the answer is fully expressed and no new facts are added.\n"
    "- Action questions (\"What is/are X doing?\" / answer verb) rewritten "
    "as \"X is/are <answer>.\" are PASS.\n"
    "- Digit↔word number forms (7 / seven, 40 / Forty, 3 / Three) are PASS.\n"
    "- Natural paraphrases that express the answer and add no extra facts "
    "are PASS (for example: omitting a location phrase already in the "
    "question, using an antonym for a no-answer such as \"closed\" for "
    "\"not open\").\n"
    "- Mild wording changes (\"This is a zebra\", \"The desk is organized\") "
    "that keep the same meaning are PASS.\n"
    "\n"
    "SUSPICIOUS only if ANY of the following clearly occur:\n"
    "- Broken grammar that makes the sentence hard to read.\n"
    "- Missing or incorrect answer.\n"
    "- Hallucinated objects/attributes/counts not in the question or answer.\n"
    "- Changed meaning or contradiction with the answer.\n"
    "- Wrong subject or objects not present in the question and answer.\n"
    "\n"
    "Do NOT mark SUSPICIOUS for style alone, articles, digit/word numbers, "
    "omitted redundant nouns, or generation-style quantifier phrasing "
    "(\"Not both …\", \"Not all …\").\n"
    "If unsure between PASS and SUSPICIOUS, choose PASS.\n"
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
    "Question: How many layers are in this cake?\n"
    "Answer: 7\n"
    "Caption: There are seven layers.\n"
    "Label: PASS\n"
    "\n"
    "Example 3\n"
    "Question: What are the animals doing?\n"
    "Answer: eating\n"
    "Caption: The animals are eating.\n"
    "Label: PASS\n"
    "\n"
    "Example 4\n"
    "Question: Is the train moving?\n"
    "Answer: no\n"
    "Caption: The train is not moving.\n"
    "Label: PASS\n"
    "\n"
    "Example 5\n"
    "Question: How many flags do you see?\n"
    "Answer: 1\n"
    "Caption: There is a flag.\n"
    "Label: PASS\n"
    "\n"
    "Example 6\n"
    "Question: Are both giraffes standing?\n"
    "Answer: no\n"
    "Caption: Not both giraffes are standing.\n"
    "Label: PASS\n"
    "\n"
    "Example 7\n"
    "Question: Are all the flowers white?\n"
    "Answer: no\n"
    "Caption: Not all the flowers are white.\n"
    "Label: PASS\n"
    "\n"
    "Example 8\n"
    "Question: How many people are watching?\n"
    "Answer: 40\n"
    "Caption: Forty people are watching.\n"
    "Label: PASS\n"
    "\n"
    "Example 9\n"
    "Question: Are there any flowers on the shower curtain?\n"
    "Answer: yes\n"
    "Caption: There are flowers on the shower curtain.\n"
    "Label: PASS\n"
    "\n"
    "Example 10\n"
    "Question: What color are the headlights?\n"
    "Answer: yellow\n"
    "Caption: The headlights are yellow.\n"
    "Label: PASS\n"
    "\n"
    "Example 11\n"
    "Question: What animal is this?\n"
    "Answer: zebra\n"
    "Caption: This is a zebra.\n"
    "Label: PASS\n"
    "\n"
    "Example 12\n"
    "Question: What game is being played?\n"
    "Answer: soccer\n"
    "Caption: Two children are playing soccer.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 13\n"
    "Question: Is the dog sleeping?\n"
    "Answer: yes\n"
    "Caption: The brown dog is sleeping on the couch.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 14\n"
    "Question: How many people are there?\n"
    "Answer: 2\n"
    "Caption: Two people are smiling.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 15\n"
    "Question: What kind of weather it is?\n"
    "Answer: sunny\n"
    "Caption: The weather it is is a sunny weather it.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 16\n"
    "Question: Is this elephant all alone?\n"
    "Answer: no\n"
    "Caption: Not all the elephants are alone.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 17\n"
    "Question: Is the tennis net purple?\n"
    "Answer: no\n"
    "Caption: The green fence is not purple.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 18\n"
    "Question: Did one of the guys just take off his shirt?\n"
    "Answer: no\n"
    "Caption: One of the guys is not shirtless.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 19\n"
    "Question: Are both men smiling?\n"
    "Answer: no\n"
    "Caption: Neither man is smiling.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 20\n"
    "Question: Does the weather appear to be snowy?\n"
    "Answer: yes\n"
    "Caption: Snow appears on the ground.\n"
    "Label: SUSPICIOUS\n"
    "\n"
    "Example 21\n"
    "Question: Are they standing in a mud puddle?\n"
    "Answer: no\n"
    "Caption: Not both giraffes are standing.\n"
    "Label: SUSPICIOUS\n"
    "\n"
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
        f'Return a JSON array of exactly {len(items)} objects with keys '
        f'"id" and "verdict" (PASS or SUSPICIOUS).'
    )
    return _JUDGE_SYSTEM_PROMPT, "\n".join(lines)


def parse_judge_response(
    raw: str,
    items: Sequence[JudgeItem],
) -> List[JudgeResult]:
    """Parse model JSON array into per-item :class:`JudgeResult`.

    Fail-closed toward SUSPICIOUS: parse errors or missing ids → SUSPICIOUS
    (caption is kept and logged, never hard-dropped by the judge).
    """
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")
    expected_ids = {item.index for item in items}
    results: Dict[int, JudgeResult] = {}

    if start < 0 or end <= start:
        return [
            JudgeResult(
                index=i,
                verdict=LlmVerdict.SUSPICIOUS,
                detail="parse_no_json_array",
            )
            for i in sorted(expected_ids)
        ]

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return [
            JudgeResult(
                index=i,
                verdict=LlmVerdict.SUSPICIOUS,
                detail=f"parse_json_error:{exc}",
            )
            for i in sorted(expected_ids)
        ]

    if not isinstance(data, list):
        return [
            JudgeResult(
                index=i,
                verdict=LlmVerdict.SUSPICIOUS,
                detail="parse_not_a_list",
            )
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
            # Treat legacy FAIL and any other token as SUSPICIOUS.
            results[idx_int] = JudgeResult(
                index=idx_int,
                verdict=LlmVerdict.SUSPICIOUS,
                detail=f"llm_verdict:{verdict_raw or 'SUSPICIOUS'}",
            )

    out: List[JudgeResult] = []
    for i in sorted(expected_ids):
        if i in results:
            out.append(results[i])
        else:
            out.append(
                JudgeResult(
                    index=i,
                    verdict=LlmVerdict.SUSPICIOUS,
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
        items: Items to judge (each with a unique ``index`` — may be global
            row ids; they are remapped to ``0..n-1`` for the model prompt).
        config: Unused today; reserved for future prompt tuning.

    Returns:
        One :class:`JudgeResult` per input item, preserving the original
        ``index`` values from ``items``.
    """
    del config
    if not items:
        return []

    # Remap to dense 0..n-1 ids so the model (and JSON parse) stay reliable
    # when callers pass sparse global row indices.
    local_items = [
        JudgeItem(
            index=i,
            question=item.question,
            answer=item.answer,
            caption=item.caption,
        )
        for i, item in enumerate(items)
    ]
    original_indices = [item.index for item in items]

    system, user = build_judge_prompt(local_items)
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
                index=orig,
                verdict=LlmVerdict.SUSPICIOUS,
                detail=f"llm_judge_error:{exc}",
            )
            for orig in original_indices
        ]

    content = ""
    msg = raw.get("message") or {}
    if isinstance(msg, dict):
        content = str(msg.get("content") or "")
    if not content:
        return [
            JudgeResult(
                index=orig,
                verdict=LlmVerdict.SUSPICIOUS,
                detail="empty_judge_response",
            )
            for orig in original_indices
        ]

    local_results = parse_judge_response(content, local_items)
    by_local = {jr.index: jr for jr in local_results}
    out: List[JudgeResult] = []
    for local_i, orig in enumerate(original_indices):
        item = items[local_i]
        jr = by_local.get(local_i)
        if jr is None:
            verdict = LlmVerdict.SUSPICIOUS
            detail = "missing_id_in_response"
        else:
            verdict = jr.verdict
            detail = jr.detail

        # Small-model safety net: recover false SUSPICIOUS when the caption is
        # a high-precision lexical restatement of Q+A; and demote false PASS
        # on known meaning traps (e.g. all-alone → Not all … alone).
        if solitude_quantifier_trap(item.question, item.answer, item.caption):
            verdict = LlmVerdict.SUSPICIOUS
            detail = "solitude_quantifier_trap"
        elif neither_stronger_than_no_trap(
            item.question, item.answer, item.caption
        ):
            verdict = LlmVerdict.SUSPICIOUS
            detail = "neither_stronger_than_no"
        elif verdict == LlmVerdict.SUSPICIOUS and lexical_caption_looks_faithful(
            item.question, item.answer, item.caption
        ):
            verdict = LlmVerdict.PASS
            detail = "lexical_override_pass"

        out.append(JudgeResult(index=orig, verdict=verdict, detail=detail))
    return out
