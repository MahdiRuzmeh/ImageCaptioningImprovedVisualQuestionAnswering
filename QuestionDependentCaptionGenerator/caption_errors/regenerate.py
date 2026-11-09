"""Regenerate a caption that the judge marked WRONG.

Uses the generation rules of ``llm_prompts.SYSTEM_PROMPT`` (same style as the
original captions), plus feedback: the previous caption and what was wrong
with it. The few-shot captions are NOT shown again, because the 3B model
copied them into ~900 unrelated captions ("A bus is in front of the
building.").
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_prompts import SYSTEM_PROMPT, PROMPT_VERSION as GENERATION_PROMPT_VERSION

from .judge import OllamaChat, TransportError, chat_with_retry, _strip_fences

REGEN_PROMPT_VERSION = f"r1+gen{GENERATION_PROMPT_VERSION}"

REASON_HINTS = {
    "answer_missing": "it does not state the answer",
    "answer_contradicted": "it states a different or opposite answer",
    "meaning_changed": ("it changes a word or relation of the question, or "
                        "denies something the question did not ask about"),
    "unrelated": "it is about something the question does not ask about",
    "added_facts": "it adds details that are not in the question or answer",
    "fast_validator": "it breaks the caption format rules",
}

REPLY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"captions": {"type": "array", "items": {"type": "string"}}},
    "required": ["captions"],
}


def build_messages(items: Sequence[Tuple[str, str, str, str]]) -> List[Dict[str, str]]:
    """items: (question, answer, previous_caption, reason)."""
    # The wrong caption itself is NOT shown: given "Wrong caption: X", the
    # 3B model wrote "not X" instead of a new caption for the pair.
    lines = [
        "Write a caption for each question-answer pair below. An earlier "
        "caption for each pair was rejected; the reason is given so you can "
        "avoid the same mistake.",
        "",
    ]
    for i, (q, a, _prev, reason) in enumerate(items, start=1):
        hint = REASON_HINTS.get(reason, "its meaning was wrong")
        lines += [f"{i}. Q: {q}", f"   A: {a}",
                  f"   (earlier caption rejected: {hint})"]
    lines += [
        "",
        "For each pair write ONE short declarative sentence that states exactly "
        "the answer to the question. Keep the question's own words for its "
        "subject and relation (do not swap them for other objects). If the "
        "answer is yes, state the asked fact as true. If the answer is no, "
        "state that exactly the asked fact is not true. Otherwise put the "
        "answer into the sentence. Add nothing that is not in the question or "
        "answer.",
        f'Reply as {{"captions": [...]}} with exactly {len(items)} strings, '
        f"same order.",
    ]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def _clean(cap: Any) -> Optional[str]:
    if not isinstance(cap, str):
        return None
    cap = " ".join(cap.strip().strip("\"'").split())
    cap = re.sub(r"^(caption|new caption)\s*:\s*", "", cap, flags=re.I).strip()
    if not cap:
        return None
    if cap[-1] not in ".!":
        cap += "."
    return cap[0].upper() + cap[1:]


def parse_reply(raw: str, n: int) -> List[Optional[str]]:
    text = _strip_fences(raw or "")
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e > s:
            try:
                data = json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        data = data.get("captions")
    if not isinstance(data, list) or len(data) != n:
        return [None] * n
    return [_clean(c) for c in data]


def regenerate(client: OllamaChat, items: Sequence[Tuple[str, str, str, str]], *,
               retries: int = 3, backoff_s: float = 2.0,
               temperature: float = 0.3) -> List[Optional[str]]:
    """New caption per item, or None when the reply could not be read.

    A small temperature makes round 2 differ from round 1. Raises
    :class:`TransportError` when the model can't be reached.
    """
    if not items:
        return []
    reply = chat_with_retry(client, build_messages(items),
                            num_predict=48 * len(items) + 48,
                            schema=REPLY_SCHEMA, retries=retries,
                            backoff_s=backoff_s, temperature=temperature)
    if reply.transport_failed:
        raise TransportError(reply.error)
    out = parse_reply(reply.content, len(items))
    if len(items) > 1 and all(c is None for c in out):
        # Batch reply unreadable: one item at a time.
        out = [regenerate(client, [it], retries=retries, backoff_s=backoff_s,
                          temperature=temperature)[0] for it in items]
    return out
