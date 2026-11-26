"""Regenerate a caption that the judge marked WRONG.

Uses the generation rules of ``llm_prompts.SYSTEM_PROMPT`` (same style as the
original captions), plus what was wrong with the earlier caption. The
few-shot captions are NOT shown again, because the 3B model copied them into
~900 unrelated captions ("A bus is in front of the building.").

Captions are asked for in packs (``regenerate_checked``, default 5 per
request, like generation) and every new caption is checked against its pack;
rejects are asked again one at a time.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_prompts import SYSTEM_PROMPT, PROMPT_VERSION as GENERATION_PROMPT_VERSION
from validation.checks import has_spurious_negation
from validation.fast_validator import fast_validate

from .judge import OllamaChat, TransportError, chat_with_retry, _strip_fences

REGEN_PROMPT_VERSION = f"r2+gen{GENERATION_PROMPT_VERSION}"

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


# ---------------------------------------------------------------- packed + checked
def check_caption(question: str, answer: str, caption: Optional[str], *,
                  batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
                  batch_captions: Optional[Sequence[Optional[str]]] = None,
                  self_index: int = -1) -> Optional[str]:
    """Reject reason for a new caption, or None if it may go to the judge.

    Same hard rejects as generation (``fast_validate``: format, echo,
    polarity, quantifier and, with the pack as context, ``batch_contamination``
    = caption belongs to another pair of the pack), plus ``spurious_negation``:
    a "not" that a non-yes/no answer never implied ("What color tie? / green"
    -> "The man does not have a green tie.").
    """
    if not caption:
        return "unparsed"
    fast = fast_validate(question, answer, caption, batch_pairs=batch_pairs,
                         batch_captions=batch_captions, self_index=self_index)
    if fast.is_fail:
        return fast.reasons[0] if fast.reasons else "fast_fail"
    if has_spurious_negation(answer, caption, question):
        return "spurious_negation"
    return None


def regenerate_checked(client: OllamaChat,
                       items: Sequence[Tuple[str, str, str, str]], *,
                       batch_size: int = 5, single_fallback: bool = True,
                       retries: int = 3, backoff_s: float = 2.0,
                       temperature: float = 0.3,
                       ) -> List[Tuple[Optional[str], Optional[str]]]:
    """Regenerate in packs of ``batch_size``, then redo rejects one by one.

    Mirrors ``llm_client.OllamaClient.captions_with_retry``: a packed
    request first, each caption checked against its pack
    (:func:`check_caption`), and every rejected (or unreadable) item asked
    again on its own when ``single_fallback`` is on.

    Returns per item ``(caption, None)`` when the caption passed the checks,
    or ``(caption_or_None, reject_reason)``.
    """
    kw = dict(retries=retries, backoff_s=backoff_s, temperature=temperature)
    out: List[Tuple[Optional[str], Optional[str]]] = []
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        pack = list(items[start:start + size])
        caps = regenerate(client, pack, **kw)
        pairs = [(q, a) for q, a, _p, _r in pack]
        res: List[Tuple[Optional[str], Optional[str]]] = []
        for i, ((q, a, _p, _r), cap) in enumerate(zip(pack, caps)):
            ctx = dict(batch_pairs=pairs, batch_captions=caps, self_index=i) \
                if len(pack) > 1 else {}
            res.append((cap, check_caption(q, a, cap, **ctx)))
        if single_fallback and len(pack) > 1:
            for i, (cap, why) in enumerate(res):
                if why is None:
                    continue
                q, a, prev, reason = pack[i]
                single = regenerate(client, [pack[i]], **kw)[0]
                why1 = check_caption(q, a, single)
                if why1 is None or single:
                    res[i] = (single, why1)
        out += res
    return out
