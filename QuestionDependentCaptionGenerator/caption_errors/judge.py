"""Semantic caption judge: is a caption WRONG for its question + answer?

Text only (question, answer, caption); no image. Grammar is judged separately
from meaning so that captions with only wording problems are NOT regenerated.

Verdicts (exact strings, written to the captions JSON):

* ``ok``                  - states exactly the answer to the question
* ``GRAMMAR_ISSUE_ONLY``  - meaning is right, wording is bad
* ``WRONG``               - meaning is wrong (see ``REASONS``)

``verdict_reason`` is one of ``REASONS`` for WRONG, else ``None``.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bumped whenever the judge prompt / few-shots change (stored in progress
# headers: a resumed run refuses to mix two prompt versions).
JUDGE_PROMPT_VERSION = "j1"

OK = "ok"
GRAMMAR = "GRAMMAR_ISSUE_ONLY"
WRONG = "WRONG"
VERDICTS = (OK, GRAMMAR, WRONG)

REASONS = ("answer_missing", "answer_contradicted", "meaning_changed",
           "unrelated", "added_facts")

# Replies that mean "no reply from the model" (retry; never a verdict).
TRANSPORT = ("connection_error", "http_error", "timeout", "http_json_error",
             "empty_response", "exception")


# ---------------------------------------------------------------- prompt
_SYSTEM = (
    "You check captions generated from visual question-answer pairs. "
    "Each caption must state, as a plain sentence, exactly the fact given by "
    "its question and answer. You judge MEANING first and grammar second. "
    "Reply with JSON only."
)

_RULES = """\
For each item you get a QUESTION about an image, its ANSWER, and a CAPTION.
Decide whether the CAPTION states exactly the answer to the question.

Verdicts:
- "ok": the caption states the answer to the question, with the same meaning.
- "GRAMMAR_ISSUE_ONLY": the meaning is right, but the sentence has a grammar or
  wording mistake (wrong article, wrong plural, odd word form).
- "WRONG": the meaning is wrong. Use one reason:
  - "answer_missing": the caption does not state the answer.
  - "answer_contradicted": the caption states a different or opposite answer.
  - "meaning_changed": the answer is there, but a word of the question was
    replaced (e.g. "wall" became "door"), the relation was reversed, or the
    negation denies something the question did not ask about.
  - "unrelated": the caption talks about something the question is not about.
  - "added_facts": the answer is right, but the caption adds objects, places,
    colors, numbers or reasons that are not in the question or answer.

These are "ok" (do NOT mark them WRONG):
- Numbers as digits or words (7 / seven). Answer 1 as "a" / "an" / "one".
- Answer 0 or "none" written as "no X" / "There are no X".
- A "no" answer written with an opposite word: "Is the door open? / no" ->
  "The door is closed."
- Natural paraphrases, and leaving out a question noun that is only context,
  as long as the asked fact is still stated.
- Yes/no questions: "yes" -> the caption affirms the asked fact; "no" -> the
  caption denies exactly the asked fact.
Only meaning decides between ok and WRONG. Grammar never makes a caption WRONG.
When the meaning is right and only wording is off, use "GRAMMAR_ISSUE_ONLY".
"""

# (question, answer, caption, verdict, reason)
FEW_SHOTS: List[Tuple[str, str, str, str, Optional[str]]] = [
    ("How many plates are on the table?", "2",
     "There are two plates on the table.", OK, None),
    ("How many boats are visible?", "0", "No boats are visible.", OK, None),
    ("Is the window open?", "no", "The window is closed.", OK, None),
    ("What is the woman holding?", "umbrella",
     "The woman is holding an umbrella.", OK, None),
    ("What are the horses doing?", "grazing", "The horses are grazing.",
     OK, None),
    ("Is anybody on the bench?", "yes", "Someone is on the bench.", OK, None),
    ("How many chairs in all are in the room?", "4",
     "There are four chairs in alls.", GRAMMAR, None),
    ("What game is being played?", "tennis", "A tennis is being played.",
     GRAMMAR, None),
    ("Does this cat chase mice?", "yes", "This cat chase mice.", GRAMMAR, None),
    ("What is hanging on the wall?", "clock",
     "A clock is hanging on the door.", WRONG, "meaning_changed"),
    ("Is the man, in red, running?", "no", "The man is not in red.",
     WRONG, "meaning_changed"),
    ("Is the oven door open or closed?", "open", "The oven door is closed.",
     WRONG, "answer_contradicted"),
    ("What animal is in the field?", "sheep",
     "No animal is in the field.", WRONG, "answer_contradicted"),
    ("What is on the shelf?", "books", "Something is on the shelf.",
     WRONG, "answer_missing"),
    ("Do these dogs like water?", "yes", "A yes is shown for these dogs.",
     WRONG, "answer_missing"),
    ("Is this a beach?", "yes", "A bus is in front of the building.",
     WRONG, "unrelated"),
    ("Is the boy smiling?", "no", "The sky is cloudy.", WRONG, "unrelated"),
    ("What color is the truck?", "blue",
     "The blue pickup truck is parked near the house.", WRONG, "added_facts"),
    ("What is covering the table?", "nothing",
     "The table is not covered, so the wood can be seen.", WRONG,
     "added_facts"),
]


def _item_block(i: int, q: str, a: str, c: str) -> str:
    return f"{i}. QUESTION: {q}\n   ANSWER: {a}\n   CAPTION: {c}"


def _reply_for(rows: Sequence[Tuple[str, Optional[str]]]) -> str:
    return json.dumps({"items": [
        {"id": i, "verdict": v, "reason": r or "none"}
        for i, (v, r) in enumerate(rows, start=1)]})


def build_messages(items: Sequence[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    """Chat messages for one judge request; items numbered from 1."""
    ex_items = "\n".join(_item_block(i, q, a, c) for i, (q, a, c, _v, _r)
                         in enumerate(FEW_SHOTS, start=1))
    ex_reply = _reply_for([(v, r) for _q, _a, _c, v, r in FEW_SHOTS])
    lines = [
        _RULES, "Example items:", ex_items, "", "Example reply:", ex_reply, "",
        "Now judge these items. Each item is independent.", "",
        "\n".join(_item_block(i, q, a, c)
                  for i, (q, a, c) in enumerate(items, start=1)),
        "",
        f'Reply with {{"items": [...]}} holding exactly {len(items)} objects '
        f'(ids 1..{len(items)}) with keys "id", "verdict" (ok, '
        f'GRAMMAR_ISSUE_ONLY or WRONG) and "reason" ({", ".join(REASONS)}, '
        f'or none when the verdict is not WRONG).',
    ]
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


# Ollama structured output: forces the reply into this shape.
REPLY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "reason": {"type": "string", "enum": list(REASONS) + ["none"]},
        },
        "required": ["id", "verdict", "reason"]}}},
    "required": ["items"],
}


# ---------------------------------------------------------------- parse
_VERDICT_ALIASES = {
    "ok": OK, "pass": OK, "correct": OK,
    "grammar_issue_only": GRAMMAR, "grammar_only": GRAMMAR, "grammar": GRAMMAR,
    "grammar_issue": GRAMMAR,
    "wrong": WRONG, "fail": WRONG, "incorrect": WRONG,
}


def normalize_verdict(value: Any) -> Optional[str]:
    key = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    return _VERDICT_ALIASES.get(key)


def normalize_reason(value: Any) -> Optional[str]:
    key = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    return key if key in REASONS else None


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.I)
    return m.group(1).strip() if m else t


def parse_reply(raw: str, n: int) -> List[Optional[Tuple[str, Optional[str]]]]:
    """Model reply -> per-item ``(verdict, reason)`` or ``None`` (unreadable).

    Accepts ``{"items": [...]}`` or a bare array. Items are matched by ``id``
    (1..n); if ids are missing but the count matches, by position. A WRONG
    without a valid reason gets ``meaning_changed``; a reason on a non-WRONG
    verdict is dropped.
    """
    out: List[Optional[Tuple[str, Optional[str]]]] = [None] * n
    text = _strip_fences(raw or "")
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            s, e = text.find(open_c), text.rfind(close_c)
            if s != -1 and e > s:
                try:
                    data = json.loads(text[s:e + 1])
                    break
                except json.JSONDecodeError:
                    continue
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list):
        return out

    def one(entry: Any) -> Optional[Tuple[str, Optional[str]]]:
        if not isinstance(entry, dict):
            return None
        v = normalize_verdict(entry.get("verdict"))
        if v is None:
            return None
        if v != WRONG:
            return v, None
        return v, normalize_reason(entry.get("reason")) or "meaning_changed"

    by_id: Dict[int, Any] = {}
    for entry in data:
        if isinstance(entry, dict):
            try:
                by_id.setdefault(int(entry.get("id")), entry)
            except (TypeError, ValueError):
                pass
    if by_id and all(1 <= k <= n for k in by_id):
        for k, entry in by_id.items():
            out[k - 1] = one(entry)
    elif len(data) == n:
        out = [one(e) for e in data]
    return out


# ---------------------------------------------------------------- ollama
@dataclass
class ChatReply:
    content: str = ""
    error: str = ""          # "" or one of TRANSPORT (+ ":" detail)

    @property
    def transport_failed(self) -> bool:
        return self.error.startswith(TRANSPORT)


class OllamaChat:
    """Minimal /api/chat client (temperature 0, optional JSON schema)."""

    def __init__(self, host: str, model: str, *, num_ctx: int = 4096,
                 timeout_s: float = 180.0, temperature: float = 0.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]], *, num_predict: int,
             schema: Optional[Dict[str, Any]] = None,
             temperature: Optional[float] = None) -> ChatReply:
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature if temperature is None
                        else temperature,
                        "num_ctx": self.num_ctx, "num_predict": num_predict},
        }
        if schema is not None:
            payload["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return ChatReply(error=f"http_error:{exc.code}")
        except urllib.error.URLError as exc:
            return ChatReply(error=f"connection_error:{exc.reason}")
        except TimeoutError:
            return ChatReply(error="timeout")
        except json.JSONDecodeError:
            return ChatReply(error="http_json_error")
        except Exception as exc:   # connection reset while reading, ...
            return ChatReply(error=f"exception:{type(exc).__name__}:{exc}")
        content = str((raw.get("message") or {}).get("content") or "")
        if not content:
            return ChatReply(error=f"empty_response:{raw.get('error')!r}")
        return ChatReply(content=content)


def chat_with_retry(client: OllamaChat, messages: List[Dict[str, str]], *,
                    num_predict: int, schema: Optional[Dict[str, Any]],
                    retries: int, backoff_s: float,
                    temperature: Optional[float] = None) -> ChatReply:
    """Retry only transport failures (no reply). A real reply is returned."""
    reply = ChatReply(error="exception:not_sent")
    for attempt in range(retries + 1):
        reply = client.chat(messages, num_predict=num_predict, schema=schema,
                            temperature=temperature)
        if not reply.transport_failed:
            return reply
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt))
    return reply


class TransportError(RuntimeError):
    """The model never replied (after retries). Nothing must be recorded."""


def judge_items(client: OllamaChat, items: Sequence[Tuple[str, str, str]], *,
                retries: int = 3, backoff_s: float = 2.0,
                ) -> List[Optional[Tuple[str, Optional[str]]]]:
    """Judge a batch. Unreadable items are retried one by one, once.

    Returns per item ``(verdict, reason)`` or ``None`` (still unreadable:
    the caller leaves the row unjudged). Raises :class:`TransportError` when
    the model can't be reached.
    """
    if not items:
        return []

    def ask(batch: Sequence[Tuple[str, str, str]]):
        reply = chat_with_retry(client, build_messages(batch),
                                num_predict=40 * len(batch) + 64,
                                schema=REPLY_SCHEMA, retries=retries,
                                backoff_s=backoff_s)
        if reply.transport_failed:
            raise TransportError(reply.error)
        return parse_reply(reply.content, len(batch))

    results = ask(items)
    for i, res in enumerate(results):
        if res is None:
            results[i] = ask([items[i]])[0]
    return results
