"""Binary filter for questions that are not directly answerable from a static image.

Dataset generation always labels questions:

    DIRECTLY_VISUAL | NOT_DIRECTLY_VISUAL

Cascade (v17+):

1. **Fast checker (blacklist)** — high-precision ``_NON_VISUAL_CANDIDATE_RE``
   match → ``NOT_DIRECTLY_VISUAL`` (``visual_filter_source=blacklist``), no LLM.
2. **Else UNKNOWN** — batched into ``classify_batch``; Qwen returns
   ``NEEDS_OCR`` / ``NEEDS_KNOWLEDGE`` / ``NEEDS_OPINION`` / ``VISUAL``,
   mapped to binary labels (``visual_filter_source=llm_classifier``).

Only the LLM asserts ``DIRECTLY_VISUAL``. Soft / ambiguous cues are left
UNKNOWN so the LLM can rescue visible cases (e.g. ``old man wearing glasses``).

``DIRECTLY_VISUAL`` means a human could reasonably answer by looking at the
image alone. ``NOT_DIRECTLY_VISUAL`` means answering needs rendered text
(OCR), personal opinion/preference, or external factual knowledge
unavailable from appearance.

``generate.py`` always constructs a ``QuestionClassifier`` (Ollama). The
offline ``--drop-subjective-candidates`` regex gate remains available on
:func:`filter_non_visual_questions` for tests, but the CLI ignores it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

CLASSIFIER_PROMPT_VERSION = "v18c_visual_rescue"

QUESTION_LABELS = (
    "DIRECTLY_VISUAL",
    "NOT_DIRECTLY_VISUAL",
)

# Four-way confirmation tokens returned by the LLM; mapped to binary labels.
CONFIRM_LABELS = (
    "NEEDS_OCR",
    "NEEDS_KNOWLEDGE",
    "NEEDS_OPINION",
    "VISUAL",
)

# Provenance of a DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL decision, stored per
# row as ``visual_filter_source`` for later error analysis.
VISUAL_FILTER_BLACKLIST = "blacklist"
VISUAL_FILTER_LLM = "llm_classifier"
# Legacy strings kept for reading older audit / output JSON only.
VISUAL_FILTER_FAST_PATH = "fast_path"
VISUAL_FILTER_DEFAULT = "default_visual"

# Offline / candidate-drop regex (broadened beyond the old subjective-only gate).
_CANDIDATE_RE = re.compile(
    r"""
    \bhave\s+you\s+ever\b |
    \bwould\s+you\s+(prefer|like|want)\b |
    \bdo\s+you\s+(like|think|want|prefer)\b |
    \bwould\s+you\b |
    \bdo\s+you\b |
    \bare\s+you\s+allowed\b |
    \b(safe|healthy|nutritious|beautiful|comfortable|dangerous|
       expensive|valuable|tasty|delicious|attractive|ugly|
       endangered|personality|professional)\b |
    \bwhat\s+is\s+the\s+name\b |
    \bwhat\s+(?:does|do)\s+.+\bsay\b |
    \bwhat\s+is\s+written\b |
    \bdoes\s+this\s+\w+\s+work\b
    """,
    re.I | re.X,
)

_SYSTEM_PROMPT = (
    "You confirm whether a VQA question needs something beyond looking at "
    "the image.\n"
    "\n"
    "Return ONLY one of these labels:\n"
    "\n"
    "VISUAL — default. Prefer VISUAL whenever the answer is visible in the "
    "pixels. Includes: object recognition; visible actions (doing, eating, "
    "throwing, playing, holding); attributes and clothing (including "
    "wearing glasses = eyewear); counting / \"how many\"; existence "
    "\"is there\"; relative comparisons (\"same size\", \"same color\"); "
    "contents of containers by appearance (\"what's in the glass/plate\" — "
    "not reading a printed label); common category labels (\"what kind of "
    "animal/food/room\" when the answer is cat/dog/pizza/kitchen, not a "
    "breed or brand); scene/sport/room type; lighting / day-vs-night; "
    "weather; B/W vs sepia; apparent sex/gender from appearance; WHERE a "
    "plate/sign is or WHETHER text is present (without reading it).\n"
    "\n"
    "NEEDS_OCR — only reading or decoding rendered text, digits, logos, "
    "brand names, signs, train/bus numbers, license-plate digits, letter "
    "glyphs, or exact clock/watch time. Do NOT use OCR for drinks/food/"
    "objects by appearance, or for eyeglasses/eyewear.\n"
    "\n"
    "Day vs night or time of day FROM LIGHTING OR SHADOWS is VISUAL, not OCR.\n"
    "\n"
    "NEEDS_KNOWLEDGE — external facts unavailable from appearance (breed/"
    "species taxonomy, manufacturer, country of a flag, animal sounds, "
    "price, designed-for purpose, digital/official, free-range, tourists, "
    "organic, named place, distance, regional geography). NOT knowledge: "
    "\"what kind of animal\" meaning cat/dog; \"are they eating\"; "
    "counting; relative same-size comparisons; apparent sex from look.\n"
    "\n"
    "NEEDS_OPINION — preference, emotion, absolute guessed age/size "
    "(\"how old\", \"how big\"), social relationships, condition/nutrition "
    "judgments. NOT opinion: \"what is the person doing\" or other visible "
    "actions.\n"
)

_FEW_SHOT_BLOCK = (
    "Examples (contrastive pairs — note the difference):\n"
    "Q: What brand is shown?\n"
    "NEEDS_OCR\n"
    "Q: What's in the glass?\n"
    "VISUAL\n"
    "Q: What word is written?\n"
    "NEEDS_OCR\n"
    "Q: Is she wearing glasses?\n"
    "VISUAL\n"
    "Q: What license plate number?\n"
    "NEEDS_OCR\n"
    "Q: Where is the license plate located?\n"
    "VISUAL\n"
    "Q: What time is it?\n"
    "NEEDS_OCR\n"
    "Q: Can you tell what time of day it is by the shadow?\n"
    "VISUAL\n"
    "Q: What breed is this dog?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: What kind of animal is in the picture?\n"
    "VISUAL\n"
    "Q: What sound does this animal make?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: Are the animals eating?\n"
    "VISUAL\n"
    "Q: How big is the sandwich?\n"
    "NEEDS_OPINION\n"
    "Q: Are both of these animals the same size?\n"
    "VISUAL\n"
    "Q: How old is animal?\n"
    "NEEDS_OPINION\n"
    "Q: What is this person doing?\n"
    "VISUAL\n"
    "Q: Is this person sad?\n"
    "NEEDS_OPINION\n"
    "Q: What is the person doing?\n"
    "VISUAL\n"
    "Q: What is the sex of the player?\n"
    "VISUAL\n"
    "Q: Who manufactured this?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: What country is this flag from?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: What is the price?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: What mountain was this taken at?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: Are these animals native to Iceland?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: Is there a Wal-Mart within a mile of this place?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: Is the pizza sauce organic?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: Does this train work?\n"
    "NEEDS_KNOWLEDGE\n"
    "Q: Would you eat this?\n"
    "NEEDS_OPINION\n"
    "Q: Do you like this?\n"
    "NEEDS_OPINION\n"
    "Q: Is this beautiful?\n"
    "NEEDS_OPINION\n"
    "Q: Do this man and woman know each other?\n"
    "NEEDS_OPINION\n"
    "Q: What is the numbers of the train?\n"
    "NEEDS_OCR\n"
    "Q: The plane's stand resembles what letter?\n"
    "NEEDS_OCR\n"
    "Q: What is the name of the hotel?\n"
    "NEEDS_OCR\n"
    "Q: How many hot dogs are there?\n"
    "VISUAL\n"
    "Q: Can you count all of the mice?\n"
    "VISUAL\n"
    "Q: Is there a banana on the plate?\n"
    "VISUAL\n"
    "Q: What is he wearing?\n"
    "VISUAL\n"
    "Q: What sport is being played?\n"
    "VISUAL\n"
    "Q: What kind of room is this?\n"
    "VISUAL\n"
    "Q: What is the name of the small round green fruit next to the apple?\n"
    "VISUAL\n"
    "Q: Are they playing polo?\n"
    "VISUAL\n"
    "Q: Is this a library or professional office?\n"
    "VISUAL\n"
    "Q: Is there text in the top right corner of this picture?\n"
    "VISUAL\n"
    "Q: Was this picture taken at night?\n"
    "VISUAL\n"
    "Q: Do you find meat in the salad?\n"
    "VISUAL"
)

_USER_PROMPT_INTRO = (
    "Confirm whether this VQA question needs OCR, external knowledge, or "
    "personal opinion beyond looking at the image.\n"
    "Return ONLY one label: NEEDS_OCR, NEEDS_KNOWLEDGE, NEEDS_OPINION, or "
    "VISUAL.\n"
    "Prefer VISUAL whenever the answer is visible in the pixels "
    "(actions, counting, clothing/eyewear, sport, room, contents of a "
    "glass, kind of animal vs breed, same size vs how big, apparent sex).\n"
    "Do NOT invent labels such as NEEDS_COUNTING — use VISUAL for counts.\n"
)

_USER_PROMPT_TEMPLATE = (
    _USER_PROMPT_INTRO
    + "\n"
    + _FEW_SHOT_BLOCK
    + "\n\nQ: {question}"
)


def _build_batch_user_prompt(questions: Sequence[str]) -> str:
    """Pack numbered questions into one user prompt (JSON-array labels)."""
    lines: List[str] = [
        "Confirm whether each VQA question needs OCR, external knowledge, "
        "or personal opinion beyond looking at the image.",
        "Return ONLY a JSON array of label strings: NEEDS_OCR, "
        "NEEDS_KNOWLEDGE, NEEDS_OPINION, or VISUAL.",
        "Prefer VISUAL whenever the answer is visible in the pixels "
        "(actions, counting, clothing/eyewear, sport, room, contents of a "
        "glass, kind of animal vs breed, same size vs how big, apparent sex).",
        "Do NOT invent labels such as NEEDS_COUNTING — use VISUAL for counts.",
        "",
        _FEW_SHOT_BLOCK,
        "",
        "Now classify the questions below.",
        "Return ONLY a JSON array of label strings "
        f"(length {len(questions)}, same order, no keys, no extra text):",
        "",
    ]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. Q: {q}")
    lines.append("")
    lines.append("JSON array:")
    return "\n".join(lines)


# Fast checker: high-precision auto-NDV. Match → NOT_DIRECTLY_VISUAL with no LLM.
# Soft / ambiguous cues (bare old/young, emotion words, intention "trying to",
# bare "text"/"license", cold/warm, general "what kind of room/shoes") stay
# UNKNOWN → batched LLM.
#
# Note: ``made of`` is intentionally NOT listed (visible material) while
# ``who made`` is (maker/brand knowledge).
_NON_VISUAL_CANDIDATE_RE = re.compile(
    r"""
    # --- personal preference / opinion (high precision) ---
    \bhave\s+you\s+ever\b |
    \b(?:do|would|did|have|can|could)\s+you\b |
    \bdo\s+we\b | \bwould\s+one\b | \byour\b | \bprefer\b | \bfavorite\b |
    \bwhose\b |
    \bhow\s+(?:old|big|small|large|tall|heavy|long|wide)\b |
    \bknow\s+each\s+other\b |
    \blooking\s+forward\b |
    \b(?:do|does|did)\s+(?:the|a|an|he|she|they|this|that|these|those|
        his|her|their|people|dogs?|kids?|children|man|woman|person)\s+
        \w*(?:\s+\w+){0,3}\s+like\b |
    \bin\s+(?:good|bad|poor)\s+shape\b |
    \b(?:safe|safety|healthy|nutritious|tasty|delicious|beautiful|ugly|
       attractive|comfortable|dangerous|expensive|valuable|cheap|personality|
       romantic|scared|afraid|
       protein|calorie|carb|edible|sugar|vitamin|homemade|vegan)\b |
    \bnative\s+to\b |
    \bsmall\s+(?:town|city|village)\b |
    \bbig\s+event\b |
    \bbalanced\s+meal\b |

    # --- OCR / reading rendered text or digits (high precision) ---
    \bsays?\b | \bsaying\b | \bwritten\b | \bprinted\b | \bspelled\b |
    \b(?:words?|letters?|initials|caption|slogan)\b |
    \bname\s+on\b | \bwhat\s+name\s+is\s+on\b | \bname\s+is\s+on\b |
    \bwhat\s+is\s+the\s+name\s+of\s+the\s+(?:hotel|restaurant|store|shop|
        company|team|street|building)\b |
    \bnamed\b | \bbrand\b | \blogo\b |
    \bcompany\b | \badvertis\w*\b | \bmentioned\b | \blanguage\b |
    \blisted\s+on\b |
    \bwhat\s+time\b | \b(?:month|year|date)\b |
    \blicense\s+plate\s+numbers?\b | \bplate\s+numbers?\b |
    \bphone\s+number\b | \bwebsite\b | \bscore\b |
    \bwhat\s+(?:is|are)\s+the\s+numbers?\b |
    \bwhat\s+number\s+(?:bus|train|plane|flight|truck|taxi|jersey|shirt|
        uniform|player)\b |
    \bwhich\s+number\b |
    \bwhat\s+numbers?\s+(?:can\s+be\s+seen|are\s+(?:on|visible|shown))\b |
    \bwhat\s+is\s+the\s+number\s+of\s+(?:the\s+)?
        (?:bus|train|plane|flight|truck|taxi|jersey|shirt|uniform)\b |
    \bnumbers?\s+of\s+the\s+(?:train|bus|plane|truck|car|jersey|shirt)\b |
    \b(?:train|bus|jersey|shirt|gate|room)\s+numbers?\b |
    \bbus\s+number\b |

    # --- outside-world knowledge ---
    \ballowed\b | \blegal\b | \brules?\b | \bendangered\b |
    \b(?:breed|species)\b |
    \bwhat\s+(?:kind|type)\s+of\s+(?:dog|cat|breed)\b |
    \bsound\s+(?:does|might|is|can|would)\b | \bwhat\s+sound\b |
    \bwho\s+(?:made|makes|built|owns|invented)\b | \bmanufactur\w*\b |
    \bcost\b | \bprice\b |
    \bpopular\b | \bfamous\b |
    \bdesigned\s+for\b | \bdigital\b | \bofficial\b |
    \bfree[-\s]?range\b | \btourists?\b | \borganic\b |
    \bpurpos |                                        # purpose / purposes / on purpose
    \bwork(?:s|ing)?\s*\?*\s*$ |
    \bwhat\s+will\s+happen\b | \bgoing\s+to\s+happen\b |
    \bantique\b | \bmiles?\b | \bwal[-\s]?mart\b | \bmidwest\b |
    \bamerican\s+flag\b | \bno\s+parking\b |
    \bairline\b | \bclimate\b | \bhumid\b |
    \bspecialize\b | \bculinary\b |
    \balarm\s+set\b |
    \bbreak\s+if\s+dropped\b |
    \bbrothers?\b | \bhigh\s+school\b |
    \bintersection\b |

    # --- hypothetical / counterfactual ---
    \bwould\s+(?:this|that|it|the|a|an)\b |

    # --- geography / place identity (outside the pixels) ---
    \bcountry\b | \bnation\w*\b | \bcontinent\b |
    \bwhich\s+part\s+of\s+the\s+world\b |
    \btaken\s+(?:at|in)\b |
    \bwhat\s+(?:mountain|lake|river|street|beach|park)\b |
    \bchinatown\b | \bsan\s+francisco\b | \bbrisbane\b | \balaska\b |

    # --- non-visual senses ---
    \bsquishy\b | \bsmell\w*\b | \btaste\w*\b | \bloud\b |
    \btemperature\b | \bsoft\s+to\s+the\s+touch\b
    """,
    re.I | re.X,
)

# Backward-compatible alias for older imports / tests.
_NON_VISUAL_SUSPECT_RE = _NON_VISUAL_CANDIDATE_RE


# Phrasings that would trip a hard-NDV marker while describing perception.
# Stripped before the blacklist test ("Do you see a boat?" stays UNKNOWN→LLM).
_SUSPECT_EXEMPT_RE = re.compile(
    r"""
    \b(?:can|could|do|did|would)\s+you\s+
        (?:see|spot|find|count|tell|make\s+out)\b |
    \b(?:can|could)\s+be\s+seen\b |
    \bwhat\s+time\s+of\s+(?:day|year)\b |
    \btaken\s+at\s+night\b |
    \btaken\s+during\s+the\s+day\b |
    \bcaptured\s+at\s+night\b |
    \bat\s+night\b |
    \bnighttime\b |
    \bdaytime\b |
    \bday\s+or\s+night\b |
    \bnext\s+to\b |
    \b(?:to|on)\s+the\s+right\b |
    \bright\s+side\b |
    \btrash\s+can\b |
    \bcity\s+bus(?:es)?\b |
    \blook(?:s|ing)?\s+like\b |
    \bblack\s+and\s+white\b | \bsepia\b |
    \bdo\s+you\s+think\b.{0,60}\b(?:black\s+and\s+white|sepia|coloration)\b
    """,
    re.I | re.X,
)


def is_non_visual_candidate(question: str) -> bool:
    """True when the fast checker should assert NOT_DIRECTLY_VISUAL.

    High-precision blacklist match → hard NDV (no LLM). Non-matches are
    UNKNOWN and go to the batched LLM classifier.
    """
    q = (question or "").strip()
    if not q:
        return False
    return bool(_NON_VISUAL_CANDIDATE_RE.search(_SUSPECT_EXEMPT_RE.sub(" ", q)))


def is_non_visual_suspect(question: str) -> bool:
    """Alias for :func:`is_non_visual_candidate` (older call sites)."""
    return is_non_visual_candidate(question)


def is_fast_path_visual(question: str) -> bool:
    """Deprecated visual-DV fast path (removed in v17). Always False."""
    return False


def is_subjective_candidate(question: str) -> bool:
    """True if the question matches the offline drop candidate regex."""
    return bool(_CANDIDATE_RE.search(question or ""))


def confirm_to_binary(confirm: str) -> Tuple[str, Optional[str]]:
    """Map a four-way confirm token to (binary_label, non_visual_reason)."""
    token = (confirm or "").strip().upper().replace("-", "_").replace(" ", "_")
    if token == "VISUAL" or token.startswith("NEEDS_COUNT"):
        # Model sometimes invents NEEDS_COUNTING; counting is visual.
        return "DIRECTLY_VISUAL", None
    if token in ("NEEDS_OCR", "NEEDS_KNOWLEDGE", "NEEDS_OPINION"):
        return "NOT_DIRECTLY_VISUAL", token
    # Legacy binary labels from older prompt versions.
    if token == "DIRECTLY_VISUAL":
        return "DIRECTLY_VISUAL", None
    if token == "NOT_DIRECTLY_VISUAL":
        return "NOT_DIRECTLY_VISUAL", None
    raise ValueError(f"unknown confirm token: {confirm!r}")


# High-precision VISUAL rescue after LLM over-drop (3B often mislabels these).
_VISUAL_RESCUE_RE = re.compile(
    r"""
    \bwhat\s+(?:is|are)\s+.+\bdoing\b |
    \bare\s+(?:the\s+|these\s+|both\s+(?:of\s+)?(?:the\s+|these\s+)?)?
        (?:animals?|people|they|kids?|children|birds?|dogs?|cats?)\s+
        (?:eating|playing|holding|running|sitting|standing|sleeping)\b |
    \bsame\s+(?:size|colou?r|shape)\b |
    \bwhat\s+kind\s+of\s+(?:animal|animals|food|room|hat|shoes?|tree|trees|
        fence|fruit|fruits?|plane|floors?|flooring)\b |
    \bwhat(?:'s|\s+is)\s+in\s+(?:the\s+)?(?:glass|bowl|plate|cup|box|picture|
        image|basket|pot|mug)\b |
    \bwearing\s+glasses\b |
    \b(?:sex|gender)\s+of\b |
    \bhow\s+many\b |
    \b(?:is|are)\s+there\b |
    \bwhat\s+(?:sport|colou?r|room)\b |
    \bwhat\s+is\s+(?:he|she|the\s+(?:man|woman|person|people))\s+wearing\b
    """,
    re.I | re.X,
)


def maybe_rescue_visual(
    question: str, label: str, reason: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Flip clear visual over-drops from the LLM back to DIRECTLY_VISUAL."""
    if label != "NOT_DIRECTLY_VISUAL":
        return label, reason
    q = (question or "").strip()
    if q and _VISUAL_RESCUE_RE.search(q):
        return "DIRECTLY_VISUAL", None
    return label, reason


def parse_classifier_label(raw: str) -> Optional[str]:
    """Extract a confirm or binary label from a model response.

    Prefers the four-way confirm tokens; falls back to legacy binary labels.
    Returns the raw confirm/binary token (not yet mapped to binary + reason).
    """
    text = (raw or "").strip().upper()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    text = text.replace("-", "_").replace(" ", "_")
    if "NEEDS_COUNT" in text:
        return "VISUAL"
    for label in (
        "NEEDS_KNOWLEDGE",
        "NEEDS_OPINION",
        "NEEDS_OCR",
        "NOT_DIRECTLY_VISUAL",
        "DIRECTLY_VISUAL",
        "VISUAL",
    ):
        if text == label or text.startswith(label):
            return label
        if label in text.split():
            return label
    legacy = re.split(r"[\s,.:;]+", text)[0] if text else ""
    if legacy in {"VISUAL"}:
        return "VISUAL"
    if legacy in {"SUBJECTIVE_PERSONAL", "COMMONSENSE", "OCR", "SUBJECTIVE"}:
        # Map old four-way drops onto the closest confirm reason.
        if legacy == "OCR":
            return "NEEDS_OCR"
        if legacy == "COMMONSENSE":
            return "NEEDS_KNOWLEDGE"
        return "NEEDS_OPINION"
    return None


def parse_classifier_label_list(
    raw: str, expected: int
) -> Tuple[Optional[List[str]], str]:
    """Parse a JSON array of confirm/binary labels (or one bare label).

    Returns:
        (labels, detail) — labels is None on failure. Each label is a confirm
        or legacy binary token suitable for :func:`confirm_to_binary`.
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.S).strip()
    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        arr_text = text[start : end + 1]
        try:
            data = json.loads(arr_text)
        except json.JSONDecodeError as exc:
            if expected != 1:
                return None, f"parse_json_error:{exc}"
            data = None
        if isinstance(data, list):
            if len(data) != expected:
                return (
                    None,
                    f"parse_length_mismatch:expected {expected} got {len(data)}",
                )
            out: List[str] = []
            for i, item in enumerate(data):
                label = parse_classifier_label(str(item))
                if label is None:
                    return None, f"parse_item_fail:index {i} value={item!r}"
                out.append(label)
            return out, "ok"
        if expected != 1:
            return None, f"parse_not_a_list:{type(data).__name__}"

    if expected == 1:
        label = parse_classifier_label(text)
        if label is not None:
            return [label], "ok"
        return None, f"parse_fail:{raw!r}"

    return None, f"parse_no_json_array:{raw!r}"


class QuestionClassifier:
    """Ollama-backed blacklist-confirm question classifier."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        num_ctx: int = 4096,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.num_ctx = num_ctx

    def _chat(
        self, user_content: str, *, num_predict: int
    ) -> Tuple[Optional[str], str]:
        """POST one /api/chat turn. Returns (content_or_None, detail)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return None, f"http_error:{exc.code}"
        except urllib.error.URLError as exc:
            return None, f"connection_error:{exc.reason}"
        except TimeoutError:
            return None, "timeout"
        except json.JSONDecodeError as exc:
            return None, f"json_error:{exc}"

        content = ""
        msg = raw.get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        return content, "ok"

    def classify_one(
        self, question: str
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Classify one question.

        Returns:
            (binary_label_or_None, detail, non_visual_reason_or_None)
        """
        content, detail = self._chat(
            _USER_PROMPT_TEMPLATE.format(question=question),
            num_predict=24,
        )
        if content is None:
            return None, detail, None
        confirm = parse_classifier_label(content)
        if confirm is None:
            return None, f"parse_fail:{content!r}", None
        try:
            label, reason = confirm_to_binary(confirm)
        except ValueError:
            return None, f"parse_fail:{content!r}", None
        label, reason = maybe_rescue_visual(question, label, reason)
        return label, "ok", reason

    def classify_batch(
        self, questions: Sequence[str]
    ) -> Tuple[Optional[List[Tuple[str, Optional[str]]]], str]:
        """Classify a packed batch.

        Returns:
            (results_or_None, detail) where each result is
            ``(binary_label, non_visual_reason)``. On parse/HTTP failure
            returns ``(None, detail)`` so the caller can salvage with
            :meth:`classify_one`.
        """
        if not questions:
            return [], "ok"
        if len(questions) == 1:
            label, detail, reason = self.classify_one(questions[0])
            if label is None:
                return None, detail
            return [(label, reason)], detail

        content, detail = self._chat(
            _build_batch_user_prompt(questions),
            num_predict=max(24, len(questions) * 8 + 16),
        )
        if content is None:
            return None, detail
        confirms, parse_detail = parse_classifier_label_list(
            content, expected=len(questions)
        )
        if confirms is None:
            return None, parse_detail
        out: List[Tuple[str, Optional[str]]] = []
        for q, confirm in zip(questions, confirms):
            try:
                label, reason = confirm_to_binary(confirm)
            except ValueError:
                return None, f"parse_item_fail:value={confirm!r}"
            out.append(maybe_rescue_visual(q, label, reason))
        return out, "ok"

    def metadata(self) -> Dict[str, str]:
        """Reproducibility fields for output JSON info."""
        return {
            "model": self.model,
            "host": self.host,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
        }


def _fresh_label_counts() -> Dict[str, int]:
    """Empty label counter dict for classifier accounting."""
    counts: Dict[str, int] = {lab: 0 for lab in QUESTION_LABELS}
    counts["OFFLINE_CANDIDATE_DROP"] = 0
    counts["PARSE_FAIL_DROP"] = 0
    counts["PARSE_FAIL_KEEP"] = 0
    counts["BLACKLIST_NDV"] = 0
    return counts


def _question_id_set(rows: Sequence[Dict[str, Any]]) -> Set[int]:
    """Collect integer question_id values from row dicts."""
    out: Set[int] = set()
    for row in rows:
        try:
            out.add(int(row["question_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_classifier_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    """Load a classifier checkpoint sidecar, or None if missing/corrupt."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    kept = data.get("kept")
    dropped = data.get("dropped")
    if not isinstance(info, dict):
        return None
    if not isinstance(kept, list) or not isinstance(dropped, list):
        return None
    return data


def validate_classifier_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    pre_classify_count: int,
    input_count: int,
    classifier_meta: Optional[Dict[str, Any]] = None,
    fast_path: bool = True,
) -> bool:
    """True when a checkpoint matches the current run configuration.

    ``fast_path`` here means blacklist auto-NDV enabled (``blacklist_drop``).
    Checkpoints built with auto-drop on cannot be reused when
    ``--no-blacklist-drop`` / ``--no-fast-path`` is set (and vice versa).
    """
    info = checkpoint.get("info") or {}
    if info.get("prompt_version") != CLASSIFIER_PROMPT_VERSION:
        return False
    enabled = info.get("blacklist_drop_enabled")
    if enabled is None:
        enabled = info.get("fast_path_enabled", True)
    if bool(enabled) != bool(fast_path):
        return False
    if int(info.get("pre_classify_count", -1)) != pre_classify_count:
        return False
    if int(info.get("input_count", -1)) != input_count:
        return False
    if classifier_meta:
        if info.get("model") and info.get("model") != classifier_meta.get("model"):
            return False
        if info.get("host") and info.get("host") != classifier_meta.get("host"):
            return False
    return True


def save_classifier_checkpoint(
    path: Path,
    kept: Sequence[Dict],
    dropped: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
) -> None:
    """Atomically persist classifier progress for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": dict(info),
        "kept": list(kept),
        "dropped": list(dropped),
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp.json",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def delete_classifier_checkpoint(path: Path) -> None:
    """Remove classifier checkpoint after successful main output write."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def filter_non_visual_questions(
    rows: Sequence[Dict],
    classifier: Optional[QuestionClassifier] = None,
    *,
    offline_drop_candidates: bool = False,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
    resume: bool = True,
    classifier_meta: Optional[Dict[str, Any]] = None,
    input_count: int = 0,
    fast_path: bool = True,
    batch_size: int = 10,
) -> Tuple[List[Dict], List[Dict[str, Any]], Dict[str, int]]:
    """Keep DIRECTLY_VISUAL rows; collect NOT_DIRECTLY_VISUAL drops for a sidecar.

    Args:
        rows: caption rows (dicts with at least ``question``).
        classifier: Ollama classifier; UNKNOWN rows are batched to the LLM.
        offline_drop_candidates: when True and classifier is unavailable,
            drop all regex candidates (conservative offline mode).
        checkpoint_path: optional sidecar for incremental classifier resume.
        checkpoint_every: save checkpoint every N newly classified questions.
        resume: load and continue from ``checkpoint_path`` when valid.
        classifier_meta: model/host/prompt metadata for checkpoint validation.
        input_count: raw VQA input count before OCR/dedup (for validation).
        fast_path: when True (default), high-precision blacklist matches are
            dropped as NOT_DIRECTLY_VISUAL without LLM. When False
            (``--no-blacklist-drop`` / ``--no-fast-path``), every question is
            UNKNOWN and goes to the batched LLM.
        batch_size: pack this many UNKNOWN questions into one Ollama call
            (JSON array of labels); salvage with :meth:`classify_one` on
            batch parse failure.

    Returns:
        (kept_rows, dropped_rows, label_counts)

        Kept rows and ``dropped_rows`` both carry ``visual_filter_source``
        (``blacklist`` / ``llm_classifier``);
        ``dropped_rows`` entries also include ``label``, optional
        ``non_visual_reason``, and optional ``detail``.
    """
    n_total = len(rows)
    pre_classify_count = n_total
    batch_n = max(1, int(batch_size))
    blacklist_drop = bool(fast_path)

    if classifier is not None and checkpoint_path is not None and resume:
        existing = load_classifier_checkpoint(checkpoint_path)
        if existing and validate_classifier_checkpoint(
            existing,
            pre_classify_count=pre_classify_count,
            input_count=input_count,
            classifier_meta=classifier_meta,
            fast_path=blacklist_drop,
        ):
            info = existing.get("info") or {}
            if info.get("status") == "complete":
                kept = list(existing.get("kept") or [])
                dropped = list(existing.get("dropped") or [])
                label_counts = dict(info.get("label_counts") or _fresh_label_counts())
                print(
                    f"Classifier resume: loaded complete checkpoint "
                    f"({len(kept)} kept, {len(dropped)} dropped) "
                    f"from {checkpoint_path}",
                    flush=True,
                )
                return kept, dropped, label_counts

    kept: List[Dict] = []
    dropped: List[Dict[str, Any]] = []
    label_counts = _fresh_label_counts()
    classified_ids: Set[int] = set()

    if (
        classifier is not None
        and checkpoint_path is not None
        and resume
    ):
        existing = load_classifier_checkpoint(checkpoint_path)
        if existing and validate_classifier_checkpoint(
            existing,
            pre_classify_count=pre_classify_count,
            input_count=input_count,
            classifier_meta=classifier_meta,
            fast_path=blacklist_drop,
        ):
            kept = list(existing.get("kept") or [])
            dropped = list(existing.get("dropped") or [])
            ckpt_info = existing.get("info") or {}
            label_counts = dict(ckpt_info.get("label_counts") or _fresh_label_counts())
            classified_ids = _question_id_set(kept) | _question_id_set(dropped)
            print(
                f"Classifier resume: continuing from checkpoint "
                f"({len(classified_ids)}/{n_total} done) "
                f"-> {checkpoint_path}",
                flush=True,
            )

    ckpt_every = max(1, int(checkpoint_every))

    def _build_checkpoint_info(status: str) -> Dict[str, Any]:
        classified_count = len(classified_ids)
        out: Dict[str, Any] = {
            "status": status,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
            "blacklist_drop_enabled": bool(blacklist_drop),
            "fast_path_enabled": bool(blacklist_drop),  # legacy alias
            "batch_size": batch_n,
            "input_count": input_count,
            "pre_classify_count": pre_classify_count,
            "label_counts": dict(label_counts),
            "classified_count": classified_count,
            "total_to_classify": n_total,
            "post_filter_count": len(kept),
        }
        if classifier_meta:
            out.update(
                {
                    k: classifier_meta[k]
                    for k in ("model", "host")
                    if k in classifier_meta
                }
            )
        return out

    def _maybe_save_checkpoint(status: str, force: bool = False) -> None:
        if checkpoint_path is None or classifier is None:
            return
        classified_count = len(classified_ids)
        if not force and classified_count % ckpt_every != 0:
            return
        save_classifier_checkpoint(
            checkpoint_path,
            kept,
            dropped,
            _build_checkpoint_info(status),
        )

    if classifier is not None and n_total:
        n_blacklist = (
            sum(
                1
                for row in rows
                if is_non_visual_candidate(str(row.get("question") or ""))
            )
            if blacklist_drop
            else 0
        )
        n_unknown = n_total - n_blacklist
        print(
            f"Question classifier: {n_total} questions "
            f"(blacklist->NDV | UNKNOWN->batched LLM), "
            f"{n_blacklist} blacklist NDV (no LLM), "
            f"{n_unknown} UNKNOWN to confirm with the LLM "
            f"(batch-size={batch_n})"
            + ("" if blacklist_drop else " (--no-blacklist-drop)")
            + "...",
            flush=True,
        )
    elif offline_drop_candidates:
        n_cand = sum(
            1
            for row in rows
            if is_subjective_candidate(str(row.get("question") or ""))
        )
        print(
            f"Offline candidates: {n_cand}/{n_total} "
            "(dropping without Qwen)...",
            flush=True,
        )

    progress_every = max(1, min(25, n_total // 20)) if n_total else 1
    done = 0
    llm_calls = 0
    newly_classified = 0
    # Buffer of (row, qid, question) awaiting a packed LLM call.
    llm_buffer: List[Tuple[Dict, Optional[int], str]] = []

    def _apply_llm_label(
        row: Dict,
        qid: Optional[int],
        label: Optional[str],
        detail: str,
        non_visual_reason: Optional[str] = None,
    ) -> None:
        nonlocal newly_classified
        if label is None:
            # Prefer keep on parse failure (only LLM asserts DIRECTLY_VISUAL;
            # dropping UNKNOWN-on-parse massively over-filters visual Qs).
            label_counts["PARSE_FAIL_KEEP"] = label_counts.get("PARSE_FAIL_KEEP", 0) + 1
            label_counts["DIRECTLY_VISUAL"] += 1
            row["visual_filter_source"] = VISUAL_FILTER_LLM
            kept.append(row)
        else:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label == "DIRECTLY_VISUAL":
                row["visual_filter_source"] = VISUAL_FILTER_LLM
                kept.append(row)
            else:
                dropped.append(
                    _drop_record(
                        row,
                        label,
                        detail,
                        VISUAL_FILTER_LLM,
                        non_visual_reason=non_visual_reason,
                    )
                )
        if qid is not None:
            classified_ids.add(qid)
        newly_classified += 1

    def _flush_llm_buffer() -> None:
        nonlocal llm_calls
        if not llm_buffer or classifier is None:
            return
        questions = [q for _, _, q in llm_buffer]
        llm_calls += 1
        results, detail = classifier.classify_batch(questions)
        if results is None:
            # Salvage: one classify_one call per buffered question.
            for row, qid, q in llm_buffer:
                llm_calls += 1
                label, one_detail, reason = classifier.classify_one(q)
                _apply_llm_label(
                    row, qid, label, one_detail or detail, reason
                )
        else:
            for (row, qid, _), (label, reason) in zip(llm_buffer, results):
                _apply_llm_label(row, qid, label, detail, reason)
        llm_buffer.clear()
        _maybe_save_checkpoint("in_progress")

    for row in rows:
        q = str(row.get("question") or "")

        if classifier is None:
            if offline_drop_candidates and is_subjective_candidate(q):
                label_counts["OFFLINE_CANDIDATE_DROP"] += 1
                label_counts["NOT_DIRECTLY_VISUAL"] += 1
                dropped.append(
                    _drop_record(row, "NOT_DIRECTLY_VISUAL", "offline_candidate")
                )
                continue
            kept.append(row)
            label_counts["DIRECTLY_VISUAL"] += 1
            continue

        try:
            qid = int(row["question_id"])
        except (KeyError, TypeError, ValueError):
            qid = None

        if qid is not None and qid in classified_ids:
            continue

        try:
            done += 1
            if done == 1 or done % progress_every == 0 or done == n_total:
                print(
                    f"  classify progress: {len(classified_ids)}/{n_total} "
                    f"(LLM calls: {llm_calls}, buffered: {len(llm_buffer)})",
                    flush=True,
                )

            # Fast checker: high-precision blacklist → hard NDV (no LLM).
            if blacklist_drop and is_non_visual_candidate(q):
                label_counts["BLACKLIST_NDV"] += 1
                label_counts["NOT_DIRECTLY_VISUAL"] += 1
                dropped.append(
                    _drop_record(
                        row,
                        "NOT_DIRECTLY_VISUAL",
                        "blacklist",
                        VISUAL_FILTER_BLACKLIST,
                        non_visual_reason="BLACKLIST",
                    )
                )
                if qid is not None:
                    classified_ids.add(qid)
                newly_classified += 1
                _maybe_save_checkpoint("in_progress")
                continue

            # UNKNOWN → batched LLM (only path that can assert DIRECTLY_VISUAL).
            llm_buffer.append((row, qid, q))
            if len(llm_buffer) >= batch_n:
                _flush_llm_buffer()
        except KeyboardInterrupt:
            _maybe_save_checkpoint("in_progress", force=True)
            raise

    if classifier is not None:
        try:
            _flush_llm_buffer()
        except KeyboardInterrupt:
            _maybe_save_checkpoint("in_progress", force=True)
            raise

    if classifier is not None and checkpoint_path is not None:
        if newly_classified or len(classified_ids) == n_total:
            _maybe_save_checkpoint("complete", force=True)

    return kept, dropped, label_counts


def _drop_record(
    row: Dict,
    label: str,
    detail: str = "",
    visual_filter_source: str = "",
    non_visual_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a sidecar annotation from a source row."""
    out: Dict[str, Any] = {
        "question_id": row.get("question_id"),
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "answer_count": row.get("answer_count"),
        "answer_consensus": row.get("answer_consensus"),
        "label": label,
    }
    if visual_filter_source:
        out["visual_filter_source"] = visual_filter_source
    if non_visual_reason:
        out["non_visual_reason"] = non_visual_reason
    if detail:
        out["detail"] = detail
    return out
