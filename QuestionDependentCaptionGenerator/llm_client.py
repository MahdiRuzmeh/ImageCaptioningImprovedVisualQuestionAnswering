"""Ollama client baraye yek model: sequential ya concurrent API request.

Har fail reason-dar hast ta log file betune tozih bede chera ``fallback``
be ``llm_fallback`` tabdil nashod.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from caption_rules import DIGIT_TO_WORD
from llm_prompts import chat_messages

_WORD_TO_DIGIT = {word: digit for digit, word in DIGIT_TO_WORD.items()}


def _numeric_equivalents(token: str) -> Set[str]:
    """A token plus its digit<->word number form (e.g. '2' <-> 'two')."""
    equivalents = {token}
    if token in DIGIT_TO_WORD:
        equivalents.add(DIGIT_TO_WORD[token])
    if token in _WORD_TO_DIGIT:
        equivalents.add(_WORD_TO_DIGIT[token])
    return equivalents



# Suffixes stripped longest-first so a word matches only one bucket (a word
# can't end in both "ing" and "s"). Used for a light stem comparison so verb/
# noun inflections count as the same word (answer 'stands' <-> caption
# 'standing', answer 'dogs' <-> caption 'dog').
_INFLECTION_SUFFIXES = ("ing", "edly", "ed", "es", "s")


def _stem(word: str) -> str:
    """Strip a common inflection suffix, but only if >=3 letters remain."""
    for suf in _INFLECTION_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _token_present(token: str, caption_lower: str) -> bool:
    """Match a token in the caption: exact word, numeric equivalent, or shared stem.

    - '2' matches 'two' (``_numeric_equivalents``).
    - 'stands' matches 'standing', 'dogs' matches 'dog' (shared stem, via a
      light suffix-stripping heuristic — not a real lemmatizer, but enough to
      stop false 'answer_mismatch' rejections on simple inflections).
    """
    if any(re.search(rf"\b{re.escape(t)}\b", caption_lower) for t in _numeric_equivalents(token)):
        return True
    token_stem = _stem(token)
    if len(token_stem) < 3:
        return False
    caption_words = re.findall(r"[a-z']+", caption_lower)
    return any(_stem(w) == token_stem for w in caption_words)


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Markdown code fence ro az javab LLM pak mikone."""
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.I)
    if m:
        return m.group(1).strip()
    return t


def _preview(text: str, limit: int = 400) -> str:
    """Short one-line preview for log files."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


@dataclass
class ParseResult:
    """Result of parsing a model response into a caption list."""

    captions: Optional[List[str]] = None
    reason: str = "ok"
    detail: str = ""


def _clean_caption(cap: str) -> Optional[str]:
    """Normalize one caption string; return None if empty after cleanup."""
    cap = " ".join(cap.strip().split())
    if not cap:
        return None
    cap = re.sub(r"^(caption|output|result)\s*:\s*", "", cap, flags=re.I)
    cap = re.sub(r"^->\s*", "", cap).strip()
    if not cap:
        return None
    # Drop a leading echoed "Q: ... Caption:" prefix if model pasted the prompt
    cap = re.sub(r"^(?:Q:|Question:).+?(?:Caption:\s*)", "", cap, flags=re.I).strip()
    if not cap:
        return None
    words = cap.split()
    if len(words) > 30:
        cap = " ".join(words[:30]).rstrip(".,;") + "."
    if cap and cap[-1] not in ".!?":
        cap = cap + "."
    return cap


def parse_caption_list(raw: str, expected: int) -> ParseResult:
    """Parse model response into ``expected`` caption strings.

    Accepts:
      - JSON array of strings (preferred for batches)
      - plain single sentence when ``expected == 1`` (common on single
        retries; small models often ignore the JSON-array instruction)

    Returns:
        ParseResult with captions on success, or reason/detail on failure.
    """
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")

    # ---- Preferred: JSON array ----
    if start >= 0 and end > start:
        arr_text = text[start : end + 1]
        try:
            data = json.loads(arr_text)
        except json.JSONDecodeError as exc:
            data = None
            if expected != 1:
                return ParseResult(
                    reason="parse_json_error",
                    detail=f"{exc}; preview={_preview(raw)}",
                )
        if isinstance(data, list):
            if len(data) != expected:
                return ParseResult(
                    reason="parse_length_mismatch",
                    detail=(
                        f"expected {expected} captions, got {len(data)}; "
                        f"preview={_preview(raw)}"
                    ),
                )
            out: List[str] = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    return ParseResult(
                        reason="parse_item_not_string",
                        detail=(
                            f"index {i} is {type(item).__name__}; "
                            f"preview={_preview(raw)}"
                        ),
                    )
                cleaned = _clean_caption(item)
                if cleaned is None:
                    return ParseResult(
                        reason="parse_empty_caption",
                        detail=f"index {i} empty; preview={_preview(raw)}",
                    )
                out.append(cleaned)
            return ParseResult(captions=out, reason="ok", detail="")
        if expected != 1:
            return ParseResult(
                reason="parse_not_a_list",
                detail=f"got {type(data).__name__}; preview={_preview(raw)}",
            )

    # ---- Fallback: plain caption text (single-item calls) ----
    # Your log: model returned "The animals are eating." without JSON —
    # that is a valid caption; accept it when expected==1.
    if expected == 1:
        cleaned = _clean_caption(text)
        if cleaned is not None:
            return ParseResult(
                captions=[cleaned],
                reason="ok",
                detail="accepted plain text (not JSON array)",
            )
        return ParseResult(
            reason="parse_empty_caption",
            detail=f"plain text empty; preview={_preview(raw)}",
        )

    return ParseResult(
        reason="parse_no_json_array",
        detail=f"no [..] in response; preview={_preview(raw)}",
    )

# Yes/no answers: caption declarative bashe, lazem nist "yes" toye jomle bashe
_YES = {"yes", "yeah", "yep", "true", "maybe"}
_NO = {"no", "none", "0", "zero", "n/a", "not", "nothing"}

# Negation markers that flip the meaning of a sentence. If the gold answer is
# NOT itself a yes/no-style answer, a caption containing one of these is very
# likely a hallucinated meaning-flip (e.g. Q: 'Who made the cock?' A: 'rolex'
# -> LLM outputs 'No cock was made by Rolex.' — wrong, but 'rolex' still
# passes a naive substring check).
_NEGATION_RE = re.compile(
    r"\b(no|not|n't|never|none|nobody|nothing|neither|without|cannot|can't|"
    r"no one|nowhere)\b",
    re.I,
)

# Question embeds its own negation — a negative caption for answer=yes can be OK
# (e.g. "Is there a light that is not turned on?" + yes → "... is not turned on.").
_QUESTION_NEGATION_RE = re.compile(
    r"\b(not|n't|never|no|none|nobody|nothing|neither|without|cannot|can't)\b",
    re.I,
)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "this", "that",
    "these", "those", "there", "here", "it", "its", "do", "does", "did",
    "can", "could", "will", "would", "have", "has", "had", "you", "your",
    "what", "which", "who", "where", "when", "how", "many", "much", "any",
    "some", "from", "by", "as", "if", "than", "then", "so", "too", "very",
    "just", "about", "into", "over", "after", "before", "between", "out",
    "up", "down", "off", "again", "further", "once", "all", "both", "each",
    "few", "more", "most", "other", "such", "only", "own", "same",
    "s", "t", "don", "now", "i", "me", "my", "we", "our",
    "he", "she", "they", "them", "his", "her", "their",
    # Negation / weak tokens — do not count as grounding overlap
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "without", "cannot", "one", "least", "also", "than", "enough",
}

# Structural sanity check: brackets/labels and stray quotation marks mean the
# model echoed formatting instead of writing a plain sentence.
_BRACKET_CHARS = "[]{}"
_QUOTE_CHARS = "\"\u201c\u201d"

# The model should never write 'the answer is ...' / 'the answer' — it must
# weave the answer into a natural sentence about the image instead.
_ANSWER_PHRASE_RE = re.compile(r"\bthe answer\b", re.I)


def _content_words(text: str) -> Set[str]:
    """Content tokens (stemmed) after dropping stopwords / short tokens."""
    words = re.findall(r"[a-z']+", text.lower())
    out: Set[str] = set()
    for w in words:
        if w in _STOPWORDS or len(w) < 3:
            continue
        out.add(_stem(w))
    return out


_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "gold", "golden", "silver", "beige",
    "tan", "cream", "maroon", "navy", "teal", "cyan", "magenta", "violet",
    "blond", "blonde", "brunette",
}


def _normalize_phrase(text: str) -> str:
    """Light normalize: lowercase, strip punctuation, collapse spaces."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def answer_requires_verbatim(answer: str) -> bool:
    """True for proper nouns, numbers, colors, or short non-yes/no answers."""
    a = answer.strip()
    if not a:
        return False
    low = a.lower()
    if low in _YES or low in _NO:
        return False
    tokens = [t for t in re.split(r"\W+", a) if t]
    if not tokens:
        return False
    if len(tokens) <= 3:
        return True
    if any(t.isdigit() or t in _WORD_TO_DIGIT or t in DIGIT_TO_WORD for t in tokens):
        return True
    if any(t.lower() in _COLOR_WORDS for t in tokens):
        return True
    # Capitalized multi-word name (Loon Mountain)
    caps = sum(1 for t in a.split() if t[:1].isupper())
    if caps >= 2:
        return True
    return False


def answer_verbatim_in_caption(answer: str, caption: str) -> bool:
    """Require answer phrase (light-normalized) to appear in the caption."""
    a_norm = _normalize_phrase(answer)
    c_norm = _normalize_phrase(caption)
    if not a_norm or not c_norm:
        return False
    if a_norm in c_norm:
        return True
    # Number digit <-> word equivalence for single-token numeric answers
    tokens = a_norm.split()
    if len(tokens) == 1:
        for eq in _numeric_equivalents(tokens[0]):
            if re.search(rf"\b{re.escape(eq)}\b", c_norm):
                return True
    # All non-stop tokens must be present (100%)
    content_tokens = [t for t in tokens if t not in _STOPWORDS]
    if not content_tokens:
        content_tokens = tokens
    return all(_token_present(t, c_norm) for t in content_tokens)


def question_relation_preserved(question: str, caption: str) -> Tuple[bool, float]:
    """Check that main question content words appear in the caption.

    Returns (ok, overlap_ratio). For short questions (≤4 content stems),
    **all** must appear — catches shade→free-range and wall→hill swaps that
    still share a subject noun. Longer questions need ≥50% overlap.
    """
    q_words = _content_words(question)
    if not q_words:
        return True, 1.0
    c_words = _content_words(caption)
    if not c_words:
        return False, 0.0
    overlap = len(q_words & c_words)
    ratio = overlap / len(q_words)
    if len(q_words) <= 4:
        min_hits = len(q_words)
    else:
        min_hits = max(2, int(round(0.5 * len(q_words))))
    ok = overlap >= min_hits
    return ok, ratio


def has_unsupported_facts(question: str, answer: str, caption: str) -> bool:
    """True if caption introduces many content words absent from Q+A.

    Conservative: only reject when a clear majority of caption content is new
    (avoids false rejects on light paraphrases).
    """
    allowed = _content_words(f"{question} {answer}")
    cap = _content_words(caption)
    if not cap:
        return False
    extra = cap - allowed
    if not extra:
        return False
    # Hard fail when >2 novel stems and they dominate the caption
    if len(extra) >= 3 and len(extra) / len(cap) >= 0.4:
        return True
    if len(extra) >= 4:
        return True
    return False


def is_semantically_suspicious(
    question: str,
    answer: str,
    caption: str,
    *,
    relation_ratio: float,
) -> bool:
    """Borderline cases that should be escalated to the LLM PASS/FAIL judge."""
    a = answer.strip().lower()
    if a in _YES or a in _NO:
        if relation_ratio < 0.75:
            return True
    else:
        if relation_ratio < 0.65:
            return True
    allowed = _content_words(f"{question} {answer}")
    extra = _content_words(caption) - allowed
    if len(extra) >= 2:
        return True
    if answer_requires_verbatim(answer):
        return True
    return False


def caption_format_is_valid(caption: str) -> Tuple[bool, str]:
    """Structural check for one clean declarative sentence.

    Rejects captions that are:
      - empty, or fewer than 2 words
      - a question (contains '?')
      - wrapped/labeled with brackets or quotation marks
      - more than one sentence (an internal '.'/'!'/'?' before the final
        terminator, e.g. 'This is a home. Not a restaurant.')
      - littered with a stray double period ('..')
      - using the meta-phrase 'the answer'/'the answer is' instead of a
        natural sentence

    Returns:
        (ok, reason) — reason is 'ok' or a short machine-readable code.
    """
    c = caption.strip()
    if not c:
        return False, "empty_caption"
    if len(c.split()) < 2:
        return False, "too_short"
    if "?" in c:
        return False, "contains_question_mark"
    if any(ch in c for ch in _BRACKET_CHARS):
        return False, "contains_brackets"
    if any(ch in c for ch in _QUOTE_CHARS):
        return False, "contains_quotes"
    if ".." in c:
        return False, "double_period"
    if _ANSWER_PHRASE_RE.search(c):
        return False, "contains_answer_phrase"
    body = c[:-1] if c[-1] in ".!?" else c
    if re.search(r"[.!?]", body):
        return False, "multiple_sentences"
    return True, "ok"


def answer_in_caption(
    answer: str,
    caption: str,
    question: str = "",
) -> bool:
    """Check mikone javab toye caption hast; yes/no joda handle mishe.

    Strictness:
      - proper nouns / numbers / colors / short answers → verbatim (100%)
      - other answers → >=50% token match (digit/word + light stem)
      - yes/no → majority question-content overlap (see relation check in
        ``caption_is_valid``); here we still require some overlap.
    """
    a = answer.strip().lower()
    c = caption.strip().lower()
    if not a or not c:
        return False
    if a in _YES or a in _NO:
        if len(c.split()) > 30:
            return False
        if not question.strip():
            return True
        ok, _ratio = question_relation_preserved(question, caption)
        return ok
    if answer_requires_verbatim(answer):
        return answer_verbatim_in_caption(answer, caption)
    if a in c:
        return True
    tokens = [t for t in re.split(r"\W+", a) if t]
    if not tokens:
        return False
    matched = sum(1 for t in tokens if _token_present(t, c))
    return matched / len(tokens) >= 0.5


def has_spurious_negation(answer: str, caption: str) -> bool:
    """True if caption negates a statement that a non-yes/no answer never implied.

    A negation word in the caption is fine when:
      - the gold answer is itself yes/no/none-style, or
      - the answer text already contains a negation word (e.g. 'not moving'),
        so the caption is just echoing it, not flipping the meaning.
    """
    a = answer.strip().lower()
    if not a or a in _YES or a in _NO:
        return False
    if not _NEGATION_RE.search(caption):
        return False
    answer_words = set(re.split(r"\W+", a))
    if answer_words & {"no", "not", "never", "none", "nobody", "nothing", "neither", "without"}:
        return False
    return True


def has_yes_polarity_mismatch(answer: str, caption: str, question: str = "") -> bool:
    """True when answer=yes but the caption clearly negates (meaning flip).

    Skipped when the question itself embeds negation (e.g. 'not turned on'),
    where a negative surface form can still be correct for yes.
    """
    a = answer.strip().lower()
    if a not in _YES:
        return False
    if not _NEGATION_RE.search(caption):
        return False
    if question and _QUESTION_NEGATION_RE.search(question):
        return False
    return True


def is_batch_contamination(
    question: str,
    answer: str,
    caption: str,
    batch_pairs: Sequence[Tuple[str, str]],
    batch_captions: Sequence[Optional[str]],
    self_index: int,
) -> bool:
    """True if ``caption`` looks swapped from another item in the same batch.

    Triggers when:
      - another batch caption is near-identical, or
      - caption content overlaps another Q+A strictly better than this one
        while overlapping this Q+A poorly.
    """
    cap_words = _content_words(caption)
    if not cap_words:
        return False

    self_qa = _content_words(f"{question} {answer}")
    self_overlap = len(cap_words & self_qa)

    norm_cap = " ".join(caption.lower().split())
    for i, other_cap in enumerate(batch_captions):
        if i == self_index or not other_cap:
            continue
        other_norm = " ".join(other_cap.lower().split())
        if other_norm == norm_cap:
            return True
        # Near-duplicate: share most content words both ways
        other_words = _content_words(other_cap)
        if other_words and len(cap_words & other_words) / max(len(cap_words), len(other_words)) >= 0.85:
            return True

    for i, (oq, oa) in enumerate(batch_pairs):
        if i == self_index:
            continue
        other_qa = _content_words(f"{oq} {oa}")
        if not other_qa:
            continue
        other_overlap = len(cap_words & other_qa)
        # Strong match to another item, weak match to self → contamination
        if other_overlap >= 2 and other_overlap > self_overlap + 1:
            if self_overlap == 0 or other_overlap >= self_overlap * 2:
                return True
    return False


def caption_is_valid(
    answer: str,
    caption: str,
    question: str = "",
    *,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    self_index: int = -1,
) -> Tuple[bool, str]:
    """Tier-1 acceptance: format, polarity, grounding, relation, extra facts.

    Returns:
        (ok, reason) — reason is 'ok', 'needs_semantic_review', or a reject code.
        Callers should escalate ``needs_semantic_review`` to the LLM judge.
    """
    fmt_ok, fmt_reason = caption_format_is_valid(caption)
    if not fmt_ok:
        return False, fmt_reason
    if has_yes_polarity_mismatch(answer, caption, question):
        return False, "polarity_mismatch"
    if has_spurious_negation(answer, caption):
        return False, "spurious_negation"
    if question.strip():
        rel_ok, rel_ratio = question_relation_preserved(question, caption)
        if not rel_ok:
            return False, "relation_mismatch"
    else:
        rel_ratio = 1.0
    if not answer_in_caption(answer, caption, question):
        return False, "answer_mismatch"
    if has_unsupported_facts(question, answer, caption):
        return False, "unsupported_facts"
    if (
        batch_pairs is not None
        and batch_captions is not None
        and self_index >= 0
        and is_batch_contamination(
            question, answer, caption, batch_pairs, batch_captions, self_index
        )
    ):
        return False, "batch_contamination"
    if is_semantically_suspicious(
        question, answer, caption, relation_ratio=rel_ratio
    ):
        return True, "needs_semantic_review"
    return True, "ok"


def answer_mismatch_detail(answer: str, caption: str) -> str:
    """Human-readable why answer_in_caption failed."""
    a = answer.strip().lower()
    c = caption.strip().lower()
    tokens = [t for t in re.split(r"\W+", a) if t]
    missing = [t for t in tokens if not _token_present(t, c)]
    matched_pct = round(100 * (len(tokens) - len(missing)) / len(tokens)) if tokens else 0
    return (
        f"answer={answer!r} not reflected in caption={caption!r} "
        f"({matched_pct}% of tokens matched, need >=50%)"
        + (f"; missing_tokens={missing}" if missing else "")
    )


def spurious_negation_detail(answer: str, caption: str) -> str:
    """Human-readable why caption was rejected for a spurious negation."""
    hits = _NEGATION_RE.findall(caption)
    return (
        f"answer={answer!r} is not yes/no, but caption={caption!r} "
        f"contains negation word(s) {hits} — likely a meaning-flip hallucination"
    )


def polarity_mismatch_detail(answer: str, caption: str, question: str = "") -> str:
    """Human-readable why yes-answer polarity check failed."""
    hits = _NEGATION_RE.findall(caption)
    return (
        f"answer={answer!r} is yes-like but caption={caption!r} "
        f"contains negation {hits} (Q={question!r})"
    )


def batch_contamination_detail(caption: str) -> str:
    """Human-readable why batch contamination was suspected."""
    return (
        f"caption={caption!r} looks swapped from another item in the same "
        "LLM batch (near-duplicate or better match to another Q+A)"
    )


def relation_mismatch_detail(question: str, caption: str) -> str:
    """Human-readable why subject/relation overlap failed."""
    ok, ratio = question_relation_preserved(question, caption)
    return (
        f"question content not preserved in caption={caption!r} "
        f"(Q={question!r}, overlap_ratio={ratio:.2f}, ok={ok})"
    )


def unsupported_facts_detail(question: str, answer: str, caption: str) -> str:
    """Human-readable why caption added unsupported facts."""
    allowed = _content_words(f"{question} {answer}")
    extra = sorted(_content_words(caption) - allowed)
    return (
        f"caption={caption!r} adds unsupported content words {extra} "
        f"not in Q+A"
    )


def semantic_fail_detail(question: str, answer: str, caption: str) -> str:
    """Human-readable why the LLM semantic judge returned FAIL."""
    return (
        f"semantic judge FAIL for Q={question!r} A={answer!r} "
        f"caption={caption!r}"
    )


def format_invalid_detail(reason: str, caption: str) -> str:
    """Human-readable why ``caption_format_is_valid`` rejected a caption."""
    return f"caption={caption!r} failed format check: {reason}"


_FORMAT_REASONS = {
    "empty_caption",
    "too_short",
    "contains_question_mark",
    "contains_brackets",
    "contains_quotes",
    "double_period",
    "contains_answer_phrase",
    "multiple_sentences",
}

_VALIDATION_FAIL_REASONS = {
    "answer_mismatch",
    "polarity_mismatch",
    "spurious_negation",
    "batch_contamination",
    "relation_mismatch",
    "unsupported_facts",
    "semantic_fail",
} | _FORMAT_REASONS


def rejection_detail(
    reason: str,
    answer: str,
    caption: str,
    question: str = "",
) -> str:
    """Dispatch to the right human-readable detail message for a reject reason."""
    if reason in _FORMAT_REASONS:
        return format_invalid_detail(reason, caption)
    if reason == "spurious_negation":
        return spurious_negation_detail(answer, caption)
    if reason == "polarity_mismatch":
        return polarity_mismatch_detail(answer, caption, question)
    if reason == "batch_contamination":
        return batch_contamination_detail(caption)
    if reason == "relation_mismatch":
        return relation_mismatch_detail(question, caption)
    if reason == "unsupported_facts":
        return unsupported_facts_detail(question, answer, caption)
    if reason == "semantic_fail":
        return semantic_fail_detail(question, answer, caption)
    return answer_mismatch_detail(answer, caption)


@dataclass
class ChatResult:
    """One Ollama chat call outcome (batch or single)."""

    captions: Optional[List[str]] = None
    reason: str = "ok"
    detail: str = ""


@dataclass
class ItemOutcome:
    """Per Q+A outcome after batch + single retries."""

    caption: Optional[str] = None
    reason: str = "ok"
    detail: str = ""
    attempts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ollama HTTP
# ---------------------------------------------------------------------------


class OllamaClient:
    """Client sade baraye Ollama chat API (yek model, 8GB-friendly)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mistral",
        num_ctx: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 300.0,
    ) -> None:
        """Host va model ro set mikone; options baraye VRAM kam.

        ``num_ctx`` default 4096 — prompt v3 + few-shot + batch fit beshe
        (1024 ghablan truncate mikard va hame caption ha fail mishodan).
        """
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _num_predict(self, batch_size: int) -> int:
        """Max token output: ~40 token per caption + buffer (JSON overhead)."""
        return max(128, batch_size * 40 + 64)

    def chat_captions(self, pairs: Sequence[Tuple[str, str]]) -> ChatResult:
        """Yek packed batch Q+A mifreste; ChatResult ba captions ya fail reason."""
        if not pairs:
            return ChatResult(captions=[], reason="ok", detail="empty batch")

        payload = {
            "model": self.model,
            "messages": chat_messages(pairs),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self._num_predict(len(pairs)),
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
            body_txt = ""
            try:
                body_txt = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return ChatResult(
                reason="http_error",
                detail=f"HTTP {exc.code}: {_preview(body_txt or str(exc))}",
            )
        except urllib.error.URLError as exc:
            return ChatResult(
                reason="connection_error",
                detail=f"cannot reach Ollama at {self.host}: {exc.reason}",
            )
        except TimeoutError:
            return ChatResult(
                reason="timeout",
                detail=f"Ollama request timed out after {self.timeout_s}s",
            )
        except json.JSONDecodeError as exc:
            return ChatResult(
                reason="http_json_error",
                detail=f"Ollama response not JSON: {exc}",
            )

        content = ""
        msg = raw.get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        if not content:
            err = raw.get("error")
            return ChatResult(
                reason="empty_response",
                detail=f"model returned empty content; error={err!r}",
            )

        parsed = parse_caption_list(content, expected=len(pairs))
        if parsed.captions is None:
            return ChatResult(reason=parsed.reason, detail=parsed.detail)
        return ChatResult(captions=parsed.captions, reason="ok", detail="")

    def semantic_judge(
        self,
        question: str,
        answer: str,
        caption: str,
    ) -> Tuple[bool, str]:
        """Tier-2 LLM judge: PASS only if caption matches Q+A with no extras.

        Returns:
            (pass, detail) — pass True iff model returns PASS.
        """
        system = (
            "You are a strict caption validator. Reply with only PASS or FAIL."
        )
        user = (
            "Given QUESTION, ANSWER and CAPTION, return PASS only if the "
            "caption correctly expresses the answer to the question and "
            "adds no unsupported factual information. Otherwise return FAIL.\n\n"
            f"QUESTION: {question}\n"
            f"ANSWER: {answer}\n"
            f"CAPTION: {caption}\n"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": min(self.num_ctx, 2048),
                "num_predict": 8,
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
        except Exception as exc:
            # Fail-closed on judge errors
            return False, f"semantic_judge_error:{exc}"

        content = ""
        msg = raw.get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        text = content.strip().upper()
        if text.startswith("PASS") or re.search(r"\bPASS\b", text):
            return True, "semantic_pass"
        return False, f"semantic_fail:{_preview(content, 120)}"

    def _accept_or_escalate(
        self,
        question: str,
        answer: str,
        caption: str,
        *,
        batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        batch_captions: Optional[Sequence[Optional[str]]] = None,
        self_index: int = -1,
    ) -> Tuple[bool, str]:
        """Run Tier-1 then optional Tier-2 semantic judge."""
        ok, reason = caption_is_valid(
            answer,
            caption,
            question,
            batch_pairs=batch_pairs,
            batch_captions=batch_captions,
            self_index=self_index,
        )
        if not ok:
            return False, reason
        if reason != "needs_semantic_review":
            return True, "ok"
        passed, detail = self.semantic_judge(question, answer, caption)
        if passed:
            return True, "ok"
        return False, "semantic_fail"

    def captions_with_retry(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        single_retries: int = 1,
    ) -> List[ItemOutcome]:
        """Batch try, then per-item single retries with reasons.

        Validation policy (Comments7): Tier-1 lexical checks, escalate
        suspicious items to Tier-2 Qwen PASS/FAIL. On FAIL, regenerate once
        (``single_retries`` default 1); if still FAIL, leave as unresolved.

        Args:
            pairs: (question, answer) batch
            single_retries: extra single-item calls after batch miss
                (default 1 — one regenerate then drop)

        Returns:
            one ``ItemOutcome`` per input pair
        """
        pairs_list = list(pairs)
        n = len(pairs_list)
        out: List[ItemOutcome] = [
            ItemOutcome(reason="pending", detail="not attempted") for _ in range(n)
        ]

        batch = self.chat_captions(pairs_list)
        batch_caps: List[Optional[str]] = [None] * n
        if batch.captions is not None:
            tentative: List[Optional[str]] = [None] * n
            for i, cap in enumerate(batch.captions):
                q, a = pairs_list[i]
                ok, reason = self._accept_or_escalate(q, a, cap)
                if ok:
                    tentative[i] = cap
                else:
                    out[i] = ItemOutcome(
                        reason=reason,
                        detail=rejection_detail(reason, a, cap, q),
                        attempts=[f"batch:{reason}"],
                    )
            for i, cap in enumerate(tentative):
                if cap is None:
                    continue
                q, a = pairs_list[i]
                ok, reason = self._accept_or_escalate(
                    q,
                    a,
                    cap,
                    batch_pairs=pairs_list,
                    batch_captions=tentative,
                    self_index=i,
                )
                if ok:
                    batch_caps[i] = cap
                    out[i] = ItemOutcome(
                        caption=cap,
                        reason="ok",
                        detail="accepted from batch",
                        attempts=["batch:ok"],
                    )
                else:
                    out[i] = ItemOutcome(
                        reason=reason,
                        detail=rejection_detail(reason, a, cap, q),
                        attempts=[f"batch:{reason}"],
                    )
        else:
            for i in range(n):
                out[i] = ItemOutcome(
                    reason=batch.reason,
                    detail=batch.detail,
                    attempts=[f"batch:{batch.reason}"],
                )

        if all(o.caption is not None for o in out):
            return out

        for i, (q, a) in enumerate(pairs_list):
            if out[i].caption is not None:
                continue
            last = out[i]
            for attempt in range(1, single_retries + 1):
                single = self.chat_captions([(q, a)])
                tag = f"single#{attempt}"
                if single.captions is None:
                    last.attempts.append(f"{tag}:{single.reason}")
                    last.reason = single.reason
                    last.detail = single.detail
                    continue
                cap = single.captions[0]
                ok, reason = self._accept_or_escalate(q, a, cap)
                if ok:
                    out[i] = ItemOutcome(
                        caption=cap,
                        reason="ok",
                        detail=f"accepted from {tag}",
                        attempts=last.attempts + [f"{tag}:ok"],
                    )
                    break
                last.attempts.append(f"{tag}:{reason}")
                last.reason = reason
                last.detail = rejection_detail(reason, a, cap, q)
            else:
                out[i] = last
        return out


def run_batches_concurrent(
    client: OllamaClient,
    batches: Sequence[Sequence[Tuple[str, str]]],
    workers: int = 1,
    on_batch_done: Optional[Callable[[int, List[ItemOutcome]], None]] = None,
    on_batch_start: Optional[Callable[[int, int], None]] = None,
    single_retries: int = 1,
) -> List[List[ItemOutcome]]:
    """Chand packed batch ro sequential ya ba ThreadPool mifreste.

    Args:
        client: OllamaClient (yek model)
        batches: list of Q+A batches
        workers: concurrent API request (1 = sequential, 8GB safe)
        on_batch_done: callback(batch_index, outcomes) bad az har batch
        on_batch_start: callback(batch_index, batch_len) ghabl az har call
        single_retries: forwarded to ``captions_with_retry`` (default 1)
    """
    n = len(batches)
    out: List[List[ItemOutcome]] = [[] for _ in range(n)]
    workers = max(1, int(workers))

    if workers == 1:
        for i, batch in enumerate(batches):
            if on_batch_start is not None:
                on_batch_start(i, len(batch))
            caps = client.captions_with_retry(batch, single_retries=single_retries)
            out[i] = caps
            if on_batch_done is not None:
                on_batch_done(i, caps)
        return out

    def _job(
        idx: int, batch: Sequence[Tuple[str, str]]
    ) -> Tuple[int, List[ItemOutcome]]:
        if on_batch_start is not None:
            on_batch_start(idx, len(batch))
        return idx, client.captions_with_retry(batch, single_retries=single_retries)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, i, b) for i, b in enumerate(batches)]
        for fut in as_completed(futs):
            idx, caps = fut.result()
            out[idx] = caps
            if on_batch_done is not None:
                on_batch_done(idx, caps)
    return out
