"""Grounding, format, and batch-contamination checks for caption validation.

High-precision hard rejects and soft flags used by :mod:`validation.fast_validator`.
The fast layer does not judge semantic correctness — only structural and
grounding errors we are nearly certain about.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set, Tuple

from validation.config import ValidationConfig
from validation.tokens import (
    _NO,
    _YES,
    answer_requires_verbatim,
    content_words,
    normalize_phrase,
    numeric_equivalents,
    required_question_stems,
    token_present,
)

# Re-export yes/no sets for other modules
YES_ANSWERS = _YES
NO_ANSWERS = _NO

_NEGATION_RE = re.compile(
    r"(\b(?:no|not|never|none|nobody|nothing|neither|without|cannot|"
    r"no one|nowhere)\b|\w*n't\b)",
    re.I,
)

_QUESTION_NEGATION_RE = re.compile(
    r"(\b(?:not|never|no|none|nobody|nothing|neither|without|cannot)\b"
    r"|\w*n't\b)",
    re.I,
)

_NON_SENTENTIAL_NO_RE = re.compile(
    r"""
    \bno\s+(?:parking|standing|stopping|smoking|entry|entrance|exit|
              trespassing|littering|swimming|diving|fishing|hunting|
              turn|turns|u-turn|uturn|left\s+turn|right\s+turn|
              dogs|pets|photos|photography|food|drinks|outlet|service|
              vacancy|passing|dumping|loitering|skateboarding|bikes|
              cell\s+phones|shirt|shoes)\b |
    \b(?:a|an|the|this|that|any|one)\s+no\s+\w+\s+
        (?:sign|signs|symbol|marking|markings|notice|placard)\b
    """,
    re.I | re.X,
)

# ---------------------------------------------------------------------------
# 1.1–1.2 Format checks (empty, brackets, quotes, question mark)
# ---------------------------------------------------------------------------

_BRACKET_CHARS = "[]{}()"
_QUOTE_CHARS = "\"\u201c\u201d'"
_ANSWER_PHRASE_RE = re.compile(r"\bthe answer\b", re.I)

_CAPTION_AFFIRMS_RE = re.compile(r"^\s*yes\b|\bthe answer is yes\b", re.I)
_CAPTION_DENIES_RE = re.compile(r"^\s*no[,.\s]|\bthe answer is no\b", re.I)


def caption_format_is_valid(
    caption: str,
    config: Optional[ValidationConfig] = None,
) -> Tuple[bool, str]:
    """Structural check for one clean declarative sentence.

    Returns:
        (ok, reason) — reason is 'ok' or a short machine-readable code.
    """
    cfg = config or ValidationConfig()
    c = caption.strip()
    if not c:
        return False, "empty_caption"
    word_count = len(c.split())
    if word_count < cfg.min_words:
        return False, "too_short"
    if word_count > cfg.max_words:
        return False, "too_long"
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
    body = c[:-1] if c and c[-1] in ".!?" else c
    if re.search(r"[.!?]", body):
        return False, "multiple_sentences"
    return True, "ok"


# ---------------------------------------------------------------------------
# Negation and polarity
# ---------------------------------------------------------------------------


def has_sentential_negation(caption: str) -> bool:
    """True when the caption negates its own statement."""
    return bool(_NEGATION_RE.search(_NON_SENTENTIAL_NO_RE.sub(" ", caption)))


def has_spurious_negation(answer: str, caption: str) -> bool:
    """True if caption negates a statement that a non-yes/no answer never implied."""
    a = answer.strip().lower()
    if not a or a in _YES or a in _NO:
        return False
    if not has_sentential_negation(caption):
        return False
    answer_word_set = set(re.split(r"\W+", a))
    if answer_word_set & {
        "no", "not", "never", "none", "nobody", "nothing", "neither", "without"
    }:
        return False
    return True


def has_yes_polarity_mismatch(answer: str, caption: str, question: str = "") -> bool:
    """True when answer=yes but the caption clearly negates (meaning flip)."""
    a = answer.strip().lower()
    if a not in _YES:
        return False
    if not has_sentential_negation(caption):
        return False
    if question and _QUESTION_NEGATION_RE.search(question):
        return False
    return True


def has_no_polarity_mismatch(answer: str, caption: str, question: str = "") -> bool:
    """True when answer=no but the caption explicitly affirms."""
    if answer.strip().lower() != "no":
        return False
    del question
    return bool(_CAPTION_AFFIRMS_RE.search(caption))


def has_yes_denial(answer: str, caption: str) -> bool:
    """True when answer is yes-like but the caption opens with a flat 'No'."""
    if answer.strip().lower() not in _YES:
        return False
    return bool(_CAPTION_DENIES_RE.search(caption))


def echoes_question(question: str, caption: str) -> bool:
    """True when the caption just repeats the question as a statement."""
    q = normalize_phrase(question)
    c = normalize_phrase(caption)
    if not q or not c:
        return False
    if c == q or (len(q.split()) >= 4 and c.startswith(q)):
        return True
    # Interrogative tail echoed in a declarative caption (e.g. Q ends with
    # "do you see" and caption keeps that phrase instead of stating the count).
    for tail in (
        "do you see",
        "can you see",
        "are there",
        "is there",
        "is this",
        "does the",
        "do the",
    ):
        if tail in q and tail in c:
            return True
    return False


# ---------------------------------------------------------------------------
# Answer grounding
# ---------------------------------------------------------------------------


def answer_verbatim_in_caption(answer: str, caption: str) -> bool:
    """Require answer phrase (light-normalized) to appear in the caption."""
    a_norm = normalize_phrase(answer)
    c_norm = normalize_phrase(caption)
    if not a_norm or not c_norm:
        return False
    if a_norm in c_norm:
        return True
    tokens = a_norm.split()
    if len(tokens) == 1:
        for eq in numeric_equivalents(tokens[0]):
            if re.search(rf"\b{re.escape(eq)}\b", c_norm):
                return True
    content_tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    if not content_tokens:
        content_tokens = tokens
    return all(token_present(t, c_norm) for t in content_tokens)


def question_relation_preserved(
    question: str,
    caption: str,
    *,
    relation_min_ratio: float = 0.5,
) -> Tuple[bool, float]:
    """Check that required question content words appear in the caption.

    Returns:
        (ok, overlap_ratio)
    """
    q_words = required_question_stems(question)
    if not q_words:
        return True, 1.0
    c_words = content_words(caption)
    if not c_words:
        return False, 0.0
    overlap = len(q_words & c_words)
    ratio = overlap / len(q_words)
    return ratio >= relation_min_ratio, ratio


def answer_in_caption(
    answer: str,
    caption: str,
    question: str = "",
    *,
    relation_min_ratio: float = 0.5,
) -> bool:
    """Check that the answer is reflected in the caption."""
    a = answer.strip().lower()
    c = caption.strip().lower()
    if not a or not c:
        return False
    if a in _YES or a in _NO:
        if not question.strip():
            return True
        ok, _ratio = question_relation_preserved(
            question, caption, relation_min_ratio=relation_min_ratio
        )
        return ok
    if answer_requires_verbatim(answer):
        return answer_verbatim_in_caption(answer, caption)
    if a in c:
        return True
    tokens = [t for t in re.split(r"\W+", a) if t]
    if not tokens:
        return False
    matched = sum(1 for t in tokens if token_present(t, c))
    return matched / len(tokens) >= 0.5


def has_unsupported_facts(question: str, answer: str, caption: str) -> bool:
    """True if caption introduces many content words absent from Q+A."""
    allowed = content_words(f"{question} {answer}")
    cap = content_words(caption)
    if not cap:
        return False
    extra = cap - allowed
    if not extra:
        return False
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
    allowed = content_words(f"{question} {answer}")
    extra = content_words(caption) - allowed
    if len(extra) >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Batch contamination
# ---------------------------------------------------------------------------


def is_batch_contamination(
    question: str,
    answer: str,
    caption: str,
    batch_pairs: Sequence[Tuple[str, str]],
    batch_captions: Sequence[Optional[str]],
    self_index: int,
) -> bool:
    """True if caption looks swapped from another item in the same batch."""
    cap_words = content_words(caption)
    if not cap_words:
        return False

    self_qa = content_words(f"{question} {answer}")
    self_overlap = len(cap_words & self_qa)

    norm_cap = " ".join(caption.lower().split())
    for i, other_cap in enumerate(batch_captions):
        if i == self_index or not other_cap:
            continue
        other_norm = " ".join(other_cap.lower().split())
        if other_norm == norm_cap:
            return True
        other_words = content_words(other_cap)
        if other_words and len(cap_words & other_words) / max(
            len(cap_words), len(other_words)
        ) >= 0.85:
            return True

    for i, (oq, oa) in enumerate(batch_pairs):
        if i == self_index:
            continue
        other_qa = content_words(f"{oq} {oa}")
        if not other_qa:
            continue
        other_overlap = len(cap_words & other_qa)
        if other_overlap >= 2 and other_overlap > self_overlap + 1:
            if self_overlap == 0 or other_overlap >= self_overlap * 2:
                return True
    return False


# ---------------------------------------------------------------------------
# Soft flags
# ---------------------------------------------------------------------------

FLAG_RELATION_LOW = "relation_low"
FLAG_UNSUPPORTED_FACTS = "unsupported_facts_suspect"
FLAG_NO_ANSWER_WITHOUT_NEGATION = "no_answer_without_negation"
FLAG_ANSWER_PARTIAL = "answer_partial_match"
FLAG_OVERLAP_BORDERLINE = "overlap_borderline"

VALIDATION_FLAGS = (
    FLAG_RELATION_LOW,
    FLAG_UNSUPPORTED_FACTS,
    FLAG_NO_ANSWER_WITHOUT_NEGATION,
    FLAG_ANSWER_PARTIAL,
    FLAG_OVERLAP_BORDERLINE,
)

_FORMAT_REASONS = {
    "empty_caption",
    "too_short",
    "too_long",
    "contains_question_mark",
    "contains_brackets",
    "contains_quotes",
    "double_period",
    "contains_answer_phrase",
    "multiple_sentences",
}

_VALIDATION_FAIL_REASONS = {
    "answer_mismatch",
    "echoes_question",
    "polarity_mismatch",
    "spurious_negation",
    "batch_contamination",
    "semantic_fail",
    "overlap_too_low",
} | _FORMAT_REASONS


def caption_soft_flags(
    question: str,
    answer: str,
    caption: str,
    *,
    relation_min_ratio: float = 0.5,
) -> List[str]:
    """Suspicious-but-not-wrong findings, safe to keep in the dataset."""
    flags: List[str] = []
    if question.strip():
        rel_ok, _ratio = question_relation_preserved(
            question, caption, relation_min_ratio=relation_min_ratio
        )
        if not rel_ok:
            flags.append(FLAG_RELATION_LOW)
    if has_unsupported_facts(question, answer, caption):
        flags.append(FLAG_UNSUPPORTED_FACTS)
    if answer.strip().lower() == "no" and not has_sentential_negation(caption):
        flags.append(FLAG_NO_ANSWER_WITHOUT_NEGATION)
    if not answer_requires_verbatim(answer) and not answer_in_caption(
        answer, caption, question, relation_min_ratio=relation_min_ratio
    ):
        flags.append(FLAG_ANSWER_PARTIAL)
    return flags


def caption_hard_reject_reason(
    answer: str,
    caption: str,
    question: str = "",
    *,
    config: Optional[ValidationConfig] = None,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    self_index: int = -1,
) -> Optional[str]:
    """Reject code for errors we are nearly certain about, else None."""
    cfg = config or ValidationConfig()
    fmt_ok, fmt_reason = caption_format_is_valid(caption, cfg)
    if not fmt_ok:
        return fmt_reason
    if question.strip() and echoes_question(question, caption):
        return "echoes_question"
    if has_yes_polarity_mismatch(answer, caption, question):
        return "polarity_mismatch"
    if has_yes_denial(answer, caption):
        return "polarity_mismatch"
    if has_no_polarity_mismatch(answer, caption, question):
        return "polarity_mismatch"
    if has_spurious_negation(answer, caption):
        return "spurious_negation"
    if answer_requires_verbatim(answer) and not answer_verbatim_in_caption(
        answer, caption
    ):
        return "answer_mismatch"
    if (
        batch_pairs is not None
        and batch_captions is not None
        and self_index >= 0
        and is_batch_contamination(
            question, answer, caption, batch_pairs, batch_captions, self_index
        )
    ):
        return "batch_contamination"
    return None
