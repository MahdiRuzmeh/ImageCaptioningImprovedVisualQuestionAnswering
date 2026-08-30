"""Human-readable rejection and flag detail messages for validation logs."""

from __future__ import annotations

import re

from validation.checks import (
    _FORMAT_REASONS,
    question_relation_preserved,
)
from validation.tokens import content_words, token_present


def answer_mismatch_detail(answer: str, caption: str) -> str:
    """Human-readable why answer_in_caption failed."""
    a = answer.strip().lower()
    c = caption.strip().lower()
    tokens = [t for t in re.split(r"\W+", a) if t]
    missing = [t for t in tokens if not token_present(t, c)]
    matched_pct = (
        round(100 * (len(tokens) - len(missing)) / len(tokens)) if tokens else 0
    )
    return (
        f"answer={answer!r} not reflected in caption={caption!r} "
        f"({matched_pct}% of tokens matched, need >=50%)"
        + (f"; missing_tokens={missing}" if missing else "")
    )


def spurious_negation_detail(answer: str, caption: str) -> str:
    """Human-readable why caption was rejected for a spurious negation."""
    from validation.checks import _NEGATION_RE

    hits = _NEGATION_RE.findall(caption)
    return (
        f"answer={answer!r} is not yes/no, but caption={caption!r} "
        f"contains negation word(s) {hits} — likely a meaning-flip hallucination"
    )


def polarity_mismatch_detail(answer: str, caption: str, question: str = "") -> str:
    """Human-readable why the yes/no polarity check failed."""
    from validation.checks import _NEGATION_RE

    if answer.strip().lower() == "no":
        return (
            f"answer={answer!r} is negative but caption={caption!r} "
            f"explicitly affirms it (Q={question!r})"
        )
    hits = _NEGATION_RE.findall(caption)
    return (
        f"answer={answer!r} is yes-like but caption={caption!r} "
        f"contradicts it (negation {hits}) (Q={question!r})"
    )


def echoes_question_detail(question: str, caption: str) -> str:
    """Human-readable why the caption was treated as an echo of the question."""
    return (
        f"caption={caption!r} just repeats the question {question!r} "
        "instead of stating the answer"
    )


def batch_contamination_detail(caption: str) -> str:
    """Human-readable why batch contamination was suspected."""
    return (
        f"caption={caption!r} looks swapped from another item in the same "
        "LLM batch (near-duplicate or better match to another Q+A)"
    )


def relation_mismatch_detail(question: str, caption: str) -> str:
    """Human-readable why subject/relation overlap looked low (flag only)."""
    ok, ratio = question_relation_preserved(question, caption)
    return (
        f"question content only partly preserved in caption={caption!r} "
        f"(Q={question!r}, overlap_ratio={ratio:.2f}, ok={ok})"
    )


def unsupported_facts_detail(question: str, answer: str, caption: str) -> str:
    """Human-readable why caption added unsupported facts."""
    allowed = content_words(f"{question} {answer}")
    extra = sorted(content_words(caption) - allowed)
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
    """Human-readable why format check rejected a caption."""
    return f"caption={caption!r} failed format check: {reason}"


def flag_detail(flag: str, question: str, answer: str, caption: str) -> str:
    """Human-readable description of a soft validation flag."""
    from validation.checks import FLAG_ANSWER_PARTIAL, FLAG_RELATION_LOW, FLAG_UNSUPPORTED_FACTS

    if flag == FLAG_RELATION_LOW:
        return relation_mismatch_detail(question, caption)
    if flag == FLAG_UNSUPPORTED_FACTS:
        return unsupported_facts_detail(question, answer, caption)
    if flag == "no_answer_without_negation":
        return (
            f"answer={answer!r} but caption={caption!r} has no negation word — "
            "may still be a correct paraphrase, kept for review"
        )
    if flag == FLAG_ANSWER_PARTIAL:
        return answer_mismatch_detail(answer, caption)
    if flag == "overlap_borderline":
        return (
            f"overlap ratio between fail and pass thresholds for "
            f"caption={caption!r} (Q={question!r})"
        )
    return f"{flag}: caption={caption!r}"


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
    if reason == "echoes_question":
        return echoes_question_detail(question, caption)
    if reason == "batch_contamination":
        return batch_contamination_detail(caption)
    if reason == "semantic_fail":
        return semantic_fail_detail(question, answer, caption)
    if reason == "overlap_too_low":
        return (
            f"overlap ratio below fail threshold for caption={caption!r} "
            f"(Q={question!r})"
        )
    return answer_mismatch_detail(answer, caption)
