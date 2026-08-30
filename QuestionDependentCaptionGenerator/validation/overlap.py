"""Question–caption overlap ratio for the fast validator.

Formula: ``|required_question_stems ∩ caption_stems| / |required_question_stems|``
with light stemming and wh-category exclusion (see :mod:`validation.tokens`).
"""

from __future__ import annotations

from typing import Tuple

from validation.config import ValidationConfig
from validation.tokens import content_words, required_question_stems


def compute_overlap_ratio(question: str, caption: str) -> float:
    """Return the share of required question stems present in the caption.

    Args:
        question: VQA question text.
        caption: Generated declarative caption.

    Returns:
        Overlap ratio in [0.0, 1.0]. Returns 1.0 when the question has no
        required stems.
    """
    q_stems = required_question_stems(question)
    if not q_stems:
        return 1.0
    c_stems = content_words(caption)
    if not c_stems:
        return 0.0
    return len(q_stems & c_stems) / len(q_stems)


def overlap_verdict(
    question: str,
    caption: str,
    config: ValidationConfig,
) -> Tuple[str, float]:
    """Classify overlap as fail-band, pass-band, or borderline.

    Returns:
        (band, ratio) where band is ``'fail'``, ``'pass'``, or ``'borderline'``.
    """
    ratio = compute_overlap_ratio(question, caption)
    if ratio < config.overlap_fail_threshold:
        return "fail", ratio
    if ratio >= config.overlap_pass_threshold:
        return "pass", ratio
    return "borderline", ratio
