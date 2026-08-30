"""Fast (lexical) validator: PASS, FAIL, or UNKNOWN without calling an LLM.

The fast layer never decides semantic correctness. It only accepts or rejects
captions when confidence is very high; everything else goes to the LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from validation.checks import (
    FLAG_OVERLAP_BORDERLINE,
    caption_hard_reject_reason,
    caption_soft_flags,
    is_semantically_suspicious,
    question_relation_preserved,
)
from validation.config import ValidationConfig
from validation.overlap import compute_overlap_ratio, overlap_verdict


class FastVerdict(str, Enum):
    """Three-class fast validator outcome."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass
class FastResult:
    """Outcome of :func:`fast_validate` on one caption."""

    verdict: FastVerdict
    reasons: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    overlap_ratio: float = 1.0

    @property
    def is_pass(self) -> bool:
        return self.verdict == FastVerdict.PASS

    @property
    def is_fail(self) -> bool:
        return self.verdict == FastVerdict.FAIL

    @property
    def needs_llm(self) -> bool:
        return self.verdict == FastVerdict.UNKNOWN


def fast_validate(
    question: str,
    answer: str,
    caption: str,
    *,
    config: Optional[ValidationConfig] = None,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    self_index: int = -1,
) -> FastResult:
    """Run the fast validator on one (question, answer, caption) triple.

    Decision tree:
      1. Format + hard rejects → FAIL
      2. Overlap below fail threshold → FAIL
      3. Overlap pass band + no soft flags + not suspicious → PASS
      4. Otherwise → UNKNOWN (escalate to LLM judge)

    Args:
        question: VQA question text.
        answer: Mode answer string.
        caption: Caption to validate.
        config: Thresholds (defaults from :class:`ValidationConfig`).
        batch_pairs: Optional batch context for contamination check.
        batch_captions: Parallel captions for contamination check.
        self_index: Index of this item in the batch.

    Returns:
        :class:`FastResult` with verdict, machine-readable reasons, and flags.
    """
    cfg = config or ValidationConfig()
    reasons: List[str] = []

    # ---------------------------------------------------------------------------
    # Hard rejects (format, echo, polarity, verbatim answer, contamination)
    # ---------------------------------------------------------------------------
    hard = caption_hard_reject_reason(
        answer,
        caption,
        question,
        config=cfg,
        batch_pairs=batch_pairs,
        batch_captions=batch_captions,
        self_index=self_index,
    )
    if hard is not None:
        return FastResult(
            verdict=FastVerdict.FAIL,
            reasons=[hard],
            overlap_ratio=compute_overlap_ratio(question, caption),
        )

    # ---------------------------------------------------------------------------
    # 1.4 Asymmetric overlap
    # ---------------------------------------------------------------------------
    band, ratio = overlap_verdict(question, caption, cfg)
    if band == "fail":
        return FastResult(
            verdict=FastVerdict.FAIL,
            reasons=["overlap_too_low"],
            overlap_ratio=ratio,
        )

    flags = caption_soft_flags(
        question, answer, caption, relation_min_ratio=cfg.relation_min_ratio
    )
    if band == "borderline":
        if FLAG_OVERLAP_BORDERLINE not in flags:
            flags = list(flags) + [FLAG_OVERLAP_BORDERLINE]

    _rel_ok, rel_ratio = (
        question_relation_preserved(
            question, caption, relation_min_ratio=cfg.relation_min_ratio
        )
        if question.strip()
        else (True, 1.0)
    )

    if flags or is_semantically_suspicious(
        question, answer, caption, relation_ratio=rel_ratio
    ):
        return FastResult(
            verdict=FastVerdict.UNKNOWN,
            reasons=sorted(set(flags)),
            flags=flags,
            overlap_ratio=ratio,
        )

    if band == "pass":
        return FastResult(
            verdict=FastVerdict.PASS,
            reasons=[],
            flags=[],
            overlap_ratio=ratio,
        )

    # borderline without flags/suspicion still unknown
    return FastResult(
        verdict=FastVerdict.UNKNOWN,
        reasons=[FLAG_OVERLAP_BORDERLINE],
        flags=flags,
        overlap_ratio=ratio,
    )
