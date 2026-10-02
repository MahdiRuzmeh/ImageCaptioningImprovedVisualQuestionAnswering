"""Fast (lexical) validator: FAIL or UNKNOWN without calling an LLM.

The fast layer never decides semantic correctness. It only hard-rejects
captions when confidence is very high; everything else goes to the LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from validation.checks import (
    FLAG_OVERLAP_BORDERLINE,
    FLAG_OVERLAP_TOO_LOW,
    caption_hard_reject_reason,
    caption_soft_flags,
)
from validation.config import ValidationConfig
from validation.overlap import compute_overlap_ratio, overlap_verdict


class FastVerdict(str, Enum):
    """Two-class fast validator outcome (no PASS — meaning is LLM-only)."""

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
        """Always False — fast layer never auto-accepts."""
        return False

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
      2. Otherwise → UNKNOWN (escalate to LLM judge), with soft flags
         including low/borderline overlap for the judge / audit trail.

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

    band, ratio = overlap_verdict(question, caption, cfg)
    flags = caption_soft_flags(
        question, answer, caption, relation_min_ratio=cfg.relation_min_ratio
    )
    if band == "fail":
        if FLAG_OVERLAP_TOO_LOW not in flags:
            flags = list(flags) + [FLAG_OVERLAP_TOO_LOW]
    elif band == "borderline":
        if FLAG_OVERLAP_BORDERLINE not in flags:
            flags = list(flags) + [FLAG_OVERLAP_BORDERLINE]

    return FastResult(
        verdict=FastVerdict.UNKNOWN,
        reasons=sorted(set(flags)),
        flags=flags,
        overlap_ratio=ratio,
    )
