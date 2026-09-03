"""Thresholds and defaults for the two-layer caption validator.

Tune ``overlap_fail_threshold`` / ``overlap_pass_threshold`` on a pilot JSON
before a full train run (see ``validation/cli.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Bumped whenever fast/LLM validator rules or defaults change.
VALIDATOR_VERSION = "v5_judge_rules_few_shot"


@dataclass(frozen=True)
class ValidationConfig:
    """All configurable knobs for fast + LLM validation.

    Attributes:
        min_words: Captions with fewer words fail format check (default 3).
        max_words: Captions with more words fail format check (default 30).
        overlap_fail_threshold: Below this ratio → fast FAIL (default 0.30).
        overlap_pass_threshold: At/above + hard checks → fast PASS (default 0.50).
        llm_batch_size: Items per batched LLM judge call (default 10).
        relation_min_ratio: Legacy alias used in overlap computation (0.50).
    """

    min_words: int = 3
    max_words: int = 30
    overlap_fail_threshold: float = 0.30
    overlap_pass_threshold: float = 0.50
    llm_batch_size: int = 10
    relation_min_ratio: float = 0.50

    def __post_init__(self) -> None:
        if self.min_words < 1:
            raise ValueError("min_words must be >= 1")
        if self.max_words < self.min_words:
            raise ValueError("max_words must be >= min_words")
        if not 0.0 <= self.overlap_fail_threshold <= 1.0:
            raise ValueError("overlap_fail_threshold must be in [0, 1]")
        if not 0.0 <= self.overlap_pass_threshold <= 1.0:
            raise ValueError("overlap_pass_threshold must be in [0, 1]")
        if self.overlap_fail_threshold > self.overlap_pass_threshold:
            raise ValueError(
                "overlap_fail_threshold must be <= overlap_pass_threshold"
            )
        if self.llm_batch_size < 1:
            raise ValueError("llm_batch_size must be >= 1")
