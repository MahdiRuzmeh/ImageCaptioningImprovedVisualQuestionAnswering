"""Two-layer caption validator: fast PASS/FAIL/UNKNOWN + batched LLM judge.

Public API for :mod:`generate` and the standalone CLI.
"""

from validation.batch_integration import CaptionValidation, validate_generated_batch
from validation.checks import (
    FLAG_ANSWER_PARTIAL,
    FLAG_NO_ANSWER_WITHOUT_NEGATION,
    FLAG_OVERLAP_BORDERLINE,
    FLAG_RELATION_LOW,
    FLAG_UNSUPPORTED_FACTS,
    _VALIDATION_FAIL_REASONS,
    caption_hard_reject_reason,
    caption_soft_flags,
)
from validation.config import VALIDATOR_VERSION, ValidationConfig
from validation.details import rejection_detail
from validation.fast_validator import FastResult, FastVerdict, fast_validate
from validation.llm_validator import JudgeItem, JudgeResult, LlmVerdict, llm_validate_batch
from validation.logging import (
    CaptionTraceEntry,
    ValidationLogWriter,
    ValidationTrace,
    validation_log_path,
)
from validation.overlap import compute_overlap_ratio
from validation.pipeline import RowValidationOutcome, ValidationStats, validate_rows

# Legacy alias used by generate.py
VALIDATION_FAIL_REASONS = _VALIDATION_FAIL_REASONS
RELATION_MIN_RATIO = 0.5

__all__ = [
    "VALIDATOR_VERSION",
    "ValidationConfig",
    "FastVerdict",
    "FastResult",
    "fast_validate",
    "validate_rows",
    "validate_generated_batch",
    "CaptionValidation",
    "llm_validate_batch",
    "JudgeItem",
    "JudgeResult",
    "LlmVerdict",
    "ValidationTrace",
    "ValidationLogWriter",
    "ValidationStats",
    "validation_log_path",
    "caption_hard_reject_reason",
    "caption_soft_flags",
    "rejection_detail",
    "compute_overlap_ratio",
    "RELATION_MIN_RATIO",
    "_VALIDATION_FAIL_REASONS",
    "FLAG_RELATION_LOW",
    "FLAG_UNSUPPORTED_FACTS",
    "FLAG_NO_ANSWER_WITHOUT_NEGATION",
    "FLAG_ANSWER_PARTIAL",
    "FLAG_OVERLAP_BORDERLINE",
]
