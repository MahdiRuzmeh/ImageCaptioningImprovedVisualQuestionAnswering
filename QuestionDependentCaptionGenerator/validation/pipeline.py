"""Orchestrate fast + LLM validation over caption rows and aggregate stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from validation.config import ValidationConfig
from validation.fast_validator import FastResult, FastVerdict, fast_validate
from validation.llm_validator import JudgeItem, LlmVerdict, llm_validate_batch
from validation.logging import CaptionTraceEntry, ValidationLogWriter, ValidationTrace


@dataclass
class ValidationStats:
    """Counters for one validation run."""

    fast_pass_count: int = 0
    fast_fail_count: int = 0
    fast_unknown_count: int = 0
    llm_pass_count: int = 0
    llm_fail_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "fast_pass_count": self.fast_pass_count,
            "fast_fail_count": self.fast_fail_count,
            "fast_unknown_count": self.fast_unknown_count,
            "llm_pass_count": self.llm_pass_count,
            "llm_fail_count": self.llm_fail_count,
        }


@dataclass
class RowValidationOutcome:
    """Validation result for one dataset row."""

    kept: bool
    trace: ValidationTrace
    flags: List[str] = field(default_factory=list)


def _source_from_rule(rule: str) -> str:
    if rule == "needs_llm":
        return "pending"
    if rule == "llm_fallback":
        return "llm_fallback"
    return rule or "unknown"


def validate_single_row(
    row: Dict[str, Any],
    *,
    config: Optional[ValidationConfig] = None,
    client: Any = None,
    use_llm: bool = True,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    self_index: int = -1,
) -> RowValidationOutcome:
    """Validate one row's caption through fast (+ optional LLM) layers."""
    cfg = config or ValidationConfig()
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    caption = str(row.get("caption") or "")
    rule = str(row.get("rule") or "")

    trace = ValidationTrace(
        question_id=int(row.get("question_id", 0)),
        image_id=int(row.get("image_id", 0)),
        question=question,
        answer=answer,
        rule=rule,
        captions_trace=[
            CaptionTraceEntry(
                stage="generation",
                caption=caption,
                source=_source_from_rule(rule),
            )
        ],
    )

    fast = fast_validate(
        question,
        answer,
        caption,
        config=cfg,
        batch_pairs=batch_pairs,
        batch_captions=batch_captions,
        self_index=self_index,
    )
    trace.fast_verdict = fast.verdict.value
    trace.fast_reasons = list(fast.reasons)
    trace.validation_flags = list(fast.flags)

    if fast.verdict == FastVerdict.PASS:
        trace.final_verdict = "PASS"
        return RowValidationOutcome(kept=True, trace=trace, flags=fast.flags)

    if fast.verdict == FastVerdict.FAIL:
        trace.final_verdict = "FAIL"
        return RowValidationOutcome(kept=False, trace=trace, flags=fast.flags)

    # UNKNOWN → LLM judge
    if use_llm and client is not None:
        judge_results = llm_validate_batch(
            client,
            [JudgeItem(index=0, question=question, answer=answer, caption=caption)],
            config=cfg,
        )
        jv = judge_results[0] if judge_results else None
        if jv and jv.verdict == LlmVerdict.PASS:
            trace.llm_verdict = "PASS"
            trace.final_verdict = "PASS"
            return RowValidationOutcome(kept=True, trace=trace, flags=fast.flags)
        trace.llm_verdict = "FAIL"
        trace.final_verdict = "FAIL"
        return RowValidationOutcome(kept=False, trace=trace, flags=fast.flags)

    trace.llm_verdict = None
    trace.final_verdict = "UNKNOWN"
    return RowValidationOutcome(kept=False, trace=trace, flags=fast.flags)


def validate_rows(
    rows: List[Dict[str, Any]],
    *,
    config: Optional[ValidationConfig] = None,
    client: Any = None,
    use_llm: bool = True,
    log_writer: Optional[ValidationLogWriter] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], ValidationStats]:
    """Run validation on all rows; batch LLM judge for UNKNOWN items.

    Args:
        rows: Annotation dicts with question, answer, caption, rule, ids.
        config: Validation thresholds.
        client: Ollama client for LLM judge (required when ``use_llm``).
        use_llm: When False, UNKNOWN rows are dropped (audit-only mode).
        log_writer: Optional JSONL writer; one record per row.

    Returns:
        (kept_rows, failed_row_dicts, stats)
    """
    cfg = config or ValidationConfig()
    stats = ValidationStats()
    kept: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    # First pass: fast validate everything
    fast_results: List[FastResult] = []
    unknown_indices: List[int] = []

    for i, row in enumerate(rows):
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        caption = str(row.get("caption") or "")
        fast = fast_validate(question, answer, caption, config=cfg)
        fast_results.append(fast)
        if fast.verdict == FastVerdict.PASS:
            stats.fast_pass_count += 1
        elif fast.verdict == FastVerdict.FAIL:
            stats.fast_fail_count += 1
        else:
            stats.fast_unknown_count += 1
            unknown_indices.append(i)

    # Batch LLM judge for UNKNOWN
    llm_pass: Dict[int, bool] = {}
    if use_llm and client is not None and unknown_indices:
        batch_size = cfg.llm_batch_size
        for start in range(0, len(unknown_indices), batch_size):
            chunk_idx = unknown_indices[start : start + batch_size]
            items = [
                JudgeItem(
                    index=pos,
                    question=str(rows[pos]["question"]),
                    answer=str(rows[pos]["answer"]),
                    caption=str(rows[pos]["caption"]),
                )
                for pos in chunk_idx
            ]
            for jr in llm_validate_batch(client, items, config=cfg):
                llm_pass[jr.index] = jr.verdict == LlmVerdict.PASS

    for i, row in enumerate(rows):
        fast = fast_results[i]
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        caption = str(row.get("caption") or "")
        rule = str(row.get("rule") or "")

        trace = ValidationTrace(
            question_id=int(row.get("question_id", 0)),
            image_id=int(row.get("image_id", 0)),
            question=question,
            answer=answer,
            rule=rule,
            captions_trace=[
                CaptionTraceEntry(
                    stage="generation",
                    caption=caption,
                    source=_source_from_rule(rule),
                )
            ],
            fast_verdict=fast.verdict.value,
            fast_reasons=list(fast.reasons),
            validation_flags=list(fast.flags),
        )

        if fast.verdict == FastVerdict.PASS:
            trace.final_verdict = "PASS"
            out_row = dict(row)
            if fast.flags:
                out_row["validation_flags"] = sorted(fast.flags)
            else:
                out_row.pop("validation_flags", None)
            kept.append(out_row)
        elif fast.verdict == FastVerdict.FAIL:
            trace.final_verdict = "FAIL"
            failed.append(row)
            if log_writer:
                log_writer.write(trace)
            continue
        else:
            passed = llm_pass.get(i, False) if use_llm and client else False
            if passed:
                stats.llm_pass_count += 1
                trace.llm_verdict = "PASS"
                trace.final_verdict = "PASS"
                out_row = dict(row)
                if fast.flags:
                    out_row["validation_flags"] = sorted(fast.flags)
                kept.append(out_row)
            else:
                stats.llm_fail_count += 1
                trace.llm_verdict = "FAIL" if use_llm and client else None
                trace.final_verdict = "FAIL"
                failed.append(row)

        if log_writer:
            log_writer.write(trace)

    return kept, failed, stats
