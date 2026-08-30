"""Validate a batch of generated captions (fast + batched LLM judge)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from validation.config import ValidationConfig
from validation.details import rejection_detail
from validation.fast_validator import FastResult, FastVerdict, fast_validate
from validation.llm_validator import JudgeItem, LlmVerdict, llm_validate_batch


@dataclass
class CaptionValidation:
    """Outcome of validating one generated caption (compatible with legacy API)."""

    ok: bool
    reason: str = "ok"
    flags: List[str] = field(default_factory=list)

    @property
    def needs_semantic_review(self) -> bool:
        return self.ok and self.reason == "needs_semantic_review"


def validate_generated_batch(
    client: Any,
    items: Sequence[Tuple[str, str, str]],
    *,
    config: Optional[ValidationConfig] = None,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    use_llm: bool = True,
) -> List[CaptionValidation]:
    """Fast-validate each caption; batched LLM judge for UNKNOWN items.

    Args:
        client: Ollama client (for LLM judge on UNKNOWN items).
        items: (question, answer, caption) per batch slot.
        config: Validation thresholds.
        batch_pairs: Full batch Q+A for contamination checks.
        batch_captions: Parallel captions for contamination checks.
        use_llm: When False, UNKNOWN → not ok with reason ``needs_llm``.

    Returns:
        One :class:`CaptionValidation` per input item.
    """
    cfg = config or ValidationConfig()
    n = len(items)
    fast_results: List[FastResult] = []
    unknown_positions: List[int] = []

    for i, (question, answer, caption) in enumerate(items):
        fast = fast_validate(
            question,
            answer,
            caption,
            config=cfg,
            batch_pairs=batch_pairs,
            batch_captions=batch_captions,
            self_index=i,
        )
        fast_results.append(fast)
        if fast.verdict == FastVerdict.UNKNOWN:
            unknown_positions.append(i)

    llm_pass: dict[int, bool] = {}
    if use_llm and client is not None and unknown_positions:
        judge_items = [
            JudgeItem(
                index=pos,
                question=items[pos][0],
                answer=items[pos][1],
                caption=items[pos][2],
            )
            for pos in unknown_positions
        ]
        for jr in llm_validate_batch(client, judge_items, config=cfg):
            llm_pass[jr.index] = jr.verdict == LlmVerdict.PASS

    outcomes: List[CaptionValidation] = []
    for i, fast in enumerate(fast_results):
        if fast.verdict == FastVerdict.PASS:
            outcomes.append(CaptionValidation(ok=True, reason="ok", flags=fast.flags))
        elif fast.verdict == FastVerdict.FAIL:
            reason = fast.reasons[0] if fast.reasons else "fast_fail"
            outcomes.append(
                CaptionValidation(ok=False, reason=reason, flags=fast.flags)
            )
        else:
            if use_llm and client is not None:
                if llm_pass.get(i, False):
                    outcomes.append(
                        CaptionValidation(ok=True, reason="ok", flags=fast.flags)
                    )
                else:
                    outcomes.append(
                        CaptionValidation(
                            ok=False,
                            reason="semantic_fail",
                            flags=fast.flags,
                        )
                    )
            else:
                outcomes.append(
                    CaptionValidation(
                        ok=True,
                        reason="needs_semantic_review",
                        flags=fast.flags,
                    )
                )
    return outcomes
