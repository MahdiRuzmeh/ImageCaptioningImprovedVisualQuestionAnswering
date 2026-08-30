"""Ollama client for packed caption generation and validation integration.

HTTP chat + caption parsing live here; validation rules live in
:mod:`validation`.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from llm_prompts import chat_messages
from validation import (
    VALIDATOR_VERSION,
    ValidationConfig,
    _VALIDATION_FAIL_REASONS,
    caption_hard_reject_reason,
    caption_soft_flags,
    rejection_detail,
    validate_generated_batch,
)
from validation.batch_integration import CaptionValidation

# Re-export for backward compatibility
RELATION_MIN_RATIO = 0.5

__all__ = [
    "VALIDATOR_VERSION",
    "RELATION_MIN_RATIO",
    "OllamaClient",
    "ItemOutcome",
    "ChatResult",
    "ParseResult",
    "parse_caption_list",
    "run_batches_concurrent",
    "caption_hard_reject_reason",
    "caption_soft_flags",
    "rejection_detail",
    "_VALIDATION_FAIL_REASONS",
]

# Alias legacy Validation dataclass name
Validation = CaptionValidation


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
    """Parse model response into ``expected`` caption strings."""
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")

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
    flags: List[str] = field(default_factory=list)
    first_caption: Optional[str] = None
    first_reason: str = ""
    retry_kind: str = ""


# ---------------------------------------------------------------------------
# Ollama HTTP
# ---------------------------------------------------------------------------


class OllamaClient:
    """Client for Ollama chat API (single model, 8GB-friendly)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mistral",
        num_ctx: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 300.0,
        validation_config: Optional[ValidationConfig] = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.validation_config = validation_config or ValidationConfig()

    def _num_predict(self, batch_size: int) -> int:
        """Max token output: ~40 token per caption + buffer (JSON overhead)."""
        return max(128, batch_size * 40 + 64)

    def chat_captions(self, pairs: Sequence[Tuple[str, str]]) -> ChatResult:
        """Send one packed batch of Q+A; return captions or a fail reason."""
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

    def _validate_batch_captions(
        self,
        pairs_list: List[Tuple[str, str]],
        captions: List[str],
        *,
        tentative: Optional[List[Optional[str]]] = None,
    ) -> List[CaptionValidation]:
        """Run fast + batched LLM validation on generated captions."""
        items = [
            (pairs_list[i][0], pairs_list[i][1], captions[i])
            for i in range(len(captions))
        ]
        batch_caps = tentative if tentative is not None else captions
        return validate_generated_batch(
            self,
            items,
            config=self.validation_config,
            batch_pairs=pairs_list,
            batch_captions=batch_caps,
            use_llm=True,
        )

    def captions_with_retry(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        single_retries: int = 1,
    ) -> List[ItemOutcome]:
        """Batch try, then per-item single retries with reasons."""
        pairs_list = list(pairs)
        n = len(pairs_list)
        out: List[ItemOutcome] = [
            ItemOutcome(reason="pending", detail="not attempted") for _ in range(n)
        ]

        batch = self.chat_captions(pairs_list)
        tentative: List[Optional[str]] = [None] * n

        if batch.captions is not None:
            validations = self._validate_batch_captions(
                pairs_list, batch.captions
            )
            for i, (cap, result) in enumerate(zip(batch.captions, validations)):
                q, a = pairs_list[i]
                if result.ok:
                    tentative[i] = cap
                else:
                    out[i] = ItemOutcome(
                        reason=result.reason,
                        detail=rejection_detail(result.reason, a, cap, q),
                        attempts=[f"batch:{result.reason}"],
                        flags=result.flags,
                        first_caption=cap,
                        first_reason=result.reason,
                        retry_kind="validator",
                    )

            # Batch contamination pass on tentative accepts
            if any(tentative):
                post_validations = self._validate_batch_captions(
                    pairs_list,
                    [c or "" for c in tentative],
                    tentative=tentative,
                )
                for i, cap in enumerate(tentative):
                    if cap is None:
                        continue
                    q, a = pairs_list[i]
                    result = post_validations[i]
                    if result.ok:
                        out[i] = ItemOutcome(
                            caption=cap,
                            reason="ok",
                            detail="accepted from batch",
                            attempts=["batch:ok"],
                            flags=result.flags,
                        )
                    else:
                        prev = out[i]
                        out[i] = ItemOutcome(
                            reason=result.reason,
                            detail=rejection_detail(result.reason, a, cap, q),
                            attempts=prev.attempts or [f"batch:{result.reason}"],
                            flags=result.flags,
                            first_caption=prev.first_caption or cap,
                            first_reason=prev.first_reason or result.reason,
                            retry_kind=prev.retry_kind or "validator",
                        )
                        tentative[i] = None
        else:
            for i in range(n):
                out[i] = ItemOutcome(
                    reason=batch.reason,
                    detail=batch.detail,
                    attempts=[f"batch:{batch.reason}"],
                    first_reason=batch.reason,
                    retry_kind="generation",
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
                results = self._validate_batch_captions([(q, a)], [cap])
                result = results[0]
                if result.ok:
                    out[i] = ItemOutcome(
                        caption=cap,
                        reason="ok",
                        detail=f"accepted from {tag}",
                        attempts=last.attempts + [f"{tag}:ok"],
                        flags=result.flags,
                        first_caption=last.first_caption,
                        first_reason=last.first_reason,
                        retry_kind=last.retry_kind or "generation",
                    )
                    break
                last.attempts.append(f"{tag}:{result.reason}")
                last.reason = result.reason
                last.detail = rejection_detail(result.reason, a, cap, q)
                last.flags = result.flags
                if last.first_caption is None:
                    last.first_caption = cap
                    last.first_reason = result.reason
                    last.retry_kind = last.retry_kind or "validator"
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
    """Process packed batches sequentially or with a thread pool."""
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
