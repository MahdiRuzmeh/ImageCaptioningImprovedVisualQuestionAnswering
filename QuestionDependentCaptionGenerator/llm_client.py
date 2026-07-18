"""Ollama client baraye yek Mistral: sequential ya concurrent API request."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, Tuple

from llm_prompts import chat_messages


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


def parse_caption_list(raw: str, expected: int) -> Optional[List[str]]:
    """JSON array caption ha ro parse kon; length bayad == expected bashe.

    Returns:
        list caption ya None age parse fail she.
    """
    text = _strip_fences(raw)
    # Age model text ezafe gofte, avalin [ ... ] ro bardar
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != expected:
        return None
    out: List[str] = []
    for item in data:
        if not isinstance(item, str):
            return None
        cap = " ".join(item.strip().split())
        if not cap:
            return None
        # Model gahi "Caption:" ya "->" mizare — pak kon
        cap = re.sub(r"^(caption|output|result)\s*:\s*", "", cap, flags=re.I)
        cap = re.sub(r"^->\s*", "", cap).strip()
        if not cap:
            return None
        # Cap toolani: max ~30 kalame
        words = cap.split()
        if len(words) > 30:
            cap = " ".join(words[:30]).rstrip(".,;") + "."
        out.append(cap)
    return out


# Yes/no answers: caption declarative bashe, lazem nist "yes" toye jomle bashe
_YES = {"yes", "yeah", "yep", "true", "maybe"}
_NO = {"no", "none", "0", "zero", "n/a", "not", "nothing"}


def answer_in_caption(answer: str, caption: str) -> bool:
    """Check mikone javab toye caption hast; yes/no joda handle mishe."""
    a = answer.strip().lower()
    c = caption.strip().lower()
    if not a or not c:
        return False
    # Yes/no: jomle gheyre-khali + toolani nabashe kafiye
    if a in _YES or a in _NO:
        return len(c.split()) <= 30
    if a in c:
        return True
    tokens = [t for t in re.split(r"\W+", a) if t]
    if not tokens:
        return False
    return all(t in c for t in tokens)


# ---------------------------------------------------------------------------
# Ollama HTTP
# ---------------------------------------------------------------------------


class OllamaClient:
    """Client sade baraye Ollama chat API (yek model, 8GB-friendly)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mistral",
        num_ctx: int = 1024,
        temperature: float = 0.0,
        timeout_s: float = 300.0,
    ) -> None:
        """Host va model ro set mikone; options baraye VRAM kam."""
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _num_predict(self, batch_size: int) -> int:
        """Max token output: ~12 token per caption + buffer."""
        return max(64, batch_size * 16 + 32)

    def chat_captions(self, pairs: Sequence[Tuple[str, str]]) -> Optional[List[str]]:
        """Yek packed batch Q+A mifreste; list caption bargardune ya None."""
        if not pairs:
            return []
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
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        content = ""
        msg = raw.get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        if not content:
            return None
        return parse_caption_list(content, expected=len(pairs))

    def captions_with_retry(self, pairs: Sequence[Tuple[str, str]]) -> List[Optional[str]]:
        """Batch ro try kon; age fail → har item single retry.

        Returns:
            list ba len == len(pairs); None yani LLM fail shode.
        """
        pairs_list = list(pairs)
        n = len(pairs_list)
        result: List[Optional[str]] = [None] * n

        batch = self.chat_captions(pairs_list)
        if batch is not None:
            for i, cap in enumerate(batch):
                q, a = pairs_list[i]
                if answer_in_caption(a, cap):
                    result[i] = cap
            # Age hame ok → done
            if all(c is not None for c in result):
                return result

        # Single retry baraye missing / failed
        for i, (q, a) in enumerate(pairs_list):
            if result[i] is not None:
                continue
            single = self.chat_captions([(q, a)])
            if single and answer_in_caption(a, single[0]):
                result[i] = single[0]
        return result


def run_batches_concurrent(
    client: OllamaClient,
    batches: Sequence[Sequence[Tuple[str, str]]],
    workers: int = 1,
    on_batch_done: Optional[Callable[[int, List[Optional[str]]], None]] = None,
) -> List[List[Optional[str]]]:
    """Chand packed batch ro sequential ya ba ThreadPool mifreste.

    Args:
        client: OllamaClient (yek model)
        batches: list of Q+A batches
        workers: concurrent API request (1 = sequential, 8GB safe)
        on_batch_done: callback(batch_index, captions) bad az har batch
    """
    n = len(batches)
    out: List[List[Optional[str]]] = [[] for _ in range(n)]
    workers = max(1, int(workers))

    if workers == 1:
        for i, batch in enumerate(batches):
            caps = client.captions_with_retry(batch)
            out[i] = caps
            if on_batch_done is not None:
                on_batch_done(i, caps)
        return out

    def _job(idx: int, batch: Sequence[Tuple[str, str]]) -> Tuple[int, List[Optional[str]]]:
        return idx, client.captions_with_retry(batch)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, i, b) for i, b in enumerate(batches)]
        for fut in as_completed(futs):
            idx, caps = fut.result()
            out[idx] = caps
            if on_batch_done is not None:
                on_batch_done(idx, caps)
    return out
