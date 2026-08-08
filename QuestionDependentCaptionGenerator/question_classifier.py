"""Two-stage filter for subjective / personal / OCR / commonsense questions.

Stage 1: cheap keyword/regex candidate detection.
Stage 2: Qwen (Ollama) classifies only candidates into one of:
    VISUAL | SUBJECTIVE_PERSONAL | COMMONSENSE | OCR

Non-VISUAL candidates are dropped before caption generation.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

CLASSIFIER_PROMPT_VERSION = "v1_four_way_visual_filter"

QUESTION_LABELS = (
    "VISUAL",
    "SUBJECTIVE_PERSONAL",
    "COMMONSENSE",
    "OCR",
)

_CANDIDATE_RE = re.compile(
    r"""
    \bhave\s+you\s+ever\b |
    \bwould\s+you\s+(prefer|like|want)\b |
    \bdo\s+you\s+(like|think|want|prefer)\b |
    \bwould\s+you\b |
    \bdo\s+you\b |
    \b(safe|healthy|nutritious|beautiful|comfortable|dangerous|
       expensive|valuable|tasty|delicious|attractive|ugly)\b
    """,
    re.I | re.X,
)

_SYSTEM_PROMPT = (
    "Classify the following VQA question into exactly one category: "
    "VISUAL, SUBJECTIVE_PERSONAL, COMMONSENSE, OCR. "
    "Return only the category label."
)


def is_subjective_candidate(question: str) -> bool:
    """True if the question should be sent to the 4-way classifier."""
    return bool(_CANDIDATE_RE.search(question or ""))


def parse_classifier_label(raw: str) -> Optional[str]:
    """Extract a valid label from a model response."""
    text = (raw or "").strip().upper()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    # First token / line often is the label
    first = re.split(r"[\s,.:;]+", text)[0] if text else ""
    for label in QUESTION_LABELS:
        if first == label or label in text.split():
            return label
        if text.startswith(label):
            return label
    return None


class QuestionClassifier:
    """Ollama-backed 4-way question classifier (captioning-free)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        num_ctx: int = 1024,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.num_ctx = num_ctx

    def classify_one(self, question: str) -> Tuple[Optional[str], str]:
        """Classify one question. Returns (label_or_None, detail)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Classify the following VQA question into exactly one "
                        "category: VISUAL, SUBJECTIVE_PERSONAL, COMMONSENSE, OCR.\n"
                        "Return only the category.\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": 16,
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
            return None, f"http_error:{exc.code}"
        except urllib.error.URLError as exc:
            return None, f"connection_error:{exc.reason}"
        except TimeoutError:
            return None, "timeout"
        except json.JSONDecodeError as exc:
            return None, f"json_error:{exc}"

        content = ""
        msg = raw.get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        label = parse_classifier_label(content)
        if label is None:
            return None, f"parse_fail:{content!r}"
        return label, "ok"

    def metadata(self) -> Dict[str, str]:
        """Reproducibility fields for output JSON info."""
        return {
            "model": self.model,
            "host": self.host,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
        }


def filter_non_visual_questions(
    rows: Sequence[Dict],
    classifier: Optional[QuestionClassifier] = None,
    *,
    offline_drop_candidates: bool = False,
) -> Tuple[List[Dict], int, int, Dict[str, int]]:
    """Drop SUBJECTIVE_PERSONAL / COMMONSENSE / OCR candidates from ``rows``.

    Args:
        rows: caption rows (dicts with at least ``question``).
        classifier: Ollama classifier; if None and ``offline_drop_candidates``
            is False, candidates are kept (no filtering).
        offline_drop_candidates: when True and classifier is unavailable,
            drop all regex candidates (conservative offline mode).

    Returns:
        (kept_rows, subjective_excluded, classifier_ocr_excluded, label_counts)
    """
    kept: List[Dict] = []
    subjective_excluded = 0
    classifier_ocr_excluded = 0
    label_counts: Dict[str, int] = {lab: 0 for lab in QUESTION_LABELS}
    label_counts["CANDIDATE_KEPT"] = 0
    label_counts["NON_CANDIDATE"] = 0

    for row in rows:
        q = str(row.get("question") or "")
        if not is_subjective_candidate(q):
            kept.append(row)
            label_counts["NON_CANDIDATE"] += 1
            continue

        if classifier is None:
            if offline_drop_candidates:
                subjective_excluded += 1
                continue
            kept.append(row)
            label_counts["CANDIDATE_KEPT"] += 1
            continue

        label, _detail = classifier.classify_one(q)
        if label is None:
            # Fail-open for VISUAL safety? Prefer fail-closed on candidates:
            # drop when classification fails so personal Qs don't leak in.
            subjective_excluded += 1
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        if label == "VISUAL":
            kept.append(row)
        elif label == "OCR":
            classifier_ocr_excluded += 1
        else:
            subjective_excluded += 1

    return kept, subjective_excluded, classifier_ocr_excluded, label_counts
