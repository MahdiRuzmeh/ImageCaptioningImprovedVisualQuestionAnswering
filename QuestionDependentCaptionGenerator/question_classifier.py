"""Binary filter for questions that are not directly answerable from a static image.

When ``--classify-questions`` is on, questions are labeled:

    DIRECTLY_VISUAL | NOT_DIRECTLY_VISUAL

A **regex fast-path** (``_ALWAYS_VISUAL_RE``) auto-accepts question patterns
that are definitionally visual (color, count, spatial, sport/game,
material, which, doing, animal, visible expression, etc.) without calling
the LLM — typically ~60-70 % of VQA v2 questions.  Only ambiguous
questions go through Qwen (Ollama).

``DIRECTLY_VISUAL`` means the answer can be extracted from the appearance of
a static image without OCR, outside knowledge, or personal opinion/preference.

Offline ``--drop-subjective-candidates`` still uses a cheap regex candidate
gate (no LLM) and drops matching rows as NOT_DIRECTLY_VISUAL.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

CLASSIFIER_PROMPT_VERSION = "v4_sport_action_material"

QUESTION_LABELS = (
    "DIRECTLY_VISUAL",
    "NOT_DIRECTLY_VISUAL",
)

# Offline / candidate-drop regex (broadened beyond the old subjective-only gate).
_CANDIDATE_RE = re.compile(
    r"""
    \bhave\s+you\s+ever\b |
    \bwould\s+you\s+(prefer|like|want)\b |
    \bdo\s+you\s+(like|think|want|prefer)\b |
    \bwould\s+you\b |
    \bdo\s+you\b |
    \bare\s+you\s+allowed\b |
    \b(safe|healthy|nutritious|beautiful|comfortable|dangerous|
       expensive|valuable|tasty|delicious|attractive|ugly|
       endangered|personality|professional)\b |
    \bwhat\s+is\s+the\s+name\b |
    \bwhat\s+(?:does|do)\s+.+\bsay\b |
    \bwhat\s+is\s+written\b |
    \bdoes\s+this\s+\w+\s+work\b
    """,
    re.I | re.X,
)

_SYSTEM_PROMPT = (
    "Classify each VQA question as DIRECTLY_VISUAL or NOT_DIRECTLY_VISUAL.\n"
    "DIRECTLY_VISUAL: the answer must be directly extractable from the "
    "appearance of a static image, without OCR (reading text/numbers on "
    "signs, logos, names, plates), without outside knowledge, and without "
    "personal opinion or preference.\n"
    "Questions about visible properties (color, shape, count, clothing, "
    "posture, spatial position, weather, approximate age, visible actions, "
    "visible similarity, visible material/made-of, visible animals, "
    "visible expression) are DIRECTLY_VISUAL even if they require simple "
    "visual inference.\n"
    "Identifying a visible sport, game, or activity from appearance "
    "(tennis, skateboarding, polo, etc.) is DIRECTLY_VISUAL. "
    "Rules of a sport, professionalism, legality, or personal preference "
    "about a sport are NOT_DIRECTLY_VISUAL.\n"
    "'Which X has <visible attribute>?' is DIRECTLY_VISUAL.\n"
    "NOT_DIRECTLY_VISUAL: anything else (subjective, OCR, commonsense, "
    "sport rules / professionalism, mechanical function not visible, "
    "etc.).\n"
    "Return only the category label."
)

_USER_PROMPT_TEMPLATE = (
    "Classify the following VQA question as DIRECTLY_VISUAL or "
    "NOT_DIRECTLY_VISUAL.\n"
    "DIRECTLY_VISUAL means the answer is directly extractable from a static "
    "image without OCR, outside knowledge, or personal opinion.\n"
    "Identifying a visible sport/game/activity is DIRECTLY_VISUAL; "
    "sport rules or professionalism are NOT.\n"
    "Return only the category label.\n\n"
    "Examples:\n"
    "Q: What color is the bus?\n"
    "DIRECTLY_VISUAL\n"
    "Q: How many people are sitting?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Is the man wearing a hat?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What is in front of the giraffes?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Is it a cloudy day?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Are they playing polo?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What sport are they playing?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What game is being played?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What is the wall made of?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What are the people doing?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Which player has a white hat?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Does this woman look excited?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What animal does he have?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Do these ski boards have personality?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Are they endangered?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Are you allowed to use your foot in ultimate?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Does this train work?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: What is the name of the hotel?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Is this a professional game?\n"
    "NOT_DIRECTLY_VISUAL\n\n"
    "Question: {question}"
)


# Fast-path: question patterns that are definitionally DIRECTLY_VISUAL.
# Matching questions skip the LLM call entirely, saving ~60-70% of HTTP
# round-trips on typical VQA v2 data.  The gate is conservative: if a
# question also matches _CANDIDATE_RE (subjective / OCR words) it still
# goes through the LLM for a proper ruling.
_ALWAYS_VISUAL_RE = re.compile(
    r"""
    ^what\s+colou?rs?\s+ |                          # What color ...?
    ^how\s+many\s+ |                                 # How many ...?
    ^what\s+(?:sport|game|activity)\s+ |             # What sport/game/activity ...?
    ^which\s+ |                                      # Which ...?
    ^what\s+is\s+(?:the\s+)?\S+(?:\s+\S+){0,3}\s+made\s+of\b |  # What is X made of?
    ^what\s+(?:is|are)\s+.+\s+doing\b |              # What is/are ... doing?
    ^what\s+(?:animal|animals|bird|birds)\s+ |       # What animal/bird ...?
    ^what\s+is\s+(?:the\s+)?(?:painting|picture|photo|image|drawing|poster)\b |  # What is the painting ...?
    ^does\s+(?:this|that|the|he|she|it)\s+.+\s+look\s+ |  # Does X look ...?
    ^this\s+is\s+\w+ |                               # This is tennis?
    ^what\s+(?:do|does)\s+.+\shave\s+in\s+common\b | # What do X have in common?
    ^what\s+(?:kind|type)\s+of\s+ |                  # What kind/type of ...?
    ^is\s+there\s+ |                                 # Is there ...?
    ^are\s+there\s+ |                                # Are there ...?
    ^(?:is|are|was|were)\s+
        (?:the|this|that|these|those|he|she|it|they)\s+ |  # Is/Are the/this/... X?
    ^what\s+is\s+(?:in\s+front\s+of|next\s+to|on\s+top\s+of
        |in|on|at|near|behind|under|over|above|below
        |beside|between|inside|outside)\s+ |         # What is PREP ...?
    ^what\s+is\s+(?:the\s+)?\w+\s+
        (?:wearing|holding|carrying|eating|drinking
        |riding|sitting|standing|playing)\b |         # What is X V-ing?
    ^who\s+(?:is|are)\s+                              # Who is/are ...?
    """,
    re.I | re.X,
)


def is_always_visual(question: str) -> bool:
    """True when the question pattern is definitionally DIRECTLY_VISUAL.

    Returns False if the question also contains subjective / OCR markers
    (``_CANDIDATE_RE``), so ambiguous cases still go to the LLM.
    """
    q = (question or "").strip()
    if not q:
        return False
    if _CANDIDATE_RE.search(q):
        return False
    return bool(_ALWAYS_VISUAL_RE.search(q))


def is_subjective_candidate(question: str) -> bool:
    """True if the question matches the offline drop candidate regex."""
    return bool(_CANDIDATE_RE.search(question or ""))


def parse_classifier_label(raw: str) -> Optional[str]:
    """Extract a valid binary label from a model response."""
    text = (raw or "").strip().upper()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    text = text.replace("-", "_").replace(" ", "_")
    # Prefer longer / more specific label first
    for label in ("NOT_DIRECTLY_VISUAL", "DIRECTLY_VISUAL"):
        if text == label or text.startswith(label):
            return label
        if label in text.split():
            return label
    # Legacy aliases from older four-way outputs (fail-closed mapping)
    legacy = re.split(r"[\s,.:;]+", text)[0] if text else ""
    if legacy in {"VISUAL"}:
        return "DIRECTLY_VISUAL"
    if legacy in {"SUBJECTIVE_PERSONAL", "COMMONSENSE", "OCR", "SUBJECTIVE"}:
        return "NOT_DIRECTLY_VISUAL"
    return None


class QuestionClassifier:
    """Ollama-backed binary question classifier (captioning-free)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        num_ctx: int = 2048,
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
                    "content": _USER_PROMPT_TEMPLATE.format(question=question),
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": 24,
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


def _fresh_label_counts() -> Dict[str, int]:
    """Empty label counter dict for classifier accounting."""
    counts: Dict[str, int] = {lab: 0 for lab in QUESTION_LABELS}
    counts["OFFLINE_CANDIDATE_DROP"] = 0
    counts["PARSE_FAIL_DROP"] = 0
    counts["FAST_PATH_VISUAL"] = 0
    return counts


def _question_id_set(rows: Sequence[Dict[str, Any]]) -> Set[int]:
    """Collect integer question_id values from row dicts."""
    out: Set[int] = set()
    for row in rows:
        try:
            out.add(int(row["question_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_classifier_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    """Load a classifier checkpoint sidecar, or None if missing/corrupt."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    kept = data.get("kept")
    dropped = data.get("dropped")
    if not isinstance(info, dict):
        return None
    if not isinstance(kept, list) or not isinstance(dropped, list):
        return None
    return data


def validate_classifier_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    pre_classify_count: int,
    input_count: int,
    classifier_meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when a checkpoint matches the current run configuration."""
    info = checkpoint.get("info") or {}
    if info.get("prompt_version") != CLASSIFIER_PROMPT_VERSION:
        return False
    if int(info.get("pre_classify_count", -1)) != pre_classify_count:
        return False
    if int(info.get("input_count", -1)) != input_count:
        return False
    if classifier_meta:
        if info.get("model") and info.get("model") != classifier_meta.get("model"):
            return False
        if info.get("host") and info.get("host") != classifier_meta.get("host"):
            return False
    return True


def save_classifier_checkpoint(
    path: Path,
    kept: Sequence[Dict],
    dropped: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
) -> None:
    """Atomically persist classifier progress for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": dict(info),
        "kept": list(kept),
        "dropped": list(dropped),
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp.json",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def delete_classifier_checkpoint(path: Path) -> None:
    """Remove classifier checkpoint after successful main output write."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def filter_non_visual_questions(
    rows: Sequence[Dict],
    classifier: Optional[QuestionClassifier] = None,
    *,
    offline_drop_candidates: bool = False,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
    resume: bool = True,
    classifier_meta: Optional[Dict[str, Any]] = None,
    input_count: int = 0,
) -> Tuple[List[Dict], List[Dict[str, Any]], Dict[str, int]]:
    """Keep DIRECTLY_VISUAL rows; collect NOT_DIRECTLY_VISUAL drops for a sidecar.

    Args:
        rows: caption rows (dicts with at least ``question``).
        classifier: Ollama classifier; when provided, **every** question is
            classified (not only regex candidates).
        offline_drop_candidates: when True and classifier is unavailable,
            drop all regex candidates (conservative offline mode).
        checkpoint_path: optional sidecar for incremental classifier resume.
        checkpoint_every: save checkpoint every N newly classified questions.
        resume: load and continue from ``checkpoint_path`` when valid.
        classifier_meta: model/host/prompt metadata for checkpoint validation.
        input_count: raw VQA input count before OCR/dedup (for validation).

    Returns:
        (kept_rows, dropped_rows, label_counts)

        ``dropped_rows`` entries include ``label`` (and optional ``detail``)
        suitable for ``*_not_directly_visual.json``.
    """
    n_total = len(rows)
    pre_classify_count = n_total

    if classifier is not None and checkpoint_path is not None and resume:
        existing = load_classifier_checkpoint(checkpoint_path)
        if existing and validate_classifier_checkpoint(
            existing,
            pre_classify_count=pre_classify_count,
            input_count=input_count,
            classifier_meta=classifier_meta,
        ):
            info = existing.get("info") or {}
            if info.get("status") == "complete":
                kept = list(existing.get("kept") or [])
                dropped = list(existing.get("dropped") or [])
                label_counts = dict(info.get("label_counts") or _fresh_label_counts())
                print(
                    f"Classifier resume: loaded complete checkpoint "
                    f"({len(kept)} kept, {len(dropped)} dropped) "
                    f"from {checkpoint_path}",
                    flush=True,
                )
                return kept, dropped, label_counts

    kept: List[Dict] = []
    dropped: List[Dict[str, Any]] = []
    label_counts = _fresh_label_counts()
    classified_ids: Set[int] = set()

    if (
        classifier is not None
        and checkpoint_path is not None
        and resume
    ):
        existing = load_classifier_checkpoint(checkpoint_path)
        if existing and validate_classifier_checkpoint(
            existing,
            pre_classify_count=pre_classify_count,
            input_count=input_count,
            classifier_meta=classifier_meta,
        ):
            kept = list(existing.get("kept") or [])
            dropped = list(existing.get("dropped") or [])
            ckpt_info = existing.get("info") or {}
            label_counts = dict(ckpt_info.get("label_counts") or _fresh_label_counts())
            classified_ids = _question_id_set(kept) | _question_id_set(dropped)
            print(
                f"Classifier resume: continuing from checkpoint "
                f"({len(classified_ids)}/{n_total} done) "
                f"-> {checkpoint_path}",
                flush=True,
            )

    ckpt_every = max(1, int(checkpoint_every))

    def _build_checkpoint_info(status: str) -> Dict[str, Any]:
        classified_count = len(classified_ids)
        out: Dict[str, Any] = {
            "status": status,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
            "input_count": input_count,
            "pre_classify_count": pre_classify_count,
            "label_counts": dict(label_counts),
            "classified_count": classified_count,
            "total_to_classify": n_total,
            "post_filter_count": len(kept),
        }
        if classifier_meta:
            out.update(
                {
                    k: classifier_meta[k]
                    for k in ("model", "host")
                    if k in classifier_meta
                }
            )
        return out

    def _maybe_save_checkpoint(status: str, force: bool = False) -> None:
        if checkpoint_path is None or classifier is None:
            return
        classified_count = len(classified_ids)
        if not force and classified_count % ckpt_every != 0:
            return
        save_classifier_checkpoint(
            checkpoint_path,
            kept,
            dropped,
            _build_checkpoint_info(status),
        )

    if classifier is not None and n_total:
        n_fast = sum(
            1 for row in rows
            if is_always_visual(str(row.get("question") or ""))
        )
        print(
            f"Question classifier: {n_total} questions "
            f"(binary DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL), "
            f"{n_fast} fast-path visual (no LLM)...",
            flush=True,
        )
    elif offline_drop_candidates:
        n_cand = sum(
            1
            for row in rows
            if is_subjective_candidate(str(row.get("question") or ""))
        )
        print(
            f"Offline candidates: {n_cand}/{n_total} "
            "(dropping without Qwen)...",
            flush=True,
        )

    progress_every = max(1, min(25, n_total // 20)) if n_total else 1
    done = 0
    llm_calls = 0
    newly_classified = 0

    for row in rows:
        q = str(row.get("question") or "")

        if classifier is None:
            if offline_drop_candidates and is_subjective_candidate(q):
                label_counts["OFFLINE_CANDIDATE_DROP"] += 1
                label_counts["NOT_DIRECTLY_VISUAL"] += 1
                dropped.append(_drop_record(row, "NOT_DIRECTLY_VISUAL", "offline_candidate"))
                continue
            kept.append(row)
            label_counts["DIRECTLY_VISUAL"] += 1
            continue

        try:
            qid = int(row["question_id"])
        except (KeyError, TypeError, ValueError):
            qid = None

        if qid is not None and qid in classified_ids:
            continue

        try:
            done += 1
            if done == 1 or done % progress_every == 0 or done == n_total:
                print(
                    f"  classify progress: {len(classified_ids)}/{n_total} "
                    f"(LLM calls: {llm_calls})",
                    flush=True,
                )

            # Fast-path: skip LLM for obviously visual question patterns
            if is_always_visual(q):
                label_counts["FAST_PATH_VISUAL"] += 1
                label_counts["DIRECTLY_VISUAL"] += 1
                kept.append(row)
                if qid is not None:
                    classified_ids.add(qid)
                newly_classified += 1
                _maybe_save_checkpoint("in_progress")
                continue

            llm_calls += 1
            label, detail = classifier.classify_one(q)
            if label is None:
                label_counts["PARSE_FAIL_DROP"] += 1
                label_counts["NOT_DIRECTLY_VISUAL"] += 1
                dropped.append(
                    _drop_record(row, "NOT_DIRECTLY_VISUAL", detail or "parse_fail")
                )
            else:
                label_counts[label] = label_counts.get(label, 0) + 1
                if label == "DIRECTLY_VISUAL":
                    kept.append(row)
                else:
                    dropped.append(_drop_record(row, label, detail))

            if qid is not None:
                classified_ids.add(qid)
            newly_classified += 1
            _maybe_save_checkpoint("in_progress")
        except KeyboardInterrupt:
            _maybe_save_checkpoint("in_progress", force=True)
            raise

    if classifier is not None and checkpoint_path is not None:
        if newly_classified or len(classified_ids) == n_total:
            _maybe_save_checkpoint("complete", force=True)

    return kept, dropped, label_counts


def _drop_record(
    row: Dict,
    label: str,
    detail: str = "",
) -> Dict[str, Any]:
    """Build a sidecar annotation from a source row."""
    out: Dict[str, Any] = {
        "question_id": row.get("question_id"),
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "answer_count": row.get("answer_count"),
        "answer_consensus": row.get("answer_consensus"),
        "label": label,
    }
    if detail:
        out["detail"] = detail
    return out
