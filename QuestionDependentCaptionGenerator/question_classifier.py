"""Binary filter for questions that are not directly answerable from a static image.

When ``--classify-questions`` is on, questions are labeled:

    DIRECTLY_VISUAL | NOT_DIRECTLY_VISUAL

The gate is a **conservative whitelist**: a question skips the LLM only when
it matches ``_FAST_PATH_VISUAL_RE`` (colour / count / existence / plain
spatial / a small set of always-visual What-shapes) *and* carries no
``_NON_VISUAL_SUSPECT_RE`` marker.  Everything else — judgment, intention,
safety, preference, purpose, general knowledge, and anything merely
phrased like a normal VQA question — reaches Qwen (Ollama) for a real
ruling (UNKNOWN → LLM).

Earlier versions were visual-by-default, which let non-visual questions
("Is this safe?", "Is this place in a particular country?") into the final
captions file without ever being classified.

``DIRECTLY_VISUAL`` means the question can be answered by looking at the
image.  ``NOT_DIRECTLY_VISUAL`` means answering needs rendered text (OCR),
personal opinion/preference, judgment, intention, or knowledge from outside
the picture.

Every classified row records where its decision came from in
``visual_filter_source`` (``fast_path`` or ``llm_classifier``) so error
analysis can separate the two.

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

CLASSIFIER_PROMPT_VERSION = "v7_expanded_fast_path"

QUESTION_LABELS = (
    "DIRECTLY_VISUAL",
    "NOT_DIRECTLY_VISUAL",
)

# Provenance of a DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL decision, stored per
# row as ``visual_filter_source`` for later error analysis.
VISUAL_FILTER_FAST_PATH = "fast_path"
VISUAL_FILTER_LLM = "llm_classifier"

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
    "DIRECTLY_VISUAL (the default): the question can be answered simply by "
    "looking at the image. This covers objects, colors, counts, spatial "
    "positions, actions, materials, room or scene type, sport or activity, "
    "weather, approximate age, and facial expressions — even when simple "
    "visual inference is needed.\n"
    "NOT_DIRECTLY_VISUAL: only when answering requires one of three things "
    "that are not in the pixels:\n"
    "  1. reading rendered text (names on signs, logos, brands, plates, "
    "written words, clock time),\n"
    "  2. personal opinion, taste, or preference,\n"
    "  3. outside-world knowledge (sport rules, legality, species facts, "
    "who manufactured something, prices, animal sounds),\n"
    "  4. judgment or intention: whether something is safe, suitable, "
    "healthy or allowed, what someone or an animal wants or is about to do, "
    "or which country/city a place belongs to.\n"
    "When unsure, answer DIRECTLY_VISUAL.\n"
    "Return only the category label."
)

_USER_PROMPT_TEMPLATE = (
    "Classify the following VQA question as DIRECTLY_VISUAL or "
    "NOT_DIRECTLY_VISUAL.\n"
    "DIRECTLY_VISUAL means the question can be answered just by looking at "
    "the image.\n"
    "NOT_DIRECTLY_VISUAL means answering needs reading text (OCR), personal "
    "opinion, or outside-world knowledge.\n"
    "When unsure, answer DIRECTLY_VISUAL.\n"
    "Return only the category label.\n\n"
    "Examples:\n"
    "Q: What color is the bus?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Are all the flowers white?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Number of animals?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Is a military person in the picture?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What is watching the TV?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What room is this?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Could this photo be from a zoo?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What fruits are here?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Where is the lighthouse located?\n"
    "DIRECTLY_VISUAL\n"
    "Q: Do the animals have ear tags?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What sport are they playing?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What is the wall made of?\n"
    "DIRECTLY_VISUAL\n"
    "Q: What is the name of the hotel?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: What beer is advertised on the window?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: What language is on the signs?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: What sound does this animal make?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Are you allowed to use your foot in ultimate?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Which toilet would you prefer to use?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Is this pizza nutritious enough for a full dinner?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Does this train work?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Is this safe?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Is this equipment suitable for the job?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Does this animal want to eat?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Is this place in a particular country?\n"
    "NOT_DIRECTLY_VISUAL\n"
    "Q: Should the man be riding here?\n"
    "NOT_DIRECTLY_VISUAL\n\n"
    "Question: {question}"
)


# Suspect gate: questions that MIGHT need something beyond the pixels.
#
# Since the Fast Path is now a whitelist, this regex is a *veto* over it: a
# question that looks like a plain visual shape but carries one of these
# markers still goes to the LLM classifier.  The families:
#
#   1. personal opinion / preference        ("Would you prefer...")
#   2. OCR / reading rendered text          ("What is the name of the hotel?")
#   3. outside-world knowledge / causation  ("What sound does this animal make?")
#   4. judgment / intention / modality      ("Is this safe?", "Is it about to
#      rain?", "Is this place in a particular country?")
#
# Note: ``made of`` is intentionally NOT a suspect (visible material) while
# ``who made`` is (maker/brand knowledge).
_NON_VISUAL_SUSPECT_RE = re.compile(
    r"""
    # --- personal / opinion / preference ---
    \b(?:do|would|did|have|can|could)\s+you\b |
    \bdo\s+we\b | \bwould\s+one\b | \byour\b | \bprefer\b | \bfavorite\b |
    \b(?:safe|safety|healthy|nutritious|tasty|delicious|beautiful|ugly|
       attractive|comfortable|dangerous|expensive|valuable|cheap|personality|
       professional|romantic)\b |

    # --- OCR / reading rendered text ---
    \bsays?\b | \bsaying\b | \bwritten\b | \bprinted\b | \bspelled\b |
    \b(?:word|words|letter|letters|initials|caption|slogan|text)\b |
    \bname\s+(?:of|on)\b | \bnamed\b | \bbrand\b | \blogo\b |
    \bcompany\b | \badvertis\w*\b | \bmentioned\b | \blanguage\b |
    \bwhat\s+time\b | \b(?:month|year|date)\b | \blicense\b |
    \bphone\s+number\b | \bwebsite\b | \bscore\b |

    # --- outside-world knowledge / rules / causation ---
    \ballowed\b | \blegal\b | \brules?\b | \bendangered\b |
    \b(?:breed|species)\b | \bsound\s+does\b |
    \bwho\s+(?:made|makes|built|owns|invented)\b | \bmanufactur\w*\b |
    \bwhy\b | \bpurpose\b | \bused\s+for\b | \bmeant\s+for\b | \bfor\?\s*$ |
    \bcost\b | \bprice\b |
    \bwork(?:s|ing)?\s*\?*\s*$ | \bfunction\b | \bpopular\b | \bfamous\b |
    \bwhat\s+will\s+happen\b | \bgoing\s+to\s+happen\b |

    # --- judgment / modality (Comments8: should/would/could/can/think) ---
    \bshould\b | \bwould\b | \bcould\b | \bmight\b | \bmay\b | \bmust\b |
    \bcan\b | \bthink\b | \bsuppose\w*\b | \bseem\w*\b | \bprobably\b |
    \b(?:suitable|appropriate|proper|polite|rude|correct|right|wrong|
       necessary|useful|worth|better|best|good|bad)\b |

    # --- intention / future action ---
    \babout\s+to\b | \bgoing\s+to\b | \bwant(?:s|ed)?\s+to\b |
    \btry(?:ing|s)?\s+to\b | \bplan(?:s|ning)?\s+to\b | \bintend\w*\b |
    \bwill\s+\w+\b | \bnext\b |

    # --- place identity / geography (outside the pixels) ---
    \bcountry\b | \bcity\b | \bstate\b | \bnation\w*\b | \bcontinent\b |
    \bregion\b | \bprovince\b | \bwhich\s+part\s+of\s+the\s+world\b
    """,
    re.I | re.X,
)


# Frequent phrasings that trip a suspect marker while describing something
# plainly visible ("can you see" is perception, not preference; "time of day"
# is daylight, not a clock face; "can be seen" is a VQA counting idiom).
# Removed before the suspect test so they only exempt themselves — "Do you
# see a brand name?" still stays suspect.
_SUSPECT_EXEMPT_RE = re.compile(
    r"""
    \b(?:can|could|do|did|would)\s+you\s+see\b |
    \b(?:can|could)\s+be\s+seen\b |
    \bwhat\s+time\s+of\s+(?:day|year)\b |
    \bnext\s+to\b |
    \b(?:to|on)\s+the\s+right\b |
    \bright\s+side\b |
    \btrash\s+can\b |
    \bcity\s+bus(?:es)?\b |
    \bcan\s+you\s+spot\b
    """,
    re.I | re.X,
)


# Fast Path whitelist: shapes whose answer is read straight off the pixels
# with essentially no ambiguity.  Deliberately narrow — anything not listed
# here is UNKNOWN and pays for an LLM ruling.
#
#   - colour:     "What color is the bus?" / "What colors are the cows?"
#   - counting:   "How many cookies are there?" / "Number of animals?"
#   - existence:  "Is there a clock on the wall?" / "Do you see a boat?"
#   - scene type: "What sport/room/animal/food/…"
#   - spatial:    plain is/are DET NP PREP DET noun; "What is under the table?"
#   - action:     end-anchored "What is the man doing/holding/wearing?"
#   - sky:        "Is the sky clear?"
#
# Not whitelisted (stay UNKNOWN → LLM): bare what is/are/do/does, what kind/type,
# is he/she, where is, could this, does this look, who is, …
#
# The plain spatial shape requires a determiner on both sides and the question
# to end right after the second noun phrase, so a trailing predicate
# ("Is the man in the picture happy?") is left for the classifier.
_FAST_PATH_VISUAL_RE = re.compile(
    r"""
    ^\s*(?:
        what\s+colou?rs?\b |
        what\s+(?:animals?|shape|sport|game|activity|room|scene|place|
                  foods?|fruits?|dish)\b |
        number\s+of\b |
        how\s+many\b |
        (?:is|are)\s+there\b |
        is\s+the\s+sky\b |
        (?:do|can|could|did|would)\s+you\s+see\b |
        what\s+is\s+
            (?:under|over|above|below|behind|beside|next\s+to|
               in\s+front\s+of)\b |
        what\s+(?:is|are)\s+
            (?:the|this|that|he|she|it|they|these|those|a|an)\b
            [\w'\s,-]*\b(?:doing|holding|wearing)\s*\??\s*$ |
        (?:is|are)\s+
            (?:the|a|an|this|that|these|those|his|her|its|their)\s+
            [\w'-]+(?:\s+[\w'-]+){0,3}\s+
            (?:on\s+top\s+of|in\s+front\s+of|next\s+to|
               on|in|under|underneath|above|below|behind|beside|
               between|near|inside|outside|beneath)\s+
            (?:the|a|an|this|that|these|those|his|her|its|their)\s+
            [\w'-]+\s*\??\s*$
    )
    """,
    re.I | re.X,
)


def is_non_visual_suspect(question: str) -> bool:
    """True when a question carries an OCR / opinion / knowledge / judgment marker.

    Used as a veto over :func:`is_fast_path_visual`; a suspect question can
    still be labeled DIRECTLY_VISUAL, but only by the LLM classifier.
    """
    q = (question or "").strip()
    if not q:
        return False
    return bool(_NON_VISUAL_SUSPECT_RE.search(_SUSPECT_EXEMPT_RE.sub(" ", q)))


def is_fast_path_visual(question: str) -> bool:
    """True when a question may be kept DIRECTLY_VISUAL without an LLM call.

    Requires an unambiguous visual whitelist shape and no non-visual suspect
    marker. Everything else is UNKNOWN and must reach the classifier.
    """
    q = (question or "").strip()
    if not q:
        return False
    if not _FAST_PATH_VISUAL_RE.search(q):
        return False
    return not is_non_visual_suspect(q)


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
    fast_path: bool = True,
) -> bool:
    """True when a checkpoint matches the current run configuration.

    ``fast_path`` is part of the identity: a checkpoint built with the Fast
    Path enabled cannot be reused for a ``--no-fast-path`` comparison run
    (and vice versa), because the two label different questions without an
    LLM call.
    """
    info = checkpoint.get("info") or {}
    if info.get("prompt_version") != CLASSIFIER_PROMPT_VERSION:
        return False
    if bool(info.get("fast_path_enabled", True)) != bool(fast_path):
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
    fast_path: bool = True,
) -> Tuple[List[Dict], List[Dict[str, Any]], Dict[str, int]]:
    """Keep DIRECTLY_VISUAL rows; collect NOT_DIRECTLY_VISUAL drops for a sidecar.

    Args:
        rows: caption rows (dicts with at least ``question``).
        classifier: Ollama classifier; when provided, only questions that
            pass :func:`is_fast_path_visual` skip the LLM call.
        offline_drop_candidates: when True and classifier is unavailable,
            drop all regex candidates (conservative offline mode).
        checkpoint_path: optional sidecar for incremental classifier resume.
        checkpoint_every: save checkpoint every N newly classified questions.
        resume: load and continue from ``checkpoint_path`` when valid.
        classifier_meta: model/host/prompt metadata for checkpoint validation.
        input_count: raw VQA input count before OCR/dedup (for validation).
        fast_path: when False, every question goes to the LLM classifier
            (``--no-fast-path``), for measuring Fast Path false positives.

    Returns:
        (kept_rows, dropped_rows, label_counts)

        Kept rows and ``dropped_rows`` both carry ``visual_filter_source``
        (``fast_path`` / ``llm_classifier``); ``dropped_rows`` entries also
        include ``label`` (and optional ``detail``) suitable for
        ``*_not_directly_visual.json``.
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
            fast_path=fast_path,
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
            fast_path=fast_path,
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
            "fast_path_enabled": bool(fast_path),
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
        n_fast = (
            sum(
                1 for row in rows
                if is_fast_path_visual(str(row.get("question") or ""))
            )
            if fast_path
            else 0
        )
        print(
            f"Question classifier: {n_total} questions "
            f"(binary DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL), "
            f"{n_fast} on the conservative Fast Path whitelist (no LLM), "
            f"{n_total - n_fast} to classify with the LLM"
            + ("" if fast_path else " (--no-fast-path)")
            + "...",
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

            # Conservative Fast Path: only unambiguous colour / count /
            # existence / plain-spatial shapes skip the LLM ruling.
            if fast_path and is_fast_path_visual(q):
                label_counts["FAST_PATH_VISUAL"] += 1
                label_counts["DIRECTLY_VISUAL"] += 1
                row["visual_filter_source"] = VISUAL_FILTER_FAST_PATH
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
                    _drop_record(
                        row,
                        "NOT_DIRECTLY_VISUAL",
                        detail or "parse_fail",
                        VISUAL_FILTER_LLM,
                    )
                )
            else:
                label_counts[label] = label_counts.get(label, 0) + 1
                if label == "DIRECTLY_VISUAL":
                    row["visual_filter_source"] = VISUAL_FILTER_LLM
                    kept.append(row)
                else:
                    dropped.append(
                        _drop_record(row, label, detail, VISUAL_FILTER_LLM)
                    )

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
    visual_filter_source: str = "",
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
    if visual_filter_source:
        out["visual_filter_source"] = visual_filter_source
    if detail:
        out["detail"] = detail
    return out
