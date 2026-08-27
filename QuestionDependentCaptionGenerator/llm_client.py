"""Ollama client baraye yek model: sequential ya concurrent API request.

Har fail reason-dar hast ta log file betune tozih bede chera ``fallback``
be ``llm_fallback`` tabdil nashod.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from caption_rules import DIGIT_TO_WORD
from llm_prompts import chat_messages

# Bumped whenever the Tier-1 accept/reject rules change, so a captions JSON
# records which validator produced it.
VALIDATOR_VERSION = "v3_high_precision_reject_plus_flags"

_WORD_TO_DIGIT = {word: digit for digit, word in DIGIT_TO_WORD.items()}


def _numeric_equivalents(token: str) -> Set[str]:
    """A token plus its digit<->word number form (e.g. '2' <-> 'two')."""
    equivalents = {token}
    if token in DIGIT_TO_WORD:
        equivalents.add(DIGIT_TO_WORD[token])
    if token in _WORD_TO_DIGIT:
        equivalents.add(_WORD_TO_DIGIT[token])
    return equivalents



# Suffixes stripped longest-first so a word matches only one bucket (a word
# can't end in both "ing" and "s"). Used for a light stem comparison so verb/
# noun inflections count as the same word (answer 'stands' <-> caption
# 'standing', answer 'dogs' <-> caption 'dog').
_INFLECTION_SUFFIXES = ("ing", "edly", "ed", "es", "s")


def _strip_inflection(word: str) -> Tuple[str, bool]:
    """Strip one inflection suffix if >=3 letters remain; report whether it hit."""
    for suf in _INFLECTION_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)], True
    return word, False


def _stem(word: str) -> str:
    """Strip inflection suffixes, but only while >=3 letters remain.

    The strip runs **twice** so a pluralized verbal noun collapses onto its
    singular: 'buildings' -> 'building' -> 'build' now matches 'building' ->
    'build'.  With a single pass the two forms stemmed differently and a
    caption saying 'buildings' was rejected as a relation mismatch against a
    question about a 'building'.

    A trailing silent 'e' is dropped last so singular/plural pairs collapse to
    the same stem: without it 'planes' -> 'plan' but 'plane' -> 'plane', and a
    word would fail to match itself ('picture' vs 'pictured' likewise).
    """
    word, changed = _strip_inflection(word)
    if changed:
        word, _ = _strip_inflection(word)
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _token_present(token: str, caption_lower: str) -> bool:
    """Match a token in the caption: exact word, numeric equivalent, or shared stem.

    - '2' matches 'two' (``_numeric_equivalents``).
    - 'stands' matches 'standing', 'dogs' matches 'dog' (shared stem, via a
      light suffix-stripping heuristic — not a real lemmatizer, but enough to
      stop false 'answer_mismatch' rejections on simple inflections).
    """
    if any(re.search(rf"\b{re.escape(t)}\b", caption_lower) for t in _numeric_equivalents(token)):
        return True
    token_stem = _stem(token)
    if len(token_stem) < 3:
        return False
    caption_words = re.findall(r"[a-z']+", caption_lower)
    return any(_stem(w) == token_stem for w in caption_words)


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
    # Drop a leading echoed "Q: ... Caption:" prefix if model pasted the prompt
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
    """Parse model response into ``expected`` caption strings.

    Accepts:
      - JSON array of strings (preferred for batches)
      - plain single sentence when ``expected == 1`` (common on single
        retries; small models often ignore the JSON-array instruction)

    Returns:
        ParseResult with captions on success, or reason/detail on failure.
    """
    text = _strip_fences(raw)
    start = text.find("[")
    end = text.rfind("]")

    # ---- Preferred: JSON array ----
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

    # ---- Fallback: plain caption text (single-item calls) ----
    # Your log: model returned "The animals are eating." without JSON —
    # that is a valid caption; accept it when expected==1.
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

# Yes/no answers: caption declarative bashe, lazem nist "yes" toye jomle bashe
_YES = {"yes", "yeah", "yep", "true", "maybe"}
_NO = {"no", "none", "0", "zero", "n/a", "not", "nothing"}

# Negation markers that flip the meaning of a sentence. If the gold answer is
# NOT itself a yes/no-style answer, a caption containing one of these is very
# likely a hallucinated meaning-flip (e.g. Q: 'Who made the cock?' A: 'rolex'
# -> LLM outputs 'No cock was made by Rolex.' — wrong, but 'rolex' still
# passes a naive substring check).
_NEGATION_RE = re.compile(
    r"(\b(?:no|not|never|none|nobody|nothing|neither|without|cannot|"
    r"no one|nowhere)\b|\w*n't\b)",
    re.I,
)

# Question embeds its own negation — a negative caption for answer=yes can be OK
# (e.g. "Is there a light that is not turned on?" + yes → "... is not turned on.").
_QUESTION_NEGATION_RE = re.compile(
    r"(\b(?:not|never|no|none|nobody|nothing|neither|without|cannot)\b"
    r"|\w*n't\b)",
    re.I,
)

# "no" inside a noun phrase names a thing, it does not negate the sentence:
# "The sign says no parking." is a positive statement about a sign. Treating
# it as negation used to reject perfectly good captions, so these spans are
# removed before any negation test.
_NON_SENTENTIAL_NO_RE = re.compile(
    r"""
    \bno\s+(?:parking|standing|stopping|smoking|entry|entrance|exit|
              trespassing|littering|swimming|diving|fishing|hunting|
              turn|turns|u-turn|uturn|left\s+turn|right\s+turn|
              dogs|pets|photos|photography|food|drinks|outlet|service|
              vacancy|passing|dumping|loitering|skateboarding|bikes|
              cell\s+phones|shirt|shoes)\b |
    \b(?:a|an|the|this|that|any|one)\s+no\s+\w+\s+
        (?:sign|signs|symbol|marking|markings|notice|placard)\b
    """,
    re.I | re.X,
)


def has_sentential_negation(caption: str) -> bool:
    """True when the caption negates its own statement.

    Determiner-style ``no`` inside a named phrase ("no parking sign") is not
    negation — see :data:`_NON_SENTENTIAL_NO_RE`.
    """
    return bool(_NEGATION_RE.search(_NON_SENTENTIAL_NO_RE.sub(" ", caption)))

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "this", "that",
    "these", "those", "there", "here", "it", "its", "do", "does", "did",
    "can", "could", "will", "would", "have", "has", "had", "you", "your",
    "what", "which", "who", "where", "when", "why", "how", "many", "much", "any",
    "some", "from", "by", "as", "if", "than", "then", "so", "too", "very",
    "just", "about", "into", "over", "after", "before", "between", "out",
    "up", "down", "off", "again", "further", "once", "all", "both", "each",
    "few", "more", "most", "other", "such", "only", "own", "same",
    "s", "t", "don", "now", "i", "me", "my", "we", "our",
    "he", "she", "they", "them", "his", "her", "their",
    # Negation / weak tokens — do not count as grounding overlap
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "without", "cannot", "one", "least", "also", "than", "enough",
    # Depiction scaffolding: these refer to the photo itself, not to anything
    # visible in it, and question/caption pick different ones freely
    # ("What sport is shown here?" → "... can be seen."). Counting them as
    # content punished correct captions for a synonym choice.
    "show", "shows", "showed", "shown", "showing",
    "see", "sees", "seen", "seeing", "visible", "view", "viewed",
    "picture", "pictures", "pictured", "pic", "pics", "photo", "photos",
    "photograph", "photographed", "image", "images",
    "display", "displayed", "depict", "depicted", "appear", "appears",
}

# Minimum share of required question stems that must survive into the caption.
RELATION_MIN_RATIO = 0.5

# A verb here ends the wh-category noun phrase ("What animal **is** this?").
_AUX_VERBS = {
    "is", "are", "was", "were", "be", "being", "been", "am",
    "do", "does", "did", "can", "could", "have", "has", "had",
    "will", "would", "should", "may", "might", "must",
}

_WH_CATEGORY_HEADS = {"what", "which", "whose"}

# The category phrase ends here too: 'of' keeps it going ("mode of transport"),
# any other preposition/particle starts a new phrase that must be preserved
# ("What's the odd color **out in** terms of shorts?").
_NP_STOP = {
    "in", "on", "at", "for", "with", "to", "from", "out", "about", "near",
    "under", "over", "behind", "beside", "between", "inside", "outside",
    "above", "below", "next", "by",
}

# Bound on how much of the question the category phrase may absorb.
_MAX_CATEGORY_STEMS = 3

# Tokens that close an "A or B" alternation span.
_ALT_BOUNDARY = {
    "of", "in", "on", "at", "for", "with", "to", "from", "that", "and",
    "the", "a", "an",
}

# Structural sanity check: brackets/labels and stray quotation marks mean the
# model echoed formatting instead of writing a plain sentence.
_BRACKET_CHARS = "[]{}"
_QUOTE_CHARS = "\"\u201c\u201d"

# The model should never write 'the answer is ...' / 'the answer' — it must
# weave the answer into a natural sentence about the image instead.
_ANSWER_PHRASE_RE = re.compile(r"\bthe answer\b", re.I)


def _words(text: str) -> List[str]:
    """Lowercase word tokens with clitics folded away.

    ``man's`` -> ``man`` and ``what's`` -> ``what`` so a possessive does not
    look like a different word than its bare form; ``isn't`` -> ``is`` so
    contractions land on the stopword list instead of contributing 'isn'.
    """
    out: List[str] = []
    for w in re.findall(r"[a-z']+", text.lower()):
        w = w.replace("n't", "").split("'")[0]
        if w:
            out.append(w)
    return out


def _content_words(text: str) -> Set[str]:
    """Content tokens (stemmed) after dropping stopwords / short tokens."""
    return {_stem(w) for w in _words(text) if _is_content(w)}


_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "gold", "golden", "silver", "beige",
    "tan", "cream", "maroon", "navy", "teal", "cyan", "magenta", "violet",
    "blond", "blonde", "brunette",
}


def _normalize_phrase(text: str) -> str:
    """Light normalize: lowercase, strip punctuation, collapse spaces."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def answer_requires_verbatim(answer: str) -> bool:
    """True for proper nouns, numbers, colors, or short non-yes/no answers."""
    a = answer.strip()
    if not a:
        return False
    low = a.lower()
    if low in _YES or low in _NO:
        return False
    tokens = [t for t in re.split(r"\W+", a) if t]
    if not tokens:
        return False
    if len(tokens) <= 3:
        return True
    if any(t.isdigit() or t in _WORD_TO_DIGIT or t in DIGIT_TO_WORD for t in tokens):
        return True
    if any(t.lower() in _COLOR_WORDS for t in tokens):
        return True
    # Capitalized multi-word name (Loon Mountain)
    caps = sum(1 for t in a.split() if t[:1].isupper())
    if caps >= 2:
        return True
    return False


def answer_verbatim_in_caption(answer: str, caption: str) -> bool:
    """Require answer phrase (light-normalized) to appear in the caption."""
    a_norm = _normalize_phrase(answer)
    c_norm = _normalize_phrase(caption)
    if not a_norm or not c_norm:
        return False
    if a_norm in c_norm:
        return True
    # Number digit <-> word equivalence for single-token numeric answers
    tokens = a_norm.split()
    if len(tokens) == 1:
        for eq in _numeric_equivalents(tokens[0]):
            if re.search(rf"\b{re.escape(eq)}\b", c_norm):
                return True
    # All non-stop tokens must be present (100%)
    content_tokens = [t for t in tokens if t not in _STOPWORDS]
    if not content_tokens:
        content_tokens = tokens
    return all(_token_present(t, c_norm) for t in content_tokens)


def _is_content(word: str) -> bool:
    """True for tokens that carry visual content (not stopwords / too short)."""
    return word not in _STOPWORDS and len(word) >= 3


def _wh_category_stems(words: Sequence[str]) -> Set[str]:
    """Stems of the category noun phrase that the answer replaces.

    'What **animal** is this?' + dog → 'This is a dog.';
    'What **season** is it?' + summer → 'It is summer outside.';
    'What **sport** is shown here?' + skateboarding → '... skateboarding ...';
    'What **mode of transportation** is pictured?' + car → 'A car is pictured.'
    The caption names the instance, so requiring the category word punishes a
    correct answer. Only the direct ``what/which/whose <NP>`` shape counts; a
    verb right after the wh-word means there is no category NP, so 'What is on
    the table?' keeps 'table' required and a chair/table swap is still caught.
    """
    if not words or words[0] not in _WH_CATEGORY_HEADS:
        return set()
    out: Set[str] = set()
    for w in words[1:]:
        if w in _AUX_VERBS or w in _NP_STOP:
            break
        if _is_content(w):
            out.add(_stem(w))
            if len(out) >= _MAX_CATEGORY_STEMS:
                break
    return out


def _alternative_stems(words: Sequence[str]) -> Set[str]:
    """Stems of an 'A or B' alternation, which the answer can only half-echo.

    'Is the sun to the right or left of this flower?' + left can never contain
    both alternatives, so requiring all question content is unsatisfiable.
    Picking the wrong branch is still caught by ``answer_in_caption``.
    """
    if "or" not in words:
        return set()
    i = words.index("or")
    out: Set[str] = set()
    for w in reversed(words[:i]):
        if _is_content(w):
            out.add(_stem(w))
            break
    for w in words[i + 1:]:
        if w in _ALT_BOUNDARY or w in _AUX_VERBS:
            break
        if _is_content(w):
            out.add(_stem(w))
    return out


def required_question_stems(question: str) -> Set[str]:
    """Question stems a faithful caption must still contain.

    Content stems minus the two groups the answer is expected to consume: the
    wh-category noun phrase and either/or alternatives.
    """
    words = _words(question)
    stems = {_stem(w) for w in words if _is_content(w)}
    return stems - _wh_category_stems(words) - _alternative_stems(words)


def question_relation_preserved(question: str, caption: str) -> Tuple[bool, float]:
    """Check that required question content words appear in the caption.

    Returns (ok, overlap_ratio). A flat ≥50% overlap is required. The previous
    rule demanded 100% for questions with ≤4 content stems, which rejected
    ~95% correct captions ('What are the animals doing?' → 'The animals are
    eating.' scored 0.50) because short questions are the norm in VQA. Cases in
    the 0.5-0.75 band are still escalated to the Tier-2 judge by
    ``is_semantically_suspicious``, so borderline items are ruled on by Qwen
    rather than accepted blindly.
    """
    q_words = required_question_stems(question)
    if not q_words:
        return True, 1.0
    c_words = _content_words(caption)
    if not c_words:
        return False, 0.0
    overlap = len(q_words & c_words)
    ratio = overlap / len(q_words)
    return ratio >= RELATION_MIN_RATIO, ratio


def has_unsupported_facts(question: str, answer: str, caption: str) -> bool:
    """True if caption introduces many content words absent from Q+A.

    Conservative: only reject when a clear majority of caption content is new
    (avoids false rejects on light paraphrases).
    """
    allowed = _content_words(f"{question} {answer}")
    cap = _content_words(caption)
    if not cap:
        return False
    extra = cap - allowed
    if not extra:
        return False
    # Hard fail when >2 novel stems and they dominate the caption
    if len(extra) >= 3 and len(extra) / len(cap) >= 0.4:
        return True
    if len(extra) >= 4:
        return True
    return False


def is_semantically_suspicious(
    question: str,
    answer: str,
    caption: str,
    *,
    relation_ratio: float,
) -> bool:
    """Borderline cases that should be escalated to the LLM PASS/FAIL judge."""
    a = answer.strip().lower()
    if a in _YES or a in _NO:
        if relation_ratio < 0.75:
            return True
    else:
        if relation_ratio < 0.65:
            return True
    allowed = _content_words(f"{question} {answer}")
    extra = _content_words(caption) - allowed
    if len(extra) >= 2:
        return True
    if answer_requires_verbatim(answer):
        return True
    return False


def caption_format_is_valid(caption: str) -> Tuple[bool, str]:
    """Structural check for one clean declarative sentence.

    Rejects captions that are:
      - empty, or fewer than 2 words
      - a question (contains '?')
      - wrapped/labeled with brackets or quotation marks
      - more than one sentence (an internal '.'/'!'/'?' before the final
        terminator, e.g. 'This is a home. Not a restaurant.')
      - littered with a stray double period ('..')
      - using the meta-phrase 'the answer'/'the answer is' instead of a
        natural sentence

    Returns:
        (ok, reason) — reason is 'ok' or a short machine-readable code.
    """
    c = caption.strip()
    if not c:
        return False, "empty_caption"
    if len(c.split()) < 2:
        return False, "too_short"
    if "?" in c:
        return False, "contains_question_mark"
    if any(ch in c for ch in _BRACKET_CHARS):
        return False, "contains_brackets"
    if any(ch in c for ch in _QUOTE_CHARS):
        return False, "contains_quotes"
    if ".." in c:
        return False, "double_period"
    if _ANSWER_PHRASE_RE.search(c):
        return False, "contains_answer_phrase"
    body = c[:-1] if c[-1] in ".!?" else c
    if re.search(r"[.!?]", body):
        return False, "multiple_sentences"
    return True, "ok"


def answer_in_caption(
    answer: str,
    caption: str,
    question: str = "",
) -> bool:
    """Check mikone javab toye caption hast; yes/no joda handle mishe.

    Strictness:
      - proper nouns / numbers / colors / short answers → verbatim (100%)
      - other answers → >=50% token match (digit/word + light stem)
      - yes/no → majority question-content overlap (see the relation flag in
        ``caption_soft_flags``); here we still require some overlap.
    """
    a = answer.strip().lower()
    c = caption.strip().lower()
    if not a or not c:
        return False
    if a in _YES or a in _NO:
        if len(c.split()) > 30:
            return False
        if not question.strip():
            return True
        ok, _ratio = question_relation_preserved(question, caption)
        return ok
    if answer_requires_verbatim(answer):
        return answer_verbatim_in_caption(answer, caption)
    if a in c:
        return True
    tokens = [t for t in re.split(r"\W+", a) if t]
    if not tokens:
        return False
    matched = sum(1 for t in tokens if _token_present(t, c))
    return matched / len(tokens) >= 0.5


def has_spurious_negation(answer: str, caption: str) -> bool:
    """True if caption negates a statement that a non-yes/no answer never implied.

    A negation word in the caption is fine when:
      - the gold answer is itself yes/no/none-style, or
      - the answer text already contains a negation word (e.g. 'not moving'),
        so the caption is just echoing it, not flipping the meaning, or
      - the ``no`` belongs to a noun phrase ("no parking sign").
    """
    a = answer.strip().lower()
    if not a or a in _YES or a in _NO:
        return False
    if not has_sentential_negation(caption):
        return False
    answer_words = set(re.split(r"\W+", a))
    if answer_words & {"no", "not", "never", "none", "nobody", "nothing", "neither", "without"}:
        return False
    return True


def has_yes_polarity_mismatch(answer: str, caption: str, question: str = "") -> bool:
    """True when answer=yes but the caption clearly negates (meaning flip).

    Skipped when the question itself embeds negation (e.g. 'not turned on'),
    where a negative surface form can still be correct for yes.
    """
    a = answer.strip().lower()
    if a not in _YES:
        return False
    if not has_sentential_negation(caption):
        return False
    if question and _QUESTION_NEGATION_RE.search(question):
        return False
    return True


# A caption that opens with "Yes"/"No" contradicts the opposite gold answer.
# This is the only polarity direction we still reject for answer=no: a
# paraphrase without any negation word is usually correct English ("Is it
# taken during the day?" + no -> "It is taken at night."), so it is flagged
# rather than dropped.
_CAPTION_AFFIRMS_RE = re.compile(r"^\s*yes\b|\bthe answer is yes\b", re.I)
_CAPTION_DENIES_RE = re.compile(r"^\s*no[,.\s]|\bthe answer is no\b", re.I)


def has_no_polarity_mismatch(answer: str, caption: str, question: str = "") -> bool:
    """True when answer=no but the caption explicitly affirms ("Yes, ...").

    Deliberately narrow. The previous rule demanded a negation word for every
    ``no`` answer, which rejected semantically correct paraphrases such as
    'It is taken at night.' Those now come back as a
    ``no_answer_without_negation`` flag from :func:`caption_soft_flags`.
    """
    if answer.strip().lower() != "no":
        return False
    del question  # question negation no longer changes this narrow check
    return bool(_CAPTION_AFFIRMS_RE.search(caption))


def has_yes_denial(answer: str, caption: str) -> bool:
    """True when answer is yes-like but the caption opens with a flat 'No'."""
    if answer.strip().lower() not in _YES:
        return False
    return bool(_CAPTION_DENIES_RE.search(caption))


def echoes_question(question: str, caption: str) -> bool:
    """True when the caption just repeats the question as a statement."""
    q = _normalize_phrase(question)
    c = _normalize_phrase(caption)
    if not q or not c:
        return False
    return c == q or (len(q.split()) >= 4 and c.startswith(q))


def is_batch_contamination(
    question: str,
    answer: str,
    caption: str,
    batch_pairs: Sequence[Tuple[str, str]],
    batch_captions: Sequence[Optional[str]],
    self_index: int,
) -> bool:
    """True if ``caption`` looks swapped from another item in the same batch.

    Triggers when:
      - another batch caption is near-identical, or
      - caption content overlaps another Q+A strictly better than this one
        while overlapping this Q+A poorly.
    """
    cap_words = _content_words(caption)
    if not cap_words:
        return False

    self_qa = _content_words(f"{question} {answer}")
    self_overlap = len(cap_words & self_qa)

    norm_cap = " ".join(caption.lower().split())
    for i, other_cap in enumerate(batch_captions):
        if i == self_index or not other_cap:
            continue
        other_norm = " ".join(other_cap.lower().split())
        if other_norm == norm_cap:
            return True
        # Near-duplicate: share most content words both ways
        other_words = _content_words(other_cap)
        if other_words and len(cap_words & other_words) / max(len(cap_words), len(other_words)) >= 0.85:
            return True

    for i, (oq, oa) in enumerate(batch_pairs):
        if i == self_index:
            continue
        other_qa = _content_words(f"{oq} {oa}")
        if not other_qa:
            continue
        other_overlap = len(cap_words & other_qa)
        # Strong match to another item, weak match to self → contamination
        if other_overlap >= 2 and other_overlap > self_overlap + 1:
            if self_overlap == 0 or other_overlap >= self_overlap * 2:
                return True
    return False


# Soft findings: recorded on the row as ``validation_flags`` and used to
# escalate to the Tier-2 judge, but never a regex-only reject. Comments8: a
# regex must not decide semantic equivalence — if a case is merely
# suspicious, flag it instead of deleting the sample.
FLAG_RELATION_LOW = "relation_low"
FLAG_UNSUPPORTED_FACTS = "unsupported_facts_suspect"
FLAG_NO_ANSWER_WITHOUT_NEGATION = "no_answer_without_negation"
FLAG_ANSWER_PARTIAL = "answer_partial_match"

VALIDATION_FLAGS = (
    FLAG_RELATION_LOW,
    FLAG_UNSUPPORTED_FACTS,
    FLAG_NO_ANSWER_WITHOUT_NEGATION,
    FLAG_ANSWER_PARTIAL,
)


@dataclass
class Validation:
    """Outcome of validating one caption."""

    ok: bool
    reason: str = "ok"
    flags: List[str] = field(default_factory=list)

    @property
    def needs_semantic_review(self) -> bool:
        """True when a Tier-2 LLM ruling should decide this caption."""
        return self.ok and self.reason == "needs_semantic_review"


def caption_soft_flags(question: str, answer: str, caption: str) -> List[str]:
    """Suspicious-but-not-wrong findings, safe to keep in the dataset."""
    flags: List[str] = []
    if question.strip():
        rel_ok, _ratio = question_relation_preserved(question, caption)
        if not rel_ok:
            flags.append(FLAG_RELATION_LOW)
    if has_unsupported_facts(question, answer, caption):
        flags.append(FLAG_UNSUPPORTED_FACTS)
    if answer.strip().lower() == "no" and not has_sentential_negation(caption):
        flags.append(FLAG_NO_ANSWER_WITHOUT_NEGATION)
    if not answer_requires_verbatim(answer) and not answer_in_caption(
        answer, caption, question
    ):
        flags.append(FLAG_ANSWER_PARTIAL)
    return flags


def caption_hard_reject_reason(
    answer: str,
    caption: str,
    question: str = "",
    *,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    self_index: int = -1,
) -> Optional[str]:
    """Reject code for errors we are nearly certain about, else None.

    High precision by design (Comments8): broken format, an echoed question,
    a flat polarity contradiction, a missing verbatim answer (number / proper
    noun / colour / short answer), and batch contamination. Semantic judgment
    is left to the Tier-2 LLM judge.
    """
    fmt_ok, fmt_reason = caption_format_is_valid(caption)
    if not fmt_ok:
        return fmt_reason
    if question.strip() and echoes_question(question, caption):
        return "echoes_question"
    if has_yes_polarity_mismatch(answer, caption, question):
        return "polarity_mismatch"
    if has_yes_denial(answer, caption):
        return "polarity_mismatch"
    if has_no_polarity_mismatch(answer, caption, question):
        return "polarity_mismatch"
    if has_spurious_negation(answer, caption):
        return "spurious_negation"
    if answer_requires_verbatim(answer) and not answer_verbatim_in_caption(
        answer, caption
    ):
        return "answer_mismatch"
    if (
        batch_pairs is not None
        and batch_captions is not None
        and self_index >= 0
        and is_batch_contamination(
            question, answer, caption, batch_pairs, batch_captions, self_index
        )
    ):
        return "batch_contamination"
    return None


def validate_caption(
    answer: str,
    caption: str,
    question: str = "",
    *,
    batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_captions: Optional[Sequence[Optional[str]]] = None,
    self_index: int = -1,
) -> Validation:
    """Tier-1 validation: hard rejects, soft flags, Tier-2 escalation.

    Returns a :class:`Validation` whose ``reason`` is a reject code when
    ``ok`` is False, ``'needs_semantic_review'`` when the LLM judge should
    rule, or ``'ok'``.
    """
    hard = caption_hard_reject_reason(
        answer,
        caption,
        question,
        batch_pairs=batch_pairs,
        batch_captions=batch_captions,
        self_index=self_index,
    )
    if hard is not None:
        return Validation(ok=False, reason=hard)

    flags = caption_soft_flags(question, answer, caption)
    _rel_ok, rel_ratio = (
        question_relation_preserved(question, caption)
        if question.strip()
        else (True, 1.0)
    )
    if flags or is_semantically_suspicious(
        question, answer, caption, relation_ratio=rel_ratio
    ):
        return Validation(ok=True, reason="needs_semantic_review", flags=flags)
    return Validation(ok=True, reason="ok", flags=flags)


def answer_mismatch_detail(answer: str, caption: str) -> str:
    """Human-readable why answer_in_caption failed."""
    a = answer.strip().lower()
    c = caption.strip().lower()
    tokens = [t for t in re.split(r"\W+", a) if t]
    missing = [t for t in tokens if not _token_present(t, c)]
    matched_pct = round(100 * (len(tokens) - len(missing)) / len(tokens)) if tokens else 0
    return (
        f"answer={answer!r} not reflected in caption={caption!r} "
        f"({matched_pct}% of tokens matched, need >=50%)"
        + (f"; missing_tokens={missing}" if missing else "")
    )


def spurious_negation_detail(answer: str, caption: str) -> str:
    """Human-readable why caption was rejected for a spurious negation."""
    hits = _NEGATION_RE.findall(caption)
    return (
        f"answer={answer!r} is not yes/no, but caption={caption!r} "
        f"contains negation word(s) {hits} — likely a meaning-flip hallucination"
    )


def polarity_mismatch_detail(answer: str, caption: str, question: str = "") -> str:
    """Human-readable why the yes/no polarity check failed."""
    if answer.strip().lower() == "no":
        return (
            f"answer={answer!r} is negative but caption={caption!r} "
            f"explicitly affirms it (Q={question!r})"
        )
    hits = _NEGATION_RE.findall(caption)
    return (
        f"answer={answer!r} is yes-like but caption={caption!r} "
        f"contradicts it (negation {hits}) (Q={question!r})"
    )


def echoes_question_detail(question: str, caption: str) -> str:
    """Human-readable why the caption was treated as an echo of the question."""
    return (
        f"caption={caption!r} just repeats the question {question!r} "
        "instead of stating the answer"
    )


def batch_contamination_detail(caption: str) -> str:
    """Human-readable why batch contamination was suspected."""
    return (
        f"caption={caption!r} looks swapped from another item in the same "
        "LLM batch (near-duplicate or better match to another Q+A)"
    )


def relation_mismatch_detail(question: str, caption: str) -> str:
    """Human-readable why subject/relation overlap looked low (flag only)."""
    ok, ratio = question_relation_preserved(question, caption)
    return (
        f"question content only partly preserved in caption={caption!r} "
        f"(Q={question!r}, overlap_ratio={ratio:.2f}, ok={ok})"
    )


def unsupported_facts_detail(question: str, answer: str, caption: str) -> str:
    """Human-readable why caption added unsupported facts."""
    allowed = _content_words(f"{question} {answer}")
    extra = sorted(_content_words(caption) - allowed)
    return (
        f"caption={caption!r} adds unsupported content words {extra} "
        f"not in Q+A"
    )


def semantic_fail_detail(question: str, answer: str, caption: str) -> str:
    """Human-readable why the LLM semantic judge returned FAIL."""
    return (
        f"semantic judge FAIL for Q={question!r} A={answer!r} "
        f"caption={caption!r}"
    )


def format_invalid_detail(reason: str, caption: str) -> str:
    """Human-readable why ``caption_format_is_valid`` rejected a caption."""
    return f"caption={caption!r} failed format check: {reason}"


_FORMAT_REASONS = {
    "empty_caption",
    "too_short",
    "contains_question_mark",
    "contains_brackets",
    "contains_quotes",
    "double_period",
    "contains_answer_phrase",
    "multiple_sentences",
}

# Reasons that actually drop a caption. ``relation_mismatch`` and
# ``unsupported_facts`` are gone from this set: they are flags now, so the
# accounting identity in the output JSON still adds up.
_VALIDATION_FAIL_REASONS = {
    "answer_mismatch",
    "echoes_question",
    "polarity_mismatch",
    "spurious_negation",
    "batch_contamination",
    "semantic_fail",
} | _FORMAT_REASONS


def flag_detail(flag: str, question: str, answer: str, caption: str) -> str:
    """Human-readable description of a soft validation flag."""
    if flag == FLAG_RELATION_LOW:
        return relation_mismatch_detail(question, caption)
    if flag == FLAG_UNSUPPORTED_FACTS:
        return unsupported_facts_detail(question, answer, caption)
    if flag == FLAG_NO_ANSWER_WITHOUT_NEGATION:
        return (
            f"answer={answer!r} but caption={caption!r} has no negation word — "
            "may still be a correct paraphrase, kept for review"
        )
    if flag == FLAG_ANSWER_PARTIAL:
        return answer_mismatch_detail(answer, caption)
    return f"{flag}: caption={caption!r}"


def rejection_detail(
    reason: str,
    answer: str,
    caption: str,
    question: str = "",
) -> str:
    """Dispatch to the right human-readable detail message for a reject reason."""
    if reason in _FORMAT_REASONS:
        return format_invalid_detail(reason, caption)
    if reason == "spurious_negation":
        return spurious_negation_detail(answer, caption)
    if reason == "polarity_mismatch":
        return polarity_mismatch_detail(answer, caption, question)
    if reason == "echoes_question":
        return echoes_question_detail(question, caption)
    if reason == "batch_contamination":
        return batch_contamination_detail(caption)
    if reason == "semantic_fail":
        return semantic_fail_detail(question, answer, caption)
    return answer_mismatch_detail(answer, caption)


@dataclass
class ChatResult:
    """One Ollama chat call outcome (batch or single)."""

    captions: Optional[List[str]] = None
    reason: str = "ok"
    detail: str = ""


@dataclass
class ItemOutcome:
    """Per Q+A outcome after batch + single retries.

    ``first_caption`` / ``first_reason`` remember the caption that was
    rejected before a retry, so the audit log can show what the retry
    actually changed.
    """

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
    """Client sade baraye Ollama chat API (yek model, 8GB-friendly)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mistral",
        num_ctx: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 300.0,
    ) -> None:
        """Host va model ro set mikone; options baraye VRAM kam.

        ``num_ctx`` default 4096 — prompt v3 + few-shot + batch fit beshe
        (1024 ghablan truncate mikard va hame caption ha fail mishodan).
        """
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _num_predict(self, batch_size: int) -> int:
        """Max token output: ~40 token per caption + buffer (JSON overhead)."""
        return max(128, batch_size * 40 + 64)

    def chat_captions(self, pairs: Sequence[Tuple[str, str]]) -> ChatResult:
        """Yek packed batch Q+A mifreste; ChatResult ba captions ya fail reason."""
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

    def semantic_judge(
        self,
        question: str,
        answer: str,
        caption: str,
    ) -> Tuple[bool, str]:
        """Tier-2 LLM judge: PASS only if caption matches Q+A with no extras.

        Returns:
            (pass, detail) — pass True iff model returns PASS.
        """
        system = (
            "You are a strict caption validator. Reply with only PASS or FAIL."
        )
        user = (
            "Given QUESTION, ANSWER and CAPTION, return PASS only if the "
            "caption correctly expresses the answer to the question and "
            "adds no unsupported factual information. Otherwise return FAIL.\n\n"
            f"QUESTION: {question}\n"
            f"ANSWER: {answer}\n"
            f"CAPTION: {caption}\n"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": min(self.num_ctx, 2048),
                "num_predict": 8,
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
        except Exception as exc:
            # Fail-closed on judge errors
            return False, f"semantic_judge_error:{exc}"

        content = ""
        msg = raw.get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        text = content.strip().upper()
        if text.startswith("PASS") or re.search(r"\bPASS\b", text):
            return True, "semantic_pass"
        return False, f"semantic_fail:{_preview(content, 120)}"

    def _accept_or_escalate(
        self,
        question: str,
        answer: str,
        caption: str,
        *,
        batch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        batch_captions: Optional[Sequence[Optional[str]]] = None,
        self_index: int = -1,
    ) -> Validation:
        """Run Tier-1 then, for suspicious items, the Tier-2 semantic judge."""
        result = validate_caption(
            answer,
            caption,
            question,
            batch_pairs=batch_pairs,
            batch_captions=batch_captions,
            self_index=self_index,
        )
        if not result.ok:
            return result
        if not result.needs_semantic_review:
            return Validation(ok=True, reason="ok", flags=result.flags)
        passed, _detail = self.semantic_judge(question, answer, caption)
        if passed:
            return Validation(ok=True, reason="ok", flags=result.flags)
        return Validation(ok=False, reason="semantic_fail", flags=result.flags)

    def captions_with_retry(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        single_retries: int = 1,
    ) -> List[ItemOutcome]:
        """Batch try, then per-item single retries with reasons.

        Validation policy (Comments7): Tier-1 lexical checks, escalate
        suspicious items to Tier-2 Qwen PASS/FAIL. On FAIL, regenerate once
        (``single_retries`` default 1); if still FAIL, leave as unresolved.

        Args:
            pairs: (question, answer) batch
            single_retries: extra single-item calls after batch miss
                (default 1 — one regenerate then drop)

        Returns:
            one ``ItemOutcome`` per input pair
        """
        pairs_list = list(pairs)
        n = len(pairs_list)
        out: List[ItemOutcome] = [
            ItemOutcome(reason="pending", detail="not attempted") for _ in range(n)
        ]

        batch = self.chat_captions(pairs_list)
        batch_caps: List[Optional[str]] = [None] * n
        if batch.captions is not None:
            tentative: List[Optional[str]] = [None] * n
            for i, cap in enumerate(batch.captions):
                q, a = pairs_list[i]
                result = self._accept_or_escalate(q, a, cap)
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
            for i, cap in enumerate(tentative):
                if cap is None:
                    continue
                q, a = pairs_list[i]
                result = self._accept_or_escalate(
                    q,
                    a,
                    cap,
                    batch_pairs=pairs_list,
                    batch_captions=tentative,
                    self_index=i,
                )
                if result.ok:
                    batch_caps[i] = cap
                    out[i] = ItemOutcome(
                        caption=cap,
                        reason="ok",
                        detail="accepted from batch",
                        attempts=["batch:ok"],
                        flags=result.flags,
                    )
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
                result = self._accept_or_escalate(q, a, cap)
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
    """Chand packed batch ro sequential ya ba ThreadPool mifreste.

    Args:
        client: OllamaClient (yek model)
        batches: list of Q+A batches
        workers: concurrent API request (1 = sequential, 8GB safe)
        on_batch_done: callback(batch_index, outcomes) bad az har batch
        on_batch_start: callback(batch_index, batch_len) ghabl az har call
        single_retries: forwarded to ``captions_with_retry`` (default 1)
    """
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
