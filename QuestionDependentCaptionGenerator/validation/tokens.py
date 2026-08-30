"""Tokenization, stemming, and content-word helpers for caption validation.

Shared by overlap computation and grounding checks in the fast validator.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Set, Tuple

from caption_rules import DIGIT_TO_WORD

_WORD_TO_DIGIT = {word: digit for digit, word in DIGIT_TO_WORD.items()}

_INFLECTION_SUFFIXES = ("ing", "edly", "ed", "es", "s")

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
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "without", "cannot", "one", "least", "also", "than", "enough",
    "show", "shows", "showed", "shown", "showing",
    "see", "sees", "seen", "seeing", "visible", "view", "viewed",
    "picture", "pictures", "pictured", "pic", "pics", "photo", "photos",
    "photograph", "photographed", "image", "images",
    "display", "displayed", "depict", "depicted", "appear", "appears",
}

_AUX_VERBS = {
    "is", "are", "was", "were", "be", "being", "been", "am",
    "do", "does", "did", "can", "could", "have", "has", "had",
    "will", "would", "should", "may", "might", "must",
}

_WH_CATEGORY_HEADS = {"what", "which", "whose"}

_NP_STOP = {
    "in", "on", "at", "for", "with", "to", "from", "out", "about", "near",
    "under", "over", "behind", "beside", "between", "inside", "outside",
    "above", "below", "next", "by",
}

_MAX_CATEGORY_STEMS = 3

_ALT_BOUNDARY = {
    "of", "in", "on", "at", "for", "with", "to", "from", "that", "and",
    "the", "a", "an",
}

_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "gold", "golden", "silver", "beige",
    "tan", "cream", "maroon", "navy", "teal", "cyan", "magenta", "violet",
    "blond", "blonde", "brunette",
}

_YES = {"yes", "yeah", "yep", "true", "maybe"}
_NO = {"no", "none", "0", "zero", "n/a", "not", "nothing"}


def numeric_equivalents(token: str) -> Set[str]:
    """A token plus its digit<->word number form (e.g. '2' <-> 'two')."""
    equivalents = {token}
    if token in DIGIT_TO_WORD:
        equivalents.add(DIGIT_TO_WORD[token])
    if token in _WORD_TO_DIGIT:
        equivalents.add(_WORD_TO_DIGIT[token])
    return equivalents


def strip_inflection(word: str) -> Tuple[str, bool]:
    """Strip one inflection suffix if >=3 letters remain; report whether it hit."""
    for suf in _INFLECTION_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)], True
    return word, False


def stem(word: str) -> str:
    """Light stem: strip inflection suffixes twice, then trailing silent 'e'."""
    word, changed = strip_inflection(word)
    if changed:
        word, _ = strip_inflection(word)
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def words(text: str) -> List[str]:
    """Lowercase word tokens with clitics folded away."""
    out: List[str] = []
    for w in re.findall(r"[a-z']+", text.lower()):
        w = w.replace("n't", "").split("'")[0]
        if w:
            out.append(w)
    return out


def is_content(word: str) -> bool:
    """True for tokens that carry visual content (not stopwords / too short)."""
    return word not in _STOPWORDS and len(word) >= 3


def content_words(text: str) -> Set[str]:
    """Content tokens (stemmed) after dropping stopwords / short tokens."""
    return {stem(w) for w in words(text) if is_content(w)}


def token_present(token: str, caption_lower: str) -> bool:
    """Match a token in the caption: exact word, numeric equivalent, or shared stem."""
    if any(
        re.search(rf"\b{re.escape(t)}\b", caption_lower)
        for t in numeric_equivalents(token)
    ):
        return True
    token_stem = stem(token)
    if len(token_stem) < 3:
        return False
    caption_word_list = re.findall(r"[a-z']+", caption_lower)
    return any(stem(w) == token_stem for w in caption_word_list)


def normalize_phrase(text: str) -> str:
    """Light normalize: lowercase, strip punctuation, collapse spaces."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def wh_category_stems(word_list: Sequence[str]) -> Set[str]:
    """Stems of the wh-category noun phrase that the answer replaces."""
    if not word_list or word_list[0] not in _WH_CATEGORY_HEADS:
        return set()
    out: Set[str] = set()
    for w in word_list[1:]:
        if w in _AUX_VERBS or w in _NP_STOP:
            break
        if is_content(w):
            out.add(stem(w))
            if len(out) >= _MAX_CATEGORY_STEMS:
                break
    return out


def alternative_stems(word_list: Sequence[str]) -> Set[str]:
    """Stems of an 'A or B' alternation, which the answer can only half-echo."""
    if "or" not in word_list:
        return set()
    i = word_list.index("or")
    out: Set[str] = set()
    for w in reversed(word_list[:i]):
        if is_content(w):
            out.add(stem(w))
            break
    for w in word_list[i + 1 :]:
        if w in _ALT_BOUNDARY or w in _AUX_VERBS:
            break
        if is_content(w):
            out.add(stem(w))
    return out


def required_question_stems(question: str) -> Set[str]:
    """Question stems a faithful caption must still contain."""
    word_list = words(question)
    stems = {stem(w) for w in word_list if is_content(w)}
    return stems - wh_category_stems(word_list) - alternative_stems(word_list)


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
    caps = sum(1 for t in a.split() if t[:1].isupper())
    if caps >= 2:
        return True
    return False
