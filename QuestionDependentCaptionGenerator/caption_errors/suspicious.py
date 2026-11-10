"""Code-only check: could this caption still be wrong? (no LLM)

Runs on the FINAL caption of every row, after the 7B judge and the
regeneration. It marks rows for a later pass with a bigger model. It never
changes a caption or a verdict.

A caption is suspicious when:

* ``extra_words``: it has a content word that is in neither the question nor
  the answer ("Clothes are hung up to dry." -> dry; "Both cows are standing."
  -> both). Catches added facts and swapped objects. Correct rewordings are
  marked too ("closed" for "not open"): the bigger model decides.
* ``polarity_flip``: the yes/no meaning looks flipped. Answer yes and the
  caption denies; answer no and the caption affirms with no negation and no
  new word; or a "not" that a non-yes/no answer never implied
  ("What color is the tie? / green" -> "The tie is not green.").

Only words added beyond the question + answer count; words left out do not.
"""

from __future__ import annotations

import re
from typing import List

from caption_rules import DIGIT_TO_WORD
from validation.checks import (
    has_no_polarity_mismatch, has_sentential_negation, has_spurious_negation,
    has_yes_denial, has_yes_polarity_mismatch,
)

SUSPICIOUS_VERSION = "s2"   # s2: added number words count as extra

# Function words and words that only restate "this is in the image".
_STOP = set("""
a an the this that these those there here it its they them their he she his her
him we you i me my our your of in on at to for by with from as and or but so
is are was were be been being am do does did doing done has have had having
can could will would shall should may might must
what which who whom whose how why when where
many much kind type sort some any one ones
see seen sees shown show shows visible picture pictured photo photograph image
scene shot appear appears appearing present
""".split())

# Words that only express the answer's polarity / quantity form.
_ANSWER_FORM = set("""
yes no not none nothing nobody zero all every each only
""".split())

_NUM_WORDS = set(DIGIT_TO_WORD.values())
_YES = {"yes", "yeah", "yep"}


def _stem(w: str) -> str:
    for suf in ("ing", "es", "s", "ed"):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower().replace("'s", ""))
            if w not in _STOP]


def _numbers(text: str) -> set:
    """Numbers in text, as words ("2" and "two" both -> "two").

    Compound numbers also add their parts: "22" -> "twenty-two", "twenty",
    "two", because a caption's "twenty-two" is read as two words.
    """
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w.isdigit():
            word = DIGIT_TO_WORD.get(w, w)
            out.add(word)
            out.update(word.split("-"))
        elif w in _NUM_WORDS:
            out.add(w)
    return out


def extra_words(question: str, answer: str, caption: str) -> List[str]:
    """Content words of the caption found in neither question nor answer.

    A number counts as extra too when neither the question nor the answer has
    it ("What's on the plate? / donuts" -> "Two donuts are on the plate.").
    "one" stays allowed: it is mostly the article ("one of", "a").
    """
    known = {_stem(w) for w in _words(f"{question} {answer}")}
    known |= {_stem(DIGIT_TO_WORD[w]) for w in _words(answer) if w in DIGIT_TO_WORD}
    known_nums = _numbers(f"{question} {answer}")
    out: List[str] = []
    for w in _words(caption):
        if w in _ANSWER_FORM or "n't" in w:
            continue
        if w.isdigit() or w in _NUM_WORDS:
            num = DIGIT_TO_WORD.get(w, w)
            if num not in known_nums and num != "zero" and w not in out:
                out.append(w)
            continue
        if _stem(w) not in known and w not in out:
            out.append(w)
    return out


def polarity_flip(question: str, answer: str, caption: str) -> bool:
    a = answer.strip().lower()
    if has_yes_polarity_mismatch(a, caption, question) or has_yes_denial(a, caption):
        return True
    if has_no_polarity_mismatch(a, caption, question):
        return True
    if a == "no" and not has_sentential_negation(caption) \
            and not extra_words(question, answer, caption):
        return True          # "Is the dog on the sofa? / no" -> "The dog is on the sofa."
    return a not in _YES and a != "no" and has_spurious_negation(a, caption, question)


def suspicious_reasons(question: str, answer: str, caption: str) -> List[str]:
    """[] when the caption looks fine, else reasons like ``extra_words:dry,sky``."""
    if not caption:
        return ["empty_caption"]
    out: List[str] = []
    extra = extra_words(question, answer, caption)
    if extra:
        out.append("extra_words:" + ",".join(extra))
    if polarity_flip(question, answer, caption):
        out.append("polarity_flip")
    return out
