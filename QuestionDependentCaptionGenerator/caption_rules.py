"""Rule-based engine baraye sakht-e question-dependent caption az VQA v2.

Har sample = (soal, javab) → yek jomle-ye caption mesl:
    "What color is the car?" + "red" → "The car is red."
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YES = {"yes", "yeah", "yep", "true", "maybe"}
NO = {"no", "none", "0", "zero", "n/a", "not", "nothing"}

DIGIT_TO_WORD: Dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
}

ARTICLES = {"a", "an", "the", "this", "that", "these", "those"}

# Bare personal pronouns — never prefix these with "The" (bug: "The he is ...").
PRONOUNS = {"he", "she", "it", "they", "we", "you", "i", "who"}

# Quantifier-led NPs ('one of the giraffes') are already a complete subject —
# don't prefix them with "The" either (bug: "The one is of the giraffes...").
_QUANTIFIER_LEAD = {
    "one", "some", "any", "each", "all", "both", "most", "many", "few",
    "several", "none", "everyone", "everybody", "anyone", "anybody",
    "someone", "somebody", "noone", "nobody",
}

# Mass / uncountable nouns commonly seen as VQA answers — never prefix these
# with "a/an" (bug: "The animal is eating a grass.").
_MASS_ANSWER_NOUNS = {
    "water", "grass", "sand", "snow", "rain", "ice", "bread", "cheese",
    "meat", "rice", "pasta", "soup", "coffee", "tea", "milk", "juice",
    "wine", "wood", "metal", "plastic", "paper", "glass", "dirt", "mud",
    "smoke", "fog", "air", "wind", "music", "art", "equipment",
    "furniture", "luggage", "traffic", "hair", "fur", "wool", "cotton",
    "silk", "leather", "gravel", "hay", "straw", "concrete", "gold",
    "silver", "steel", "cement", "salt", "sugar", "flour", "cereal",
    "salad", "corn", "broccoli", "lettuce", "spinach", "asparagus",
    "popcorn", "spaghetti", "butter", "honey", "jam", "cream", "yogurt",
    "clothing", "makeup", "trash", "garbage", "ketchup", "mustard",
}

RuleFn = Callable[[str, str], Optional[str]]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def mode_answer(answers: List[str]) -> str:
    """Az 10 javab annotator, mode (por-tekrar-tarin) ro bargardoon."""
    return Counter(a.strip().lower() for a in answers).most_common(1)[0][0]


def answer_mode_stats(answers: List[str]) -> Tuple[str, int, float]:
    """Mode answer + annotator agreement stats.

    Returns:
        (mode_answer, answer_count, answer_consensus) — ``answer_count`` is
        how many of the annotators gave the mode answer (normalized,
        case-insensitive), ``answer_consensus`` is that count divided by
        the total number of annotators (rounded to 4 decimals), e.g.
        8 agreeing out of 10 -> (mode, 8, 0.8). This is annotator
        agreement, not a model confidence score.
    """
    normalized = [a.strip().lower() for a in answers]
    if not normalized:
        return "", 0, 0.0
    mode_ans, count = Counter(normalized).most_common(1)[0]
    consensus = round(count / len(normalized), 4)
    return mode_ans, count, consensus


# ---------------------------------------------------------------------------
# OCR-dependent question detection
# ---------------------------------------------------------------------------
#
# Some VQA v2 questions can only be answered by reading rendered text/digits
# in the image — a sign, a logo, a jersey number, a license plate, a clock
# face, a scoreboard ("What does the sign say?" -> "3M"). The downstream
# SimpleImageCaptioner (Stage 1) is a Faster R-CNN region-feature + LSTM
# captioner with no OCR/text-recognition component: it only ever sees pooled
# visual region features, so it cannot genuinely read glyphs. Training it on
# these targets ("The sign says 3M.") gives it an unlearnable label — the
# model can only memorize incidental visual correlations, not the actual
# text — so we flag these pairs to exclude them instead of quietly poisoning
# the training signal.
#
# This is a heuristic, NOT ground truth: VQA v2 has no explicit "requires
# OCR" annotation. It combines two signals:
#   1. A regex over the question text for phrasing that specifically asks
#      what is written/printed/displayed (sign says, brand, logo, license
#      plate, jersey/bus number, clock/watch time, ...).
#   2. The official VQA ``question_type`` prefix (from the annotations file,
#      NOT the questions file), when the caller has it — a few prefixes are
#      OCR-heavy enough to flag on their own even without a text-regex hit.
#
# Deliberately conservative: ambiguous prefixes like "what is the name"
# (could be "what is the name of this fruit" — not OCR — or "what is the
# name on the jersey" — OCR) are left OUT unless phrased as "name on ..." /
# "name of the street"; the bare form is a suspect for the question
# classifier instead, which can judge it per question.
# Prefer intent phrases (letter/website/initials/street name/printed) over
# bare nouns like ``sign`` so "What color is the sign?" stays visual.
#
# Comments8 additions: rendered numbers on objects/clothing ("number on the
# shirt", "train number", "shirt number"), street names, and text
# written/printed/engraved on a surface, since a Faster R-CNN captioner
# cannot read any of them.

_OCR_QUESTION_RE = re.compile(
    r"""
    \bwhat\s+(does|do|did)\s+.{0,40}?\bsays?\b |  # "what does the sign say"
    \bwhat\s+.{0,30}?\bsign\s+says?\b |           # "what the sign says"
    \b(?:is|are|was)\s+written\b |                # "what is written on..."
    \bwhat\s+is\s+printed\b |                     # "what is printed on..."
    \b(?:written|printed|inscribed|engraved|embossed|stamped|typed)\s+
        (?:on|in|across|above|below|under|at|near)\b |
    \bwhat\s+words?\b |                           # "what word(s) are on..."
    \bwhat\s+(are\s+the\s+)?letters?\b |          # "what letter(s)..."
    \bwhat\s+is\s+(the\s+)?(last\s+)?letter\b |   # "what is the last letter"
    \bthat\s+letter\s+is\b |                      # "That letter is large on..."
    \b(?:letters|initials)\s+(?:on|in|are\s+on)\b |
    \bwhat\s+are\s+the\s+initials\b |
    \bwhat\s+(is\s+)?(the\s+)?website\b |
    \bwhat\s+website\b |
    \bwhat\s+(are\s+)?(the\s+)?(two\s+)?street\s+names?\b |
    \bwhat\s+is\s+(the\s+)?street\s+name\b |
    \bname\s+of\s+(?:the\s+)?street\b |           # "name of the street"
    \b(?:street|road|avenue|boulevard)\s+(?:name|sign)\b |
    \blicense\s+(number|plate)\b |                # plate / registration number
    \bwhat\s+brand\b | \bwhat\s+is\s+the\s+brand\b |
    \bwhat\s+logo\b | \bwhat\s+team'?s?\s+logo\b |
    \bwhat\s+(is\s+the\s+)?name\s+on\b |          # name on cake/jersey
    \bwhich\s+company\s+is\s+on\b |               # company on plane
    \bwhat\s+hundred\s+block\b |                  # street number text
    \bwhat\s+number\s+is\s+(on|the|this|that)\b | # jersey / bus / plate number
    \bwhat\s+is\s+the\s+number\s+on\b |           # "what is the number on..."
    \bnumbers?\s+on\s+(?:the|this|that|his|her|their|a|an)\b |
    \bwhat\s+(?:is\s+)?(?:the\s+)?
        (?:shirt|jersey|uniform|player|bus|train|plane|flight|truck|taxi|
           room|gate|platform|track|route|channel|phone|model|
           serial|apartment|house|address)\s+number\b |
    \b(?:shirt|jersey|uniform|player|bus|train|plane|flight|truck|taxi|
        room|gate|platform|track|route|channel|phone|model|
        serial|apartment|house)\s+number\b |
    \bwhat\s+time\s+(is\s+it|does)\b              # clock / watch reading
    """,
    re.I | re.X,
)

# ``question_type`` prefixes (VQA v2's official fixed taxonomy) that are OCR-
# heavy enough to flag on their own, even when the regex above doesn't match
# the exact phrasing of a given question.
_OCR_QUESTION_TYPES = {
    "what does the",
    "what brand",
    "what number is",
    "what time",
}


def is_ocr_question(question: str, question_type: str = "") -> bool:
    """True if answering ``question`` requires reading rendered text/digits.

    Heuristic only — see the module comment above ``_OCR_QUESTION_RE`` for
    the rationale (SimpleImageCaptioner has no OCR capability) and for what
    counts as OCR here (signs/logos/brands/plates/jersey numbers/clock
    reading), plus why some ambiguous phrasing is deliberately excluded.

    Args:
        question: raw question text.
        question_type: VQA v2 ``annotations[i]["question_type"]`` if the
            caller has it; pass ``""`` (default) to fall back to a
            regex-only check on the question text.

    Returns:
        True if the pair should be treated as OCR-dependent and excluded.
    """
    if _OCR_QUESTION_RE.search(question):
        return True
    return (question_type or "").strip().lower() in _OCR_QUESTION_TYPES


def normalize_answer(answer: str) -> str:
    """Lowercase the answer; spell out small integers (0–12) as words."""
    a = answer.strip().lower()
    if a in DIGIT_TO_WORD:
        return DIGIT_TO_WORD[a]
    # "2 giraffes" → "two giraffes"
    m = re.match(r"^(\d+)\b(.*)$", a)
    if m and m.group(1) in DIGIT_TO_WORD:
        tail = m.group(2)
        return f"{DIGIT_TO_WORD[m.group(1)]}{tail}"
    return a


def capitalize_first(text: str) -> str:
    """Harf aval jomle ro bozorg kon."""
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def strip_question_mark(question: str) -> str:
    """Alamat soal (?) ro az akhar soal bardar."""
    return question.strip().rstrip("?")


def is_yes(answer: str) -> bool:
    """Check kon javab az no'e positive/yes hast ya na."""
    return answer.strip().lower() in YES


def is_no(answer: str) -> bool:
    """Check kon javab az no'e negative/no hast ya na."""
    return answer.strip().lower() in NO


def with_article(noun_phrase: str) -> str:
    """Age noun phrase article nadare, a/an ezafe kon."""
    np = noun_phrase.strip()
    if not np:
        return np
    first = np.split()[0]
    if first in ARTICLES:
        return np
    article = "an" if first[0] in "aeiou" else "a"
    return f"{article} {np}"


def smart_article(answer: str) -> str:
    """``with_article`` — but skip the article when it would be wrong.

    Bare (no article) for:
      - answers that already start with an article/determiner
      - mass/uncountable nouns ('grass', 'water', ...)
      - multi-item answers ('knife and spoon', 'cat, dog')
      - plural-looking answers ('donuts', 'leaves')
    """
    a = answer.strip()
    if not a:
        return a
    low = a.lower()
    tokens = low.split()
    if not tokens:
        return a
    if tokens[0] in ARTICLES:
        return a
    if " and " in low or "," in low:
        return a
    last = tokens[-1]
    if last in _MASS_ANSWER_NOUNS:
        return a
    if len(last) > 1 and last.endswith("s") and not last.endswith("ss"):
        return a
    return with_article(a)


def prefix_the(subject: str) -> str:
    """Add a 'The ' prefix to a bare noun-phrase subject — but leave
    demonstratives, existing articles, personal pronouns, and
    quantifier-led NPs ('one of the giraffes') untouched.

    Fixes bugs like:
        'he' -> 'The he'          (should stay 'He')
        'one of the giraffes' -> 'The one is of the giraffes' (should stay
            'One of the giraffes')
    """
    subj = subject.strip()
    if not subj:
        return subj
    low = subj.lower()
    if low.startswith(("this ", "that ", "these ", "those ")) or low in {
        "this", "that", "these", "those",
    }:
        return capitalize_first(subj)
    if low.startswith("the "):
        return f"The {subj[4:]}"
    words = low.split()
    first_word = words[0]
    if first_word in PRONOUNS:
        return capitalize_first(subj)
    if first_word in _QUANTIFIER_LEAD:
        return capitalize_first(subj)
    return f"The {subj}"


_AUX_RE = (
    r"is|are|was|were|has|have|had|can|could|will|would|do|does|did"
)
_SUBJ_LEAD_RE = (
    r"There|This|That|These|Those|The|He|She|It|They|We|You|I|"
    r"One|Some|Any|Each|All|Both|Most|Many|Few|Several|None|Someone|No one|"
    r"At least one|Not all|Not both"
)


def insert_not(sentence: str) -> str:
    """Natural negation — never the 'It is not true that ...' template.

    Examples:
        'This is a horse.' → 'This is not a horse.'
        'This plane is landing.' → 'This plane is not landing.'
        'They are playing polo.' → 'They are not playing polo.'
        'This photo shows train tracks.' → 'This photo does not show train tracks.'
    """
    s = sentence.rstrip(".")

    # Pronoun / demonstrative / existential + AUX + REST (no intervening NP)
    m0 = re.match(
        rf"^({_SUBJ_LEAD_RE})\s+({_AUX_RE})\s+(.+)$",
        s,
        re.I,
    )
    if m0:
        return f"{m0.group(1)} {m0.group(2)} not {m0.group(3)}."

    # Full NP subject + AUX + REST
    m = re.match(
        rf"^({_SUBJ_LEAD_RE})\s+(.+?)\s+({_AUX_RE})\s+(.+)$",
        s,
        re.I,
    )
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} not {m.group(4)}."

    # Lexical verb (from Does/Do rewrite): "This photo shows X." → "... does not show X."
    m2 = re.match(
        r"^(There|This|That|These|Those|The)\s+(.+?)\s+(\w+)s\s+(.+)$",
        s,
        re.I,
    )
    if m2 and m2.group(3).lower() not in {
        "is", "are", "was", "were", "has", "have", "had",
    }:
        base = m2.group(3)
        if base.lower().endswith("ies"):
            verb_base = base[:-3] + "y"
        elif base.lower().endswith("es") and base.lower()[:-2].endswith(
            ("s", "x", "z", "ch", "sh")
        ):
            verb_base = base[:-2]
        elif base.lower().endswith("s"):
            verb_base = base[:-1]
        else:
            verb_base = base
        return (
            f"{m2.group(1)} {m2.group(2)} does not "
            f"{verb_base.lower()} {m2.group(4)}."
        )

    # Last-resort: insert "not" after the first auxiliary anywhere in the sentence.
    m3 = re.search(rf"\b({_AUX_RE})\b", s, re.I)
    if m3:
        return f"{s[: m3.end()]} not{s[m3.end():]}."

    # Bare lexical clause with no aux — wrap with do-support on the first verb-ish token.
    return f"It is not {s[0].lower() + s[1:]}." if s else s

def format_the_subject(subject: str, predicate: str, be: str) -> str:
    """Jomle besaz ba 'SUBJECT is/are PREDICATE' (subject article-aware)."""
    return f"{prefix_the(subject)} {be} {predicate}."


# Trailing present-participle properties on color questions
_COLOR_PARTICIPLES = {
    "wearing",
    "holding",
    "carrying",
    "using",
    "riding",
    "driving",
    "sitting",
    "standing",
    "lying",
    "eating",
    "drinking",
    "playing",
    "writing",
    "reading",
    "covering",
    "painting",
}

_MEDIA_NOUNS = {"picture", "photo", "image", "photograph", "scene", "shot"}

_IRREGULAR_3SG = {
    "have": "has",
    "do": "does",
    "go": "goes",
    "be": "is",
}
_IRREGULAR_PAST = {
    "have": "had",
    "do": "did",
    "go": "went",
    "be": "was",
    "see": "saw",
    "make": "made",
    "take": "took",
    "come": "came",
    "show": "showed",
    "get": "got",
    "give": "gave",
    "find": "found",
    "know": "knew",
    "think": "thought",
    "say": "said",
    "wear": "wore",
    "hold": "held",
    "fall": "fell",
    "run": "ran",
    "eat": "ate",
    "drink": "drank",
    "sit": "sat",
    "stand": "stood",
    "write": "wrote",
    "read": "read",
    "ride": "rode",
    "drive": "drove",
    "fly": "flew",
    "swim": "swam",
}

# Irregular plural -> singular for the head noun of a counted NP.
_IRREGULAR_SINGULAR: Dict[str, str] = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "geese": "goose",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "knives": "knife",
    "leaves": "leaf",
    "wolves": "wolf",
    "shelves": "shelf",
    "loaves": "loaf",
    "lives": "life",
    "wives": "wife",
    "buses": "bus",
    "busses": "bus",
    "dishes": "dish",
    "boxes": "box",
    "watches": "watch",
    "benches": "bench",
    "bushes": "bush",
    "churches": "church",
}


def conjugate_3sg(verb: str) -> str:
    """Base verb → 3rd-person singular (show → shows)."""
    v = verb.strip().lower()
    if not v:
        return v
    if v in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[v]
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return v + "es"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def conjugate_past(verb: str) -> str:
    """Base verb → simple past (show → showed)."""
    v = verb.strip().lower()
    if not v:
        return v
    if v in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[v]
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def singularize_word(word: str) -> str:
    """Best-effort plural → singular for the head noun of a counted NP."""
    w = word.strip()
    low = w.lower()
    if low in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[low]
    if low.endswith("ies") and len(low) > 3:
        return low[:-3] + "y"
    if low.endswith(("ches", "shes", "sses", "xes", "zes")):
        return low[:-2]
    if low.endswith("s") and not low.endswith("ss") and len(low) > 1:
        return low[:-1]
    return low


_IRREGULAR_PLURAL: Dict[str, str] = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "goose": "geese",
    "mouse": "mice",
    "foot": "feet",
    "tooth": "teeth",
    "knife": "knives",
    "leaf": "leaves",
    "wolf": "wolves",
    "shelf": "shelves",
    "loaf": "loaves",
    "life": "lives",
    "wife": "wives",
    "bus": "buses",
    "dish": "dishes",
    "box": "boxes",
    "watch": "watches",
    "bench": "benches",
    "bush": "bushes",
    "church": "churches",
}


def _looks_plural_word(word: str) -> bool:
    """True when a single token already looks plural."""
    low = word.strip().lower()
    if not low:
        return False
    if low in _IRREGULAR_SINGULAR or low in {
        "people", "children", "men", "women", "mice", "geese", "feet", "teeth",
    }:
        return True
    if low.endswith(("ches", "shes", "sses", "xes", "zes", "ies")):
        return True
    if low.endswith("s") and not low.endswith("ss") and len(low) > 1:
        return True
    return False


def pluralize_word(word: str) -> str:
    """Best-effort singular → plural for the head noun of a counted NP."""
    low = word.strip().lower()
    if not low:
        return low
    if low in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[low]
    if _looks_plural_word(low):
        return low
    if low.endswith(("s", "x", "z", "ch", "sh")):
        return low + "es"
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return low[:-1] + "ies"
    if low.endswith("fe"):
        return low[:-2] + "ves"
    if low.endswith("f") and not low.endswith("ff"):
        return low[:-1] + "ves"
    return low + "s"


def singularize_noun_phrase(phrase: str) -> str:
    """Singularize only the head noun of a counted NP.

    Head-initial 'NOUN of NOUN' phrases singularize the first word:
        'bodies of water' -> 'body of water'; 'kinds of animals' -> 'kind of animals'
    Otherwise (plain noun, or ADJ(s) + noun) the head is the last word:
        'windows' -> 'window'; 'square lights' -> 'square light'
    """
    tokens = phrase.strip().split()
    if not tokens:
        return phrase
    if len(tokens) >= 3 and tokens[1].lower() == "of":
        tokens[0] = singularize_word(tokens[0])
    else:
        tokens[-1] = singularize_word(tokens[-1])
    return " ".join(tokens)


def pluralize_noun_phrase(phrase: str) -> str:
    """Pluralize only the head noun of a counted NP.

    'light post' -> 'light posts'; 'cookie' -> 'cookies'; 'people' -> 'people'
    Head-initial 'NOUN of NOUN': 'kind of animal' -> 'kinds of animal'
    """
    tokens = phrase.strip().split()
    if not tokens:
        return phrase
    if len(tokens) >= 3 and tokens[1].lower() == "of":
        tokens[0] = pluralize_word(tokens[0])
    else:
        tokens[-1] = pluralize_word(tokens[-1])
    return " ".join(tokens)


# Quantifier-led NP: 'one of the giraffes', 'some of these dogs', ...
_QUANT_OF_RE = re.compile(
    r"^((?:one|some|any|each|none|all|part|most|both|few|many|several)\s+of\s+"
    r"(?:the|these|those|them)\s+\S+)\s+(.+)$",
    re.I,
)


# ---------------------------------------------------------------------------
# Rule functions — har rule yek pattern soal ro handle mikone
# ---------------------------------------------------------------------------


def rule_what_color(question: str, answer: str) -> Optional[str]:
    """Pattern: color questions → subject + (optional participle) + color.

    Examples:
        'What color are the dishes?' + 'pink and yellow'
            → 'The dishes are pink and yellow.'
        'What color is the person on the elephant in the back wearing?' + 'red'
            → 'The person on the elephant in the back is wearing red.'
    """
    m = re.match(r"^what color(?:s)? (?:is|are) (?:the )?(.+)$", question, re.I)
    if not m:
        return None
    obj = m.group(1).strip()
    verb = "are" if re.search(r"\bare\b", question, re.I) else "is"

    tokens = obj.split()
    if tokens and tokens[-1].lower() in _COLOR_PARTICIPLES:
        participle = tokens[-1].lower()
        subject = " ".join(tokens[:-1]).strip()
        if not subject:
            return None
        # Keep leading article handling via format_the_subject
        return format_the_subject(subject, f"{participle} {answer}", verb)

    return format_the_subject(obj, answer, verb)


# Only two "How many ...?" shapes are safe to handle with a rule (anything
# else — "...can you see eating?", "...are standing?", "...can be seen?" —
# is too free-form to rewrite reliably and defers to the SLM instead):
#   "How many <noun> are/is there?"
#   "How many <noun> are/is in/on ...?"    (the location is dropped, not kept)
_HOW_MANY_ARE_THERE_RE = re.compile(r"^(.+?)\s+(?:are|is)\s+there$", re.I)
_HOW_MANY_ARE_IN_ON_RE = re.compile(r"^(.+?)\s+(?:are|is)\s+(?:in|on)\s+.+$", re.I)


def _singularize_full_np(phrase: str) -> str:
    """Singularize a full noun phrase, including a trailing 'of X' complement.

    Used only for a count of one, where the whole phrase denotes a single
    item/category: 'kinds of animals' -> 'kind of animal' (not 'kind of
    animals') — both the head noun and the 'of' complement become singular.
    """
    if " of " in phrase:
        head, _, tail = phrase.partition(" of ")
        head = singularize_noun_phrase(head)
        tail_tokens = tail.split()
        if tail_tokens:
            tail_tokens[-1] = singularize_word(tail_tokens[-1])
        return f"{head} of {' '.join(tail_tokens)}".strip()
    return singularize_noun_phrase(phrase)


def _pluralize_full_np(phrase: str) -> str:
    """Pluralize a counted NP for count != 1.

    'light post' -> 'light posts'; 'kinds of animals' stays plural on the
    head ('kinds') without forcing the complement.
    """
    if " of " in phrase:
        head, _, tail = phrase.partition(" of ")
        return f"{pluralize_noun_phrase(head)} of {tail}".strip()
    return pluralize_noun_phrase(phrase)


def rule_how_many(question: str, answer: str) -> Optional[str]:
    """Pattern: 'How many X are/is there|in/on ...?' → 'There is/are {answer} X.'

    Examples:
        'How many cookies are there?' + '2' → 'There are two cookies.'
        'How many light post is there?' + '4' → 'There are four light posts.'
        'How many windows are on the caboose?' + '1' → 'There is one window.'
        'How many kinds of animals are in this photo?' + '1'
            → 'There is one kind of animal.'

    Any other shape (a verb-ing predicate, 'can be seen', 'can you see',
    ...) returns None and defers to the SLM.
    """
    m = re.match(r"^how many (.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(1).strip()

    noun_m = _HOW_MANY_ARE_THERE_RE.match(rest) or _HOW_MANY_ARE_IN_ON_RE.match(rest)
    if not noun_m:
        return None
    noun = noun_m.group(1).strip()
    if not noun:
        return None

    ans = normalize_answer(answer)
    if is_no(answer) or ans in {"zero", "none"}:
        return f"There are no {_pluralize_full_np(noun)}."
    if ans in {"one", "1"}:
        return f"There is one {_singularize_full_np(noun)}."
    return f"There are {ans} {_pluralize_full_np(noun)}."


def _naturalize_doing_answer(answer: str) -> str:
    """Add a missing article on a bare object after a V-ing verb.

    'cutting tie' → 'cutting a tie'
    'eating grass' → 'eating grass'   (mass noun)
    'holding the dog' → unchanged
    'running' → unchanged
    """
    tokens = answer.strip().split()
    if len(tokens) < 2:
        return answer.strip()
    if not tokens[0].lower().endswith("ing"):
        return answer.strip()
    if tokens[1].lower() in ARTICLES | PRONOUNS | _QUANTIFIER_LEAD:
        return answer.strip()
    obj = " ".join(tokens[1:])
    obj_head = tokens[-1].lower()
    if obj_head in _MASS_ANSWER_NOUNS:
        return answer.strip()
    if _looks_plural_word(obj_head) and len(tokens) == 2:
        return answer.strip()
    return f"{tokens[0]} {smart_article(obj)}"


def rule_what_is_doing(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What is the X doing?' → 'The X is {naturalized answer}.'

    Example:
        'What is the woman doing?' + 'cutting tie'
            → 'The woman is cutting a tie.'
    """
    m = re.match(r"^what is (?:the )?(.+?) doing$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    if not subj or not answer.strip():
        return None
    pred = _naturalize_doing_answer(answer)
    return f"{prefix_the(subj)} is {pred}."


# ---------------------------------------------------------------------------
# Routing helpers — decide rule vs LLM without scattering checks in every rule.
#
# Why some categories always/often go to LLM:
#   - Does/Do/Did: auxiliary inversion + missing copulas ("look like it
#     chocolate") are too fragile for deterministic rewrite.
#   - All Is/Are/Was/Were: subject/predicate splitting is too fragile for
#     a deterministic rewrite (locatives, quantifiers, existentials, ...).
#   - What kind/type of ...: compound NP composition needs a real parser.
#   - Who (non is/are, or uncertain answer NP): "Who made X?" needs a
#     lexical-verb rewrite the rule cannot do safely.
# ---------------------------------------------------------------------------


def should_use_llm_for_does_do(question: str, answer: str = "") -> bool:
    """True for Does/Do/Did questions — always routed to LLM.

    ``rule_yesno_does_do`` is kept in the codebase for reference / future
    narrowing, but routing never applies it: auxiliary transforms are a
    frequent source of broken captions
    ("This cake looks like it chocolate.").
    """
    del answer  # answer unused; signature matches the other helpers
    q = strip_question_mark(question).lower()
    return bool(re.match(r"^(does|do|did)\s+", q))


def should_use_llm_for_what_kind_type(question: str, answer: str = "") -> bool:
    """True for 'What kind/type of ...' questions — always routed to LLM."""
    del answer
    q = strip_question_mark(question).lower()
    return bool(re.match(r"^what (?:kind|type) of ", q))


def should_use_llm_for_is_are(question: str, answer: str = "") -> bool:
    """True for all Is/Are/Was/Were questions — always routed to LLM."""
    del answer
    q = strip_question_mark(question).lower()
    return bool(re.match(r"^(is|are|was|were)\s+", q))


def should_use_llm_for_who(question: str, answer: str = "") -> bool:
    """True when a Who-question cannot be rewritten safely by ``rule_who``.

    - 'Who made the clock?' (lexical verb) → LLM
    - 'Who is in the photo?' + short answer → try rule
    - Very long / multi-phrase answers → LLM
    """
    q = strip_question_mark(question).lower()
    if not q.startswith("who"):
        return False
    # Only Who is/are ... is potentially rule-safe.
    if not re.match(r"^who (?:is|are)\s+", q):
        return True
    ans_tokens = answer.strip().split()
    if not ans_tokens or len(ans_tokens) > 4:
        return True
    return False


def can_generate_safe_rule_caption(
    question: str,
    answer: str,
    caption: str = "",
    rule_name: str = "",
) -> bool:
    """Reject captions that are known-broken templates.

    Used both inside individual rules and by ``generate_caption`` as a
    final safety net before accepting a rule output.
    """
    del question, answer, rule_name  # available for future rule-specific checks
    if not caption or not caption.strip():
        return False
    low = caption.lower().strip()
    if "the answer is" in low or low.startswith("the answer"):
        return False
    if "it is not true that" in low:
        return False
    # "The in the picture is ..." / "The on the table is ..."
    if re.match(r"^the (?:in|on|at|of|to|for|near|under|over|behind)\b", low):
        return False
    if re.search(r"\bthe (?:in|on|at) the\b", low):
        return False
    # Broken existential / double-article templates from over-eager rules
    if re.match(r"^the there\b", low):
        return False
    if re.match(r"^the the\b", low):
        return False
    if re.search(r"\bmade of is\b|\bused for is\b|\bdesigned for is\b", low):
        return False
    # PP-object chopped into a fake copula: "with his is not trunk",
    # "They standing in a mud are not puddle."
    if re.search(r"\b(?:his|her|their|its|my|your|our) is(?: not)?\b", low):
        return False
    if re.search(r"\bin a \w+ (?:is|are) not\b", low):
        return False
    return True


def caption_generation_strategy(question: str, answer: str) -> str:
    """High-level routing: return ``\"rule\"`` or ``\"llm\"``.

    ``\"rule\"`` means the rule engine is allowed to try; it may still
    return ``needs_llm`` when no rule matches confidently.
    ``\"llm\"`` means skip rules for this category and go straight to the
    SLM fallback.
    """
    if should_use_llm_for_does_do(question, answer):
        return "llm"
    if should_use_llm_for_what_kind_type(question, answer):
        return "llm"
    if should_use_llm_for_is_are(question, answer):
        return "llm"
    if should_use_llm_for_who(question, answer):
        return "llm"
    return "rule"


_WHO_THE_NOUNS = {"man", "woman", "boy", "girl", "child", "guy", "lady", "gentleman"}
_WHO_A_NOUNS = {"person", "adult", "kid", "toddler", "rider", "driver", "pilot", "chef"}


def _format_who_subject(answer: str) -> Optional[str]:
    """Build a safe subject NP from a Who-answer, or None if unsure."""
    raw = answer.strip()
    if not raw:
        return None
    a = raw.lower()
    tokens = a.split()
    if len(tokens) > 4:
        return None
    if a in PRONOUNS:
        return capitalize_first(a)
    if _looks_plural_word(tokens[-1]) or a in {"people", "children", "men", "women"}:
        return capitalize_first(a)
    if a in _WHO_THE_NOUNS:
        return f"The {a}"
    if a in _WHO_A_NOUNS:
        return capitalize_first(with_article(a))
    if len(tokens) == 1:
        # Bare common noun → article; likely proper name still gets article
        # only when it looks like a common noun (all lowercase single token).
        return capitalize_first(smart_article(a))
    # Multi-word answers like "a man in a hat" — keep if already determined.
    if tokens[0] in ARTICLES:
        return capitalize_first(a)
    return None


def rule_who(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Who is/are X?' → safe '{Subject} is/are X.'

    Examples:
        'Who is in the photo?' + 'zebras' → 'Zebras are in the photo.'
        'Who is going to eat this pizza?' + 'person'
            → 'A person is going to eat this pizza.'
        'Who is the pilot?' + 'man' → 'The man is the pilot.'

    Non is/are shapes ('Who made the clock?') and uncertain answers return
    None so routing sends them to the LLM.
    """
    m = re.match(r"^who (is|are) (.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(2).strip()
    if not rest:
        return None
    # Reject rests that would create awkward templates.
    first = rest.split()[0].lower()
    if first in {"why", "how"}:
        return None

    subj = _format_who_subject(answer)
    if subj is None:
        return None

    be = "are" if _looks_plural_word(answer.strip().split()[-1]) else "is"
    caption = f"{subj} {be} {rest}."
    if not can_generate_safe_rule_caption(question, answer, caption, "who"):
        return None
    return caption


def rule_yesno_does_do(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Does/Do/Did + subject + verb ...?' → affirmative/negative statement.

    NOTE: Kept for reference / possible future narrowing, but
    ``should_use_llm_for_does_do`` / ``caption_generation_strategy`` always
    route Does/Do/Did questions to the LLM. ``generate_caption`` skips this
    rule so fragile auxiliary rewrites never ship as training captions.

    Transformations (if ever re-enabled for a narrow subset):
        Does + SUBJ + VERB + REST → SUBJ + VERB_3sg + REST
        Do   + SUBJ + VERB + REST → SUBJ + VERB_base + REST
        Did  + SUBJ + VERB + REST → SUBJ + VERB_past + REST
    """
    m = re.match(r"^(does|do|did)\s+(.+)$", question, re.I)
    if not m:
        return None
    # Soft-disable: routing should have already diverted these; return None
    # so a direct call also defers to the LLM.
    if should_use_llm_for_does_do(question, answer):
        return None
    aux = m.group(1).lower()
    rest = m.group(2).strip()
    if not rest:
        return None

    quant_m = _QUANT_OF_RE.match(rest)
    if quant_m:
        # 'one of the elephants' is the whole subject; don't chop at "of".
        subj = quant_m.group(1).strip()
        rem_tokens = quant_m.group(2).strip().split()
        if not rem_tokens:
            return None
        verb = rem_tokens[0]
        tail = " ".join(rem_tokens[1:]).strip()
    else:
        tokens = rest.split()
        if len(tokens) < 2:
            return None
        first_word = tokens[0].lower()
        if first_word in {"a", "an"}:
            # Ambiguous multi-word NP after an indefinite article — defer to
            # the SLM instead of guessing the head noun.
            return None
        if first_word in {"this", "that", "these", "those", "the"} and len(tokens) >= 3:
            subj = " ".join(tokens[:2])
            verb = tokens[2]
            tail = " ".join(tokens[3:]).strip()
        else:
            subj = tokens[0]
            verb = tokens[1]
            tail = " ".join(tokens[2:]).strip()

    if aux == "does":
        main = conjugate_3sg(verb)
    elif aux == "did":
        main = conjugate_past(verb)
    else:
        main = verb.lower()

    pred = f"{main} {tail}".strip() if tail else main
    pos = f"{prefix_the(subj)} {pred}."

    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


# ---------------------------------------------------------------------------
# Rule list — order matters: most specific rules first. Deliberately
# excludes 'which', 'where', and 'what brand/sport/room/animal/vehicle/
# food/drink' (too free-form to split into a reliable subject/predicate
# without POS tagging) — those always defer to the SLM. There is no
# catch-all fallback rule: anything unmatched is marked "needs_llm" with an
# empty caption instead of a fabricated template sentence.
#
# Comments8 removals: ``yesno_modal_have`` (Can/Could/Will/Would/Has/Have)
# and ``what_is`` are gone entirely. The modal rule mis-ordered auxiliaries
# ("This photo be could ...", "The plane fly will ..."), and 'What is ...?'
# has too many subtypes that need a real parser ("What is it called?",
# "What is it for?", "What is the weather like?"). Both families now go
# straight to the SLM instead of a fragile template.
#
# Comments9 removals: ``what_kind_type`` and the full Is/Are family
# (``is_there``, ``are_there``, ``yesno_is_anyone`` / ``everyone`` /
# ``are_any`` / ``are_all`` / ``are_both``, ``yesno_is_this_a``,
# ``yesno_is_are_possessive``, ``yesno_is_are_coordinated``,
# ``yesno_is_are_predicate``). Compound kind/type NPs and Is/Are
# subject/predicate splits were too fragile; all now go to the SLM.
# ---------------------------------------------------------------------------

RULES: List[Tuple[str, RuleFn]] = [
    ("what_color", rule_what_color),
    ("how_many", rule_how_many),
    ("what_is_doing", rule_what_is_doing),
    ("who", rule_who),
    # Kept in the list for name compatibility / inspection, but
    # should_use_llm_for_does_do + the rule body always defer to LLM.
    ("yesno_does_do", rule_yesno_does_do),
]


def generate_caption(question: str, answer: str) -> Tuple[str, str]:
    """Az (soal, javab) yek caption + esm rule ro tolid kon.

    Returns:
        (caption, rule_name) — rule_name baraye debug/statistics. When no
        rule matches confidently, returns ``("", "needs_llm")`` — an empty
        caption, never a fabricated template sentence. The row must be
        filled in by the SLM (``generate.py --llm``) before it's usable.

    Routing (see ``caption_generation_strategy``):
        - Does/Do/Did → always ``needs_llm``
        - What kind/type of ... → always ``needs_llm``
        - All Is/Are/Was/Were → always ``needs_llm``
        - Uncertain Who → ``needs_llm``
        - Otherwise try rules; reject unsafe captions
    """
    q = strip_question_mark(question).lower()
    a = normalize_answer(answer)

    # Category-level LLM routing (do not run fragile rules at all).
    if caption_generation_strategy(question, answer) == "llm":
        return "", "needs_llm"

    for rule_name, rule_fn in RULES:
        # Does/Do/Did is also guarded here in case strategy is bypassed.
        if rule_name == "yesno_does_do":
            continue
        if rule_name == "who" and should_use_llm_for_who(q, a):
            continue

        caption = rule_fn(q, a)
        if caption and can_generate_safe_rule_caption(q, a, caption, rule_name):
            return caption, rule_name

    return "", "needs_llm"
