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
    "several", "none",
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
        (mode_answer, answer_count, answer_confidence) — ``answer_count`` is
        how many of the annotators gave the mode answer (normalized,
        case-insensitive), ``answer_confidence`` is that count divided by
        the total number of annotators (rounded to 4 decimals), e.g.
        8 agreeing out of 10 -> (mode, 8, 0.8).
    """
    normalized = [a.strip().lower() for a in answers]
    if not normalized:
        return "", 0, 0.0
    mode_ans, count = Counter(normalized).most_common(1)[0]
    confidence = round(count / len(normalized), 4)
    return mode_ans, count, confidence


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
# name on the jersey" — OCR) are left OUT to avoid over-filtering ordinary
# visual questions.

_OCR_QUESTION_RE = re.compile(
    r"""
    \bwhat\s+(does|do)\s+.{0,40}?\bsay\b |        # "what does the sign say"
    \bwhat\s+is\s+written\b |                     # "what is written on..."
    \bwhat\s+words?\b |                           # "what word(s) are on..."
    \bwhat\s+letters?\b |                         # "what letters are on..."
    \blicense\s+(number|plate)\b |                # plate / registration number
    \bwhat\s+brand\b | \bwhat\s+is\s+the\s+brand\b |
    \bwhat\s+logo\b |
    \bwhat\s+number\s+is\s+(on|the|this|that)\b | # jersey / bus / plate number
    \bwhat\s+is\s+the\s+number\s+on\b |           # "what is the number on..."
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


def format_subject_be(subject: str, predicate: str, be: str) -> str:
    """SUBJECT + be + PREDICATE — article-aware alias of ``format_the_subject``."""
    return f"{prefix_the(subject)} {be} {predicate}."


# Prepositions / location heads for "What is PREP ...?"
_LOCATION_HEADS = (
    "in front of",
    "next to",
    "on top of",
    "in back of",
    "in",
    "on",
    "at",
    "near",
    "behind",
    "under",
    "over",
    "above",
    "below",
    "beside",
    "between",
    "among",
    "around",
    "inside",
    "outside",
    "across",
    "against",
)

# Single-word prepositions used to reconstruct "What is X V-ing (PREP)?"
_TRAILING_PREPOSITIONS = {
    "in", "on", "at", "near", "behind", "under", "over", "above", "below",
    "beside", "between", "among", "around", "inside", "outside", "across",
    "against", "of", "with", "for", "to", "from", "by",
}

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

# Broader action-verb whitelist for "What is SUBJECT V-ing (...)?" reconstruction
# in rule_what_is. Kept as an explicit whitelist (not a generic \w+ing regex) to
# avoid false positives on -ing NOUNS ('building', 'morning', 'something', ...).
_ACTION_PARTICIPLES = _COLOR_PARTICIPLES | {
    "looking", "watching", "flying", "walking", "running", "throwing",
    "smiling", "talking", "pulling", "pushing", "hanging", "leaning",
    "kneeling", "jumping", "swimming", "surfing", "skiing", "sleeping",
    "resting", "waiting", "pointing", "touching", "typing", "cutting",
    "cooking", "baking", "grilling", "serving", "pouring", "chasing",
    "climbing", "kicking", "hitting", "catching", "feeding", "petting",
    "washing", "cleaning", "fixing", "making", "growing", "selling",
    "buying", "showing", "displaying", "skating", "skateboarding",
    "snowboarding", "grazing", "browsing", "sniffing", "licking",
    "biting", "kissing", "hugging", "dancing", "singing", "shouting",
    "yelling", "laughing", "crying", "taking", "coming", "going",
    "getting", "putting", "setting", "giving", "leaving", "moving",
    "turning", "passing", "crossing", "entering", "exiting", "landing",
    "boarding",
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


# Quantifier-led NP: 'one of the giraffes', 'some of these dogs', ...
_QUANT_OF_RE = re.compile(
    r"^((?:one|some|any|each|none|all|part|most|both|few|many|several)\s+of\s+"
    r"(?:the|these|those|them)\s+\S+)\s+(.+)$",
    re.I,
)

# Bare quantifier + noun (no 'of'): 'one toothbrush more used than the other'
_QUANT_BARE_RE = re.compile(
    r"^((?:one|some|any|each|both|most|many|few|several)\s+\S+)\s+(.+)$",
    re.I,
)

# Past-participle / adjective-like predicates that open a verbal complement.
_PREDICATE_STARTERS = _ACTION_PARTICIPLES | {
    "made", "done", "gone", "seen", "shown", "known", "used", "based",
    "located", "situated", "covered", "filled", "attached", "connected",
    "turned", "switched", "tucked", "closed", "opened", "broken",
    "painted", "written", "printed", "dressed", "armed", "equipped",
    "hidden", "buried", "tied", "wrapped", "folded", "parked", "stopped",
    "built", "designed", "named", "called", "colored", "coloured",
}

# Short adjectival / particle predicates commonly peeled from the right.
_SHORT_PREDICATES = {
    "on", "off", "up", "down", "out", "away", "back", "open", "closed",
    "visible", "invisible", "empty", "full", "clear", "dark", "bright",
    "strong", "weak", "wet", "dry", "hot", "cold", "warm", "cool",
    "big", "small", "large", "little", "old", "new", "young", "tall",
    "short", "long", "alone", "together", "ready", "done", "gone",
    "right", "left", "straight", "backwards", "forward", "correct",
    "safe", "dangerous", "clean", "dirty", "broken", "alive", "dead",
    "asleep", "awake", "happy", "sad", "angry", "squishy", "soft",
    "hard", "sharp", "real", "fake", "true", "false", "same", "different",
    "black", "white", "red", "blue", "green", "brown", "yellow", "pink",
    "orange", "purple", "gray", "grey",
}

_PREP_WORDS = _TRAILING_PREPOSITIONS | {
    "into", "onto", "upon", "off", "over", "through", "without", "within",
}


def _leading_np_length(tokens: List[str]) -> Optional[int]:
    """Conservative length of a leading NP: (DET)? NOUN.

    Used after a preposition to find the PP object so any leftover tokens
    can be treated as the yes/no predicate
    ('on the elephants tourists' → object 'the elephants', pred 'tourists').
    """
    if not tokens:
        return None
    if tokens[0].lower() in ARTICLES | {"this", "that", "these", "those"}:
        return 2 if len(tokens) >= 2 else None
    return 1


def _split_possessive_subject(rest: str) -> Optional[Tuple[str, str]]:
    """Possessive NP subjects: \"the zebra's tail up\" → (\"the zebra's tail\", \"up\")."""
    tokens = rest.split()
    poss_idx = next(
        (
            i
            for i, t in enumerate(tokens)
            if t.endswith("'s") or t.endswith("s'")
        ),
        None,
    )
    if poss_idx is None:
        return None
    after = tokens[poss_idx + 1 :]
    if not after:
        return None

    # Prefer a verbal/participle boundary after the possessed head noun(s).
    part_rel = next(
        (
            i
            for i, t in enumerate(after)
            if t.lower() in _PREDICATE_STARTERS
        ),
        None,
    )
    if part_rel is not None and part_rel > 0:
        cut = poss_idx + 1 + part_rel
        return " ".join(tokens[:cut]), " ".join(tokens[cut:])

    # 'the cat's eyes the same color' → possessed head + complement predicate
    if len(after) >= 2 and after[1].lower() in ARTICLES:
        cut = poss_idx + 2
        return " ".join(tokens[:cut]), " ".join(tokens[cut:])

    # 'the zebra's tail up' / 'the cat's eyes open' / 'the plane's engine on'
    if len(after) == 2:
        cut = poss_idx + 2
        return " ".join(tokens[:cut]), after[1]

    # 'the boy's hat on backwards' — possessed noun then particle/adj phrase
    if len(after) >= 2 and after[1].lower() in _SHORT_PREDICATES | _PREP_WORDS:
        cut = poss_idx + 2
        return " ".join(tokens[:cut]), " ".join(after[1:])

    # 'the man's white shirt tucked in' without hitting the participle set above
    # Fallback: first token after 's is the possessed head; rest is predicate
    # when the remainder looks predicative (≥1 token and not a bare noun-only).
    if len(after) >= 2:
        cut = poss_idx + 2
        # If more adjectives sit before a later participle, grow the possessed NP.
        for j in range(1, len(after)):
            if after[j].lower() in _PREDICATE_STARTERS | _SHORT_PREDICATES:
                cut = poss_idx + 1 + j
                break
        else:
            # e.g. 'hair brown' already handled; multi-word unknown → defer shape
            cut = poss_idx + 2 if len(after) == 2 else poss_idx + 1 + 1
        if cut < len(tokens):
            return " ".join(tokens[:cut]), " ".join(tokens[cut:])
    return None


def _split_coordinated_subject(rest: str) -> Optional[Tuple[str, str]]:
    """Coordinated NP subjects: \"the clock and owl made ...\"."""
    m = re.match(
        r"^((?:the|a|an|this|that|these|those)\s+\S+\s+and\s+(?:(?:the|a|an)\s+)?\S+)\s+(.+)$",
        rest,
        re.I,
    )
    if not m:
        return None
    subj, pred = m.group(1).strip(), m.group(2).strip()
    if not pred:
        return None
    return subj, pred


def _split_pp_modified_subject(rest: str) -> Optional[Tuple[str, str]]:
    """NP + PP-modifier + trailing predicate.

    'the people on the elephants tourists'
        → ('the people on the elephants', 'tourists')
    'the ground near the waterfront squishy'
        → ('the ground near the waterfront', 'squishy')

    When the PP consumes the remainder ('this photo from a zoo'), returns
    None so a later heuristic can treat the PP as the predicate.
    """
    tokens = rest.split()
    prep_indices = [
        i
        for i, t in enumerate(tokens)
        if t.lower() in _PREP_WORDS and 0 < i < len(tokens) - 1
    ]
    for prep_i in reversed(prep_indices):
        rem = tokens[prep_i + 1 :]
        obj_len = _leading_np_length(rem)
        if obj_len is None or obj_len >= len(rem):
            continue
        pred_tokens = rem[obj_len:]
        # Trailing predicate should be short/contentful, not another long NP
        # introduced by 'and' (coordination is handled separately).
        if pred_tokens[0].lower() == "and":
            continue
        subj = " ".join(tokens[: prep_i + 1 + obj_len])
        pred = " ".join(pred_tokens)
        return subj, pred
    return None


def _split_right_predicate(rest: str) -> Optional[Tuple[str, str]]:
    """Peel a short final particle/adjective: 'the stove light on'.

    Phrasal verbs keep the -ing verb with the particle:
    'this plane taking off' → ('this plane', 'taking off').
    """
    tokens = rest.split()
    if len(tokens) < 3:
        return None
    last = tokens[-1].lower()
    if last not in _SHORT_PREDICATES:
        return None
    subj_tokens = tokens[:-1]
    first = subj_tokens[0].lower()
    if first not in ARTICLES | PRONOUNS | _QUANTIFIER_LEAD | {
        "this", "that", "these", "those",
    }:
        return None
    if len(subj_tokens) >= 2 and subj_tokens[-1].lower() in _PREP_WORDS:
        return None
    # '... V-ing off/up/on' — particle belongs to the verb, not the subject NP
    if len(subj_tokens) >= 2 and subj_tokens[-1].lower().endswith("ing"):
        return (
            " ".join(subj_tokens[:-1]),
            f"{subj_tokens[-1]} {tokens[-1]}",
        )
    return " ".join(subj_tokens), tokens[-1]


def split_subject_predicate(rest: str) -> Tuple[str, str]:
    """Split yes/no rest into SUBJECT + PREDICATE.

    Specialized extractors run first (possessive, coordination, PP-modified
    NP, right-edge particle). Only then fall back to participle / determiner
    heuristics. Examples:

        'these wings strong' → ('these wings', 'strong')
        'the stove light on' → ('the stove light', 'on')
        \"the zebra's tail up\" → (\"the zebra's tail\", 'up')
        'the people on the elephants tourists'
            → ('the people on the elephants', 'tourists')
        'the clock and owl made in the same artistic fashion'
            → ('the clock and owl', 'made in the same artistic fashion')
        'the boy wearing glasses' → ('the boy', 'wearing glasses')
        'one of the giraffes eating' → ('one of the giraffes', 'eating')

    Returns ``("", "")`` when the subject can't be split reliably (e.g. an
    indefinite article followed by a multi-word NP like 'a military
    person') — callers should treat that as "defer to the SLM".
    """
    quant_m = _QUANT_OF_RE.match(rest)
    if quant_m:
        return quant_m.group(1).strip(), quant_m.group(2).strip()
    quant_bare_m = _QUANT_BARE_RE.match(rest)
    if quant_bare_m:
        return quant_bare_m.group(1).strip(), quant_bare_m.group(2).strip()

    tokens = rest.split()
    if len(tokens) < 2:
        return rest, ""

    det = tokens[0].lower()
    if det in {"a", "an"}:
        return "", ""

    for splitter in (
        _split_possessive_subject,
        _split_coordinated_subject,
        _split_pp_modified_subject,
    ):
        hit = splitter(rest)
        if hit is not None:
            return hit

    # Verb-ing / past-participle predicate: scan for a known starter after
    # the first token so adjective-modified subjects stay intact
    # ('the small elephant touching the big elephant').
    participle_idx = next(
        (
            i
            for i, t in enumerate(tokens[1:], start=1)
            if t.lower() in _PREDICATE_STARTERS
        ),
        None,
    )
    if participle_idx is not None:
        return (
            " ".join(tokens[:participle_idx]),
            " ".join(tokens[participle_idx:]),
        )

    right = _split_right_predicate(rest)
    if right is not None:
        return right

    # Demonstrative: this/that/these/those + NOUN (+ mods) + PREDICATE
    if det in {"this", "that", "these", "those"}:
        if len(tokens) == 2:
            return tokens[0], tokens[1]
        return " ".join(tokens[:2]), " ".join(tokens[2:])

    # "the X ..." — determiner + head noun as subject when short.
    if det == "the":
        if len(tokens) >= 3:
            return " ".join(tokens[:2]), " ".join(tokens[2:])
        return tokens[0], " ".join(tokens[1:])

    return tokens[0], " ".join(tokens[1:])


def _drop_duplicate_leading_aux(pred: str, aux: str) -> str:
    """Collapse a duplicated auxiliary from a typo'd source question.

    e.g. 'Are the bikers are in a race?' — the outer ``aux`` ('are') was
    already consumed once; if the predicate starts with the same word
    again, drop the repeat so we don't emit 'are not are in a race.'.
    """
    tokens = pred.split()
    if tokens and tokens[0].lower() == aux.lower():
        return " ".join(tokens[1:])
    return pred


# Near-content-free trailing clauses ("is shown", "is in the picture", "is
# this") — dropping these loses nothing meaningful.
_FILLER_TAIL_RE = re.compile(
    r"^(?:shown|visible|here|there|this|that|these|those|it)"
    r"(?:\s+(?:here|there|now|today))?$"
    r"|^in (?:this|the) (?:picture|photo|image|scene)$",
    re.I,
)


def _is_filler_tail(tail: str) -> bool:
    """True for near-content-free tails ('shown', 'in the picture', 'this', ...)."""
    return bool(_FILLER_TAIL_RE.match(tail.strip()))


def _split_head_tail(phrase: str) -> Tuple[str, str, str]:
    """Split '<head> is/are <tail>' into (head, aux, tail).

    'silverware is on the plates' -> ('silverware', 'is', 'on the plates')
    'is shown' -> ('', 'is', 'shown')
    'plane' (no aux) -> ('plane', '', '')
    """
    m = re.match(r"^(.*?)\s*\b(is|are)\b\s+(.+)$", phrase, re.I)
    if not m:
        return phrase.strip(), "", ""
    return m.group(1).strip(), m.group(2).lower(), m.group(3).strip()


def _describe_with_tail(head: str, aux: str, tail: str) -> str:
    """'head' alone, or 'head that is/are tail' when the tail carries real info.

    Avoids the double-copula bug ('The silverware is on the plates is a
    knife.') while keeping restrictive info instead of silently dropping it
    ('The animal that is laying next to the dog is a giraffe.').
    """
    if not tail or _is_filler_tail(tail):
        return head
    connector = f"that {aux}" if aux else "that is"
    if head:
        return f"{head} {connector} {tail}"
    return f"{connector} {tail}"


def _plain_head_tail(head: str, tail: str) -> str:
    """'head' alone, or 'head tail' (no relative clause) — for negative sentences."""
    if not tail or _is_filler_tail(tail):
        return head
    return f"{head} {tail}".strip()


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


def rule_how_many(question: str, answer: str) -> Optional[str]:
    """Pattern: 'How many X are/is there|in/on ...?' → 'There is/are {answer} X.'

    Examples:
        'How many cookies are there?' + '2' → 'There are two cookies.'
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
        return f"There are no {noun}."
    if ans in {"one", "1"}:
        return f"There is one {_singularize_full_np(noun)}."
    return f"There are {ans} {noun}."


def rule_what_is_doing(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What is the X doing?' → 'The X is {answer}.'"""
    m = re.match(r"^what is (?:the )?(.+?) doing$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    return f"{prefix_the(subj)} is {answer}."


# Broad category heads where the answer is a hyponym/instance, not a
# pre-modifier: "What kind of food ...?" + "donuts" → "The food is donuts."
# (not "donut food").
_KIND_IDENTITY_HEADS = {
    "food", "foods", "animal", "animals", "vegetable", "vegetables",
    "fruit", "fruits", "meat", "bird", "birds", "fish", "dog", "dogs",
    "cat", "cats", "flower", "flowers", "plant", "plants",
    "vehicle", "vehicles", "car", "cars", "material", "fabric", "metal",
    "wood", "plastic", "sport", "sports", "game", "games", "drink",
    "drinks", "beverage", "instrument", "tool", "tools", "furniture",
    "clothing", "clothes", "person", "people", "man", "woman", "breed",
    "species", "style", "flavor", "flavour", "color", "colour", "race",
    "ethnicity", "profession", "job", "occupation", "brand", "model",
    "pattern", "shape", "size", "texture", "weather", "emotion",
}


def _compose_kind_np(head: str, answer: str) -> str:
    """Build the NP that names the kind/type, preserving the semantic head.

    'celebration' + 'birthday' → 'a birthday celebration'
    'sign' + 'stop' → 'a stop sign'
    'court' + 'soccer' → 'a soccer court'
    'food' + 'donuts' → 'donuts'          (identity head)
    'stuffed animal' + 'turtle' → 'a turtle'  (head ends with identity noun)
    """
    ans = answer.strip()
    head = head.strip()
    if not ans:
        return smart_article(head)
    ans_l = ans.lower()
    head_l = head.lower()
    head_sing = singularize_noun_phrase(head_l)
    head_tokens = head_sing.split()
    head_key = singularize_word(head_tokens[-1]) if head_tokens else head_sing

    # Answer already contains the head ('stop sign', 'soccer ball', ...)
    if head_sing in ans_l or head_key in ans_l.split():
        return smart_article(ans)

    # Broad category → answer stands alone as the instance.
    if head_key in _KIND_IDENTITY_HEADS or head_sing in _KIND_IDENTITY_HEADS:
        return smart_article(ans)

    # Modifier + head compound: preserve the semantic head noun.
    return smart_article(f"{ans_l} {head_sing}")


def rule_what_kind_type(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What kind/type of X (is/are ...)?' → preserve head noun X.

    Examples:
        'What kind of celebration is this?' + 'birthday'
            → 'This is a birthday celebration.'
        'What kind of sign is in the picture?' + 'stop'
            → 'The sign is a stop sign.'
        'What kind of court is in the background?' + 'soccer'
            → 'The court that is in the background is a soccer court.'
        'What kind of food is shown?' + 'donuts'
            → 'The food is donuts.'
        'What kind of vegetable is on the sandwich?' + 'none'
            → 'There is no vegetable on the sandwich.'
    """
    m = re.match(r"^what (?:kind|type) of (.+)$", question, re.I)
    if not m:
        return None
    head, aux, tail = _split_head_tail(m.group(1).strip())
    if not head:
        return None
    if is_no(answer):
        return f"There is no {_plain_head_tail(head, tail)}."

    noun = _compose_kind_np(head, answer)
    q_lower = question.lower()
    if re.search(r"\b(?:is|are)\s+(?:this|that)\b", q_lower) or re.search(
        r"\b(?:this|that)\s*$", m.group(1).strip().lower()
    ):
        return f"This is {noun}."
    if re.search(r"\b(?:is|are)\s+(?:these|those)\b", q_lower) or re.search(
        r"\b(?:these|those)\s*$", m.group(1).strip().lower()
    ):
        return f"These are {noun}."

    # Filler tails ('shown', 'in the picture') drop; real locations stay as
    # a relative clause so we don't emit double-copula sentences.
    subj = _describe_with_tail(head, aux, tail)
    return f"The {subj} is {noun}."


def rule_who(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Who is/are X?' → '{Answer} is/are X.'"""
    m = re.match(r"^who (?:is|are) (.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(1).strip()
    verb = "are" if " are " in question.lower() else "is"
    return f"{capitalize_first(answer)} {verb} {rest}."


def rule_is_there(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is there a/an X?' → 'There is a X.' / 'There is no X.'"""
    m = re.match(r"^is there (?:a|an) (.+)$", question, re.I)
    if not m:
        return None
    obj = m.group(1).strip()
    if is_yes(answer):
        return f"There is {with_article(obj)}."
    if is_no(answer):
        return f"There is no {obj}."
    return f"There is {with_article(answer)} {obj}."


def rule_are_there(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Are there (any) X?' → 'There are X.' / 'There are no X.'"""
    m = re.match(r"^are there (?:any )?(.+)$", question, re.I)
    if not m:
        return None
    obj = m.group(1).strip()
    if is_yes(answer):
        return f"There are {obj}."
    if is_no(answer):
        return f"There are no {obj}."
    return f"There are {answer} {obj}."


# ---------------------------------------------------------------------------
# Yes/no rules — deliberately split into narrow, high-precision sub-rules
# instead of one generic "is_are_yesno" catch-all. Each rule only fires for
# a syntactic shape it can transform with confidence; anything else (e.g. an
# indefinite-article subject like 'a military person', whose head noun can't
# be found without POS tagging) returns None so the item defers to the SLM
# instead of producing a guessed, possibly-wrong sentence.
# ---------------------------------------------------------------------------


def rule_yesno_is_anyone(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is/Are anyone ...?' → 'Someone is {pred}.' / 'No one is {pred}.'

    Example:
        'Is anyone wearing wrist protection?' + yes
            → 'Someone is wearing wrist protection.'
        'Is anyone wearing wrist protection?' + no
            → 'No one is wearing wrist protection.'
    """
    m = re.match(r"^(?:is|are)\s+anyone\s+(.+)$", question, re.I)
    if not m:
        return None
    pred = m.group(1).strip()
    if not pred:
        return None
    if is_yes(answer):
        return f"Someone is {pred}."
    if is_no(answer):
        return f"No one is {pred}."
    return None


def rule_yesno_are_any(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Are any of ...?' → 'At least one of {subj} is {pred}.' / 'None of {subj} is {pred}.'

    Example:
        'Are any of the animals eating?' + yes
            → 'At least one of the animals is eating.'
        'Are any of the animals eating?' + no
            → 'None of the animals is eating.'
    """
    m = re.match(r"^are any of (.+)$", question, re.I)
    if not m:
        return None
    subj, pred = split_subject_predicate(m.group(1).strip())
    if not subj or not pred:
        return None
    if is_yes(answer):
        return f"At least one of {subj} is {pred}."
    if is_no(answer):
        return f"None of {subj} is {pred}."
    return None


def rule_yesno_are_all(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is/Are all ...?' → 'All {subj} {be} {pred}.' / 'Not all {subj} {be} {pred}.'

    Example:
        'Are all the flowers white?' + no
            → 'Not all the flowers are white.'
        'Are all the flowers white?' + yes
            → 'All the flowers are white.'
    """
    m = re.match(r"^(is|are)\s+all\s+(.+)$", question, re.I)
    if not m:
        return None
    be = m.group(1).lower()
    subj, pred = split_subject_predicate(m.group(2).strip())
    if not subj or not pred:
        return None
    if is_yes(answer):
        return f"All {subj} {be} {pred}."
    if is_no(answer):
        return f"Not all {subj} {be} {pred}."
    return None


def rule_yesno_are_both(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Are both ...?' → 'Both {subj} are {pred}.' / 'Not both {subj} are {pred}.'

    Example:
        'Are both giraffes standing?' + no
            → 'Not both giraffes are standing.'
        'Are both giraffes standing?' + yes
            → 'Both giraffes are standing.'
    """
    m = re.match(r"^are both (.+)$", question, re.I)
    if not m:
        return None
    subj, pred = split_subject_predicate(m.group(1).strip())
    if not subj or not pred:
        return None
    if is_yes(answer):
        return f"Both {subj} are {pred}."
    if is_no(answer):
        return f"Not both {subj} are {pred}."
    return None


def rule_yesno_does_do(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Does/Do/Did + subject + verb ...?' → affirmative/negative statement.

    Transformations:
        Does + SUBJ + VERB + REST → SUBJ + VERB_3sg + REST
        Do   + SUBJ + VERB + REST → SUBJ + VERB_base + REST
        Did  + SUBJ + VERB + REST → SUBJ + VERB_past + REST

    Example:
        'Does this photo show train tracks?' + yes
            → 'This photo shows train tracks.'
    """
    m = re.match(r"^(does|do|did)\s+(.+)$", question, re.I)
    if not m:
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


def rule_yesno_modal_have(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Can/Could/Will/Would/Has/Have/Had ...?' → keep the auxiliary.

    Example:
        'Could this photo be from a zoo?' + yes
            → 'This photo could be from a zoo.'
    """
    m = re.match(r"^(can|could|will|would|has|have|had)\s+(.+)$", question, re.I)
    if not m:
        return None
    aux = m.group(1).lower()
    rest = m.group(2).strip()
    if not rest:
        return None

    subj, pred = split_subject_predicate(rest)
    pred = _drop_duplicate_leading_aux(pred, aux)
    if not subj or not pred:
        # "Has it snowed?" style — treat whole rest as after aux
        if rest.lower().startswith(("this ", "that ", "these ", "those ", "the ")):
            pos = f"{capitalize_first(rest)}."
        else:
            return None
    else:
        pos = f"{prefix_the(subj)} {aux} {pred}."

    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


def rule_yesno_is_this_a(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is/Are this/that/these/those a/an/the X?'

    Example:
        'Is this a horse?' + no → 'This is not a horse.'
        'Is that an apple?' + yes → 'That is an apple.'
    """
    m = re.match(
        r"^(is|are|was|were)\s+(this|that|these|those)\s+(a|an|the)\s+(.+)$",
        question,
        re.I,
    )
    if not m:
        return None
    aux, det, art, noun = (
        m.group(1).lower(),
        m.group(2),
        m.group(3).lower(),
        m.group(4).strip(),
    )
    be = {"is": "is", "was": "was", "are": "are", "were": "were"}[aux]
    pos = f"{capitalize_first(det)} {be} {art} {noun}."
    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


def rule_yesno_is_are_possessive(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is/Are the X's Y ...?' possessive subjects.

    Example:
        \"Is the zebra's tail up?\" + no → \"The zebra's tail is not up.\"
        \"Are the cat's eyes open?\" + yes → \"The cat's eyes are open.\"
    """
    m = re.match(r"^(is|are|was|were)\s+(.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(2).strip()
    if not re.search(r"\S+(?:'s|s')\b", rest):
        return None
    aux = m.group(1).lower()
    be = {"is": "is", "was": "was", "are": "are", "were": "were"}[aux]
    hit = _split_possessive_subject(rest)
    if not hit:
        return None
    subj, pred = hit
    pred = _drop_duplicate_leading_aux(pred, aux)
    if not subj or not pred:
        return None
    pos = format_subject_be(subj, pred, be)
    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


def rule_yesno_is_are_coordinated(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is/Are the X and Y ...?' coordinated subjects.

    Example:
        'Are the clock and owl made in the same artistic fashion?' + no
            → 'The clock and owl are not made in the same artistic fashion.'
    """
    m = re.match(r"^(is|are|was|were)\s+(.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(2).strip()
    hit = _split_coordinated_subject(rest)
    if not hit:
        return None
    aux = m.group(1).lower()
    be = {"is": "is", "was": "was", "are": "are", "were": "were"}[aux]
    subj, pred = hit
    pred = _drop_duplicate_leading_aux(pred, aux)
    if not subj or not pred:
        return None
    pos = format_subject_be(subj, pred, be)
    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


def rule_yesno_is_are_predicate(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Is/Are/Was/Were + subject + (adjective | verb-ing | ...)?'.

    Example:
        'Are these wings strong?' + yes → 'These wings are strong.'
        'Are these wings strong?' + no  → 'These wings are not strong.'
        'Is she wearing a bathing suit?' + yes
            → 'She is wearing a bathing suit.'
        'Is one of the giraffes eating?' + yes
            → 'One of the giraffes is eating.'
        'Is the stove light on?' + yes
            → 'The stove light is on.'
        'Are the people on the elephants tourists?' + yes
            → 'The people on the elephants are tourists.'

    Subjects led by an indefinite article ('a military person') are
    ambiguous without POS tagging — this rule returns None for those so
    the SLM handles them. Possessive / coordinated / 'is this a' shapes
    are handled by earlier specialized rules.
    """
    m = re.match(r"^(is|are|was|were)\s+(.+)$", question, re.I)
    if not m:
        return None
    aux = m.group(1).lower()
    rest = m.group(2).strip()
    if not rest:
        return None
    if rest.split()[0].lower() in {"a", "an"}:
        return None

    be = {"is": "is", "was": "was", "are": "are", "were": "were"}[aux]

    subj, pred = split_subject_predicate(rest)
    pred = _drop_duplicate_leading_aux(pred, aux)
    if not subj or not pred:
        return None
    pos = format_subject_be(subj, pred, be)

    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


# Surfaces that typically *display* text — prefer "The sign says ...".
_TEXT_SURFACE_NOUNS = {
    "sign", "signs", "board", "boards", "label", "labels", "poster",
    "posters", "banner", "banners", "plaque", "plaques", "screen",
    "display", "billboard", "billboards", "shirt", "jersey", "paper",
    "page", "book", "menu", "box", "package", "bottle", "wrapper",
    "sticker", "tag", "plate", "monitor", "tv", "television", "scoreboard",
}

_TEXT_RENDER_VERBS = {
    "printed", "written", "painted", "displayed", "shown", "embossed",
    "engraved", "stamped", "drawn", "scribbled",
}

# Bare "What is V-ing PREP ...?" where "what" is the theme/subject of V-ing.
_LOCATIVE_PARTICIPLES = {
    "hanging", "sitting", "standing", "lying", "resting", "floating",
    "mounted", "attached", "tied", "placed", "located", "leaning",
    "parked", "growing", "sticking", "protruding", "dangling", "suspended",
    "shown", "displayed", "hidden", "buried", "parked", "waiting",
}


def _plural_be(answer: str) -> str:
    """Pick is/are from a bare answer noun (best-effort)."""
    ans_l = answer.strip().lower()
    if not ans_l:
        return "is"
    if ans_l in {"people", "children", "men", "women", "mice", "geese"}:
        return "are"
    if " " in ans_l:
        return "is"
    if ans_l.endswith("s") and not ans_l.endswith(("ss", "us", "is", "ous")):
        return "are"
    return "is"


def _rule_what_is_text_render(rest: str, answer: str) -> Optional[str]:
    """'What is printed/written/painted on SURFACE?' → surface says/displays answer.

    Examples:
        'printed on the orange sign' + pizza
            → \"The orange sign says 'pizza'.\"
        'written on the plane' + china airlines
            → 'China airlines is written on the plane.'
    """
    m = re.match(
        r"^(" + "|".join(_TEXT_RENDER_VERBS) + r")\s+"
        r"(on|in|across|over|under|inside|onto|upon|along)\s+(.+)$",
        rest,
        re.I,
    )
    if not m:
        return None
    participle = m.group(1).lower()
    prep = m.group(2).lower()
    surface = m.group(3).strip()
    if not surface:
        return None

    surface_core = re.sub(r"^(?:the|a|an)\s+", "", surface, flags=re.I).strip()
    head = surface_core.split()[-1].lower() if surface_core else ""
    looks_textual = head in _TEXT_SURFACE_NOUNS or any(
        key in surface_core.lower().split()
        for key in ("sign", "board", "label", "banner", "poster", "shirt", "jersey")
    )

    if looks_textual and participle in {
        "printed", "written", "painted", "displayed", "shown", "embossed",
        "engraved", "stamped",
    }:
        verb = "says" if participle in {"printed", "written", "stamped", "embossed", "engraved"} else "displays"
        return f"{prefix_the(surface_core)} {verb} '{answer}'."

    return f"{capitalize_first(answer)} is {participle} {prep} {surface}."


def _rule_what_is_bare_participle(rest: str, answer: str) -> Optional[str]:
    """'What is hanging/sitting/... PREP ...?' — answer is the theme subject.

    'hanging above the stove' + lights → 'Lights are hanging above the stove.'
    'shown here' + scooter → 'A scooter is shown here.'
    """
    tokens = rest.split()
    if not tokens or tokens[0].lower() not in _LOCATIVE_PARTICIPLES:
        return None
    ans = smart_article(answer)
    return f"{capitalize_first(ans)} {_plural_be(answer)} {rest}."


def _rule_what_is_participle(rest: str, answer: str) -> Optional[str]:
    """Sub-pattern of rule_what_is: 'What is SUBJECT V-ing (PREP ...)?'.

    Keeps the verb instead of collapsing it into 'SUBJECT is ANSWER':
        'the giraffe standing behind' + tree -> 'The giraffe is standing behind a tree.'
        'the vase sitting on' + railing      -> 'The vase is sitting on a railing.'
        'the animal eating' + grass          -> 'The animal is eating grass.'
        'she holding' + broccoli             -> 'She is holding broccoli.'
        'this person wearing on head' + hat  -> 'This person is wearing a hat on the head.'
    """
    tokens = rest.split()
    verb_idx = next(
        (i for i, t in enumerate(tokens) if t.lower() in _ACTION_PARTICIPLES),
        None,
    )
    if verb_idx is None:
        return None

    subject_tokens = tokens[:verb_idx]
    if not subject_tokens:
        return None
    if len(subject_tokens) == 1 and subject_tokens[0].lower() not in PRONOUNS:
        return None

    verb = tokens[verb_idx].lower()
    trailing_tokens = tokens[verb_idx + 1 :]
    subject = prefix_the(" ".join(subject_tokens))
    ans = smart_article(answer)

    if not trailing_tokens:
        predicate = f"{verb} {ans}"
    elif trailing_tokens[-1].lower() in _TRAILING_PREPOSITIONS:
        # Dangling preposition — the answer is its object.
        # e.g. 'sitting on' + railing -> 'sitting on a railing'
        predicate = f"{verb} {' '.join(trailing_tokens)} {ans}"
    else:
        # Trailing is already a full prepositional phrase (has its own noun);
        # the answer is the verb's direct object instead.
        # e.g. 'wearing on head' + hat -> 'wearing a hat on the head'
        tail_tokens = list(trailing_tokens)
        if (
            tail_tokens[0].lower() in _TRAILING_PREPOSITIONS
            and len(tail_tokens) >= 2
            and tail_tokens[-1].lower() not in ARTICLES
        ):
            tail_tokens = [tail_tokens[0], "the"] + tail_tokens[1:]
        predicate = f"{verb} {ans} {' '.join(tail_tokens)}"

    return f"{subject} is {predicate}."


def rule_what_is(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What is ...?' — role-aware declarative with the answer.

    Understands that 'what' is not always the sentence subject:

        'What is printed on the orange sign?' + pizza
            → \"The orange sign says 'pizza'.\"
        'What is hanging above the stove?' + lights
            → 'Lights are hanging above the stove.'
        'What is in front of the giraffes?' + tree
            → 'A tree is in front of the giraffes.'
        'What is in the picture?' + clock
            → 'The picture shows a clock.'
        'What is the giraffe standing behind?' + tree
            → 'The giraffe is standing behind a tree.'
        'What is the car?' + taxi
            → 'The car is a taxi.'
    """
    m = re.match(r"^what is\s+(.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(1).strip()
    if not rest or answer in YES | NO:
        return None

    # "What is SUBJECT V-ing (PREP ...)?" — keep the verb, don't collapse it.
    participle_cap = _rule_what_is_participle(rest, answer)
    if participle_cap:
        return participle_cap

    # "What is printed/written/... on SURFACE?"
    text_cap = _rule_what_is_text_render(rest, answer)
    if text_cap:
        return text_cap

    # "What is hanging/sitting/... PREP ...?" — answer is the theme.
    bare_cap = _rule_what_is_bare_participle(rest, answer)
    if bare_cap:
        return bare_cap

    ans_np = smart_article(answer)

    # "What is in/on the picture/photo/image?" → "The picture shows a/an {answer}."
    media_m = re.match(
        r"^(?:in|on)\s+(?:the\s+)?(picture|photo|image|photograph|scene|shot)$",
        rest,
        re.I,
    )
    if media_m:
        media = media_m.group(1).lower()
        return f"The {media} shows {ans_np}."

    # "What is PREP_PHRASE?" → "A {answer} is/are PREP_PHRASE."
    rest_l = rest.lower()
    for prep in _LOCATION_HEADS:
        if rest_l == prep or rest_l.startswith(prep + " "):
            return f"{capitalize_first(ans_np)} {_plural_be(answer)} {rest}."

    # "What is the X?" / "What is X?" → "The X is {answer}."
    # Drop trailing location fluff only when subject remains non-empty.
    subj_m = re.match(
        r"^(?:the\s+)?(.+?)(?:\s+(?:on|in|near|at|under|over|behind)\s+.+)?$",
        rest,
        re.I,
    )
    if not subj_m:
        return None
    subj = subj_m.group(1).strip()
    # Avoid swallowing pure location phrases that somehow missed the prep list
    if not subj or subj.lower().split()[0] in {
        "in",
        "on",
        "at",
        "near",
        "behind",
        "under",
        "over",
        "among",
        "between",
    }:
        return f"{capitalize_first(ans_np)} is {rest}."
    return format_the_subject(
        subj,
        smart_article(answer) if len(answer.split()) == 1 else answer,
        "is",
    )


# ---------------------------------------------------------------------------
# Rule list — order matters: most specific rules first. Deliberately
# excludes 'which', 'where', and 'what brand/sport/room/animal/vehicle/
# food/drink' (too free-form to split into a reliable subject/predicate
# without POS tagging) — those always defer to the SLM. There is no
# catch-all fallback rule: anything unmatched is marked "needs_llm" with an
# empty caption instead of a fabricated template sentence.
# ---------------------------------------------------------------------------

RULES: List[Tuple[str, RuleFn]] = [
    ("what_color", rule_what_color),
    ("how_many", rule_how_many),
    ("what_is_doing", rule_what_is_doing),
    ("what_kind_type", rule_what_kind_type),
    ("who", rule_who),
    ("is_there", rule_is_there),
    ("are_there", rule_are_there),
    ("yesno_is_anyone", rule_yesno_is_anyone),
    ("yesno_are_any", rule_yesno_are_any),
    ("yesno_are_all", rule_yesno_are_all),
    ("yesno_are_both", rule_yesno_are_both),
    ("yesno_does_do", rule_yesno_does_do),
    ("yesno_modal_have", rule_yesno_modal_have),
    ("yesno_is_this_a", rule_yesno_is_this_a),
    ("yesno_is_are_possessive", rule_yesno_is_are_possessive),
    ("yesno_is_are_coordinated", rule_yesno_is_are_coordinated),
    ("yesno_is_are_predicate", rule_yesno_is_are_predicate),
    ("what_is", rule_what_is),
]


def generate_caption(question: str, answer: str) -> Tuple[str, str]:
    """Az (soal, javab) yek caption + esm rule ro tolid kon.

    Returns:
        (caption, rule_name) — rule_name baraye debug/statistics. When no
        rule matches confidently, returns ``("", "needs_llm")`` — an empty
        caption, never a fabricated template sentence. The row must be
        filled in by the SLM (``generate.py --llm``) before it's usable.
    """
    q = strip_question_mark(question).lower()
    a = normalize_answer(answer)

    for rule_name, rule_fn in RULES:
        caption = rule_fn(q, a)
        if caption:
            return caption, rule_name

    return "", "needs_llm"
