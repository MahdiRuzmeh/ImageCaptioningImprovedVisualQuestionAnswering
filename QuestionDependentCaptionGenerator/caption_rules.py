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


def normalize_answer(answer: str) -> str:
    """Javab ro lowercase kon; adad ro be kalame tabdil kon (3 → three)."""
    a = answer.strip().lower()
    return DIGIT_TO_WORD.get(a, a)


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


def insert_not(sentence: str) -> str:
    """Negation sade: 'The boy is wearing glasses.' → 'The boy is not wearing glasses.'"""
    s = sentence.rstrip(".")
    # Copula / auxiliary already present
    m = re.match(
        r"^(There|This|That|These|Those|The|He|She|It|They|We|You|I|One|Some|Any|Each|All|Both|Most|Many|Few|Several|None)\s+(.+?)\s+"
        r"(is|are|was|were|has|have|had|can|could|will|would)\s+(.+)$",
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
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
    }:
        base = m2.group(3)
        # strip trailing 's' from 3sg (shows→show); leave irregulars alone below
        if base.lower().endswith("ies"):
            verb_base = base[:-3] + "y"
        elif base.lower().endswith("es") and base.lower()[:-2].endswith(("s", "x", "z", "ch", "sh")):
            verb_base = base[:-2]
        elif base.lower().endswith("s"):
            verb_base = base[:-1]
        else:
            verb_base = base
        return f"{m2.group(1)} {m2.group(2)} does not {verb_base.lower()} {m2.group(4)}."
    return f"It is not true that {s.lower()}."


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
    "yelling", "laughing", "crying",
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


def split_subject_predicate(rest: str) -> Tuple[str, str]:
    """Split yes/no rest into SUBJECT + PREDICATE.

    Examples:
        'these wings strong' → ('these wings', 'strong')
        'the boy wearing glasses' → ('the boy', 'wearing glasses')
        'this photo from a zoo' → ('this photo', 'from a zoo')
        'one of the giraffes eating' → ('one of the giraffes', 'eating')
        'the person on the elephant tall' → heuristic NP + last AP
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

    # Demonstrative: this/that/these/those + NOUN (+ mods) + PREDICATE
    if det in {"this", "that", "these", "those"}:
        if len(tokens) == 2:
            return tokens[0], tokens[1]
        # Default: determiner + first noun = subject; remainder = predicate
        # "these wings strong" → these wings | strong
        # "this photo show train tracks" handled elsewhere (do-support)
        return " ".join(tokens[:2]), " ".join(tokens[2:])

    # "the X ..." — take determiner + head noun as subject when short;
    # keep longer NP before a clear verbal/adjectival predicate when possible.
    if det == "the":
        if len(tokens) >= 3:
            # Prefer "the NOUN" as subject when the next token looks like a verb/adj predicate head
            return " ".join(tokens[:2]), " ".join(tokens[2:])
        return tokens[0], " ".join(tokens[1:])

    return tokens[0], " ".join(tokens[1:])


def subject_verb_from_is(rest: str) -> Tuple[str, str]:
    """Alias — keep old name for callers; delegates to ``split_subject_predicate``."""
    return split_subject_predicate(rest)


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


# Trailing question fluff for "How many X ...?" — not part of the noun phrase.
# e.g. "cookies can be seen" → "cookies", "cars are in the image" → "cars"
_HOW_MANY_STRIP_TAIL = re.compile(
    r"\s+(?:"
    r"are there|is there|"
    r"can be seen|can you see|do you see|"
    r"are visible|is visible|"
    r"(?:are|is)\s+(?:in|on|at|near|behind|under|over|inside|outside|next to)\b.*|"
    r"(?:in|on)\s+.+"
    r")$",
    re.I,
)

# Remaining verb clause after the noun: "people are standing" → predicate form.
_HOW_MANY_PREDICATE = re.compile(
    r"^(.+?)\s+((?:are|is|was|were|have|has|do|does|can|could)\s+.+)$",
    re.I,
)


def rule_how_many(question: str, answer: str) -> Optional[str]:
    """Pattern: 'How many X ...?' → 'There are {answer} X.' (or '{N} X are ...').

    Strips visibility/location scaffolding so
    "How many cookies can be seen?" → "There are two cookies."
    Keeps real predicates when present:
    "How many people are standing?" → "Two people are standing."
    Singularizes the noun when the count is one:
    "How many windows are on the caboose?" (1) → "There is one window."
    """
    m = re.match(r"^how many (.+)$", question, re.I)
    if not m:
        return None
    obj = _HOW_MANY_STRIP_TAIL.sub("", m.group(1).strip()).strip()
    if not obj:
        return None
    ans = normalize_answer(answer)

    pred_m = _HOW_MANY_PREDICATE.match(obj)
    if pred_m:
        noun, predicate = pred_m.group(1).strip(), pred_m.group(2).strip()
        if is_no(answer) or ans in {"zero", "none"}:
            return f"There are no {noun}."
        if ans in {"one", "1"}:
            # "is" for singular when the predicate starts with are/is
            predicate = re.sub(r"^(are|were)\b", "is", predicate, count=1, flags=re.I)
            return f"One {singularize_noun_phrase(noun)} {predicate}."
        return f"{capitalize_first(ans)} {noun} {predicate}."

    if is_no(answer) or ans in {"zero", "none"}:
        return f"There are no {obj}."

    if ans in {"one", "1"}:
        return f"There is one {singularize_noun_phrase(obj)}."
    return f"There are {ans} {obj}."


def rule_what_is_doing(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What is the X doing?' → 'The X is {answer}.'"""
    m = re.match(r"^what is (?:the )?(.+?) doing$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    return f"{prefix_the(subj)} is {answer}."


def rule_what_kind_type(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What kind/type of X (is/are ...)?' → 'The X (that is ...) is {answer}.'

    Examples:
        'What kind of food is shown?' + 'donuts' → 'The food is donuts.'
        'What kind of vegetable is on the sandwich?' + 'none'
            → 'There is no vegetable on the sandwich.'
        'What kind of stuffed animal is on top of the monitor?' + 'turtle'
            → 'The stuffed animal that is on top of the monitor is a turtle.'
    """
    m = re.match(r"^what (?:kind|type) of (.+)$", question, re.I)
    if not m:
        return None
    head, aux, tail = _split_head_tail(m.group(1).strip())
    if not head:
        return None
    if is_no(answer):
        return f"There is no {_plain_head_tail(head, tail)}."
    noun = smart_article(answer)
    q_lower = question.lower()
    if "this" in q_lower or "that" in q_lower:
        return f"This is {noun}."
    if "these" in q_lower or "those" in q_lower:
        return f"These are {noun}."
    subj = _describe_with_tail(head, aux, tail)
    return f"The {subj} is {noun}."


def rule_where(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Where is/are the X?' → 'The X is/are {answer}.'"""
    m = re.match(r"^where (?:is|are) (?:the )?(.+)$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    verb = "are" if " are " in question.lower() else "is"
    return f"{prefix_the(subj)} {verb} {answer}."


def rule_who(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Who is/are X?' → '{Answer} is/are X.'"""
    m = re.match(r"^who (?:is|are) (.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(1).strip()
    verb = "are" if " are " in question.lower() else "is"
    return f"{capitalize_first(answer)} {verb} {rest}."


def rule_which(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Which X ...?' → 'The X is {answer}.'"""
    m = re.match(r"^which (.+)$", question, re.I)
    if not m:
        return None
    rest = m.group(1).strip()
    return f"The {rest} is {answer}."


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


def rule_is_are_yesno(question: str, answer: str) -> Optional[str]:
    """Yes/no questions with auxiliaries → affirmative / negative statement.

    Transformations:
        Does + SUBJ + VERB + REST  → SUBJ + VERB_3sg + REST
        Do   + SUBJ + VERB + REST  → SUBJ + VERB_base + REST
        Did  + SUBJ + VERB + REST  → SUBJ + VERB_past + REST
        Is/Are/Was/Were + SUBJ + PRED → SUBJ + copula + PRED
        Has/Have/Had / Can/...     → keep auxiliary + rest

    Examples:
        'Does this photo show train tracks?' + yes
            → 'This photo shows train tracks.'
        'Are these wings strong?' + yes
            → 'These wings are strong.'
        'Are these wings strong?' + no
            → 'These wings are not strong.'
        'Is she wearing a bathing suit?' + yes
            → 'She is wearing a bathing suit.'
        'Is one of the giraffes eating?' + yes
            → 'One of the giraffes is eating.'
    """
    m = re.match(
        r"^(is|are|was|were|does|do|did|can|could|will|would|has|have|had)\s+(.+)$",
        question,
        re.I,
    )
    if not m:
        return None

    aux = m.group(1).lower()
    rest = m.group(2).strip()
    if not rest:
        return None

    pos: Optional[str] = None

    # ---- Do-support: Does / Do / Did ----
    if aux in {"does", "do", "did"}:
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
            # Subject = leading NP (demonstrative+noun or the+noun or first token)
            if tokens[0].lower() in {"this", "that", "these", "those", "the"} and len(tokens) >= 3:
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

    # ---- Copula: Is / Are / Was / Were ----
    elif aux in {"is", "are", "was", "were"}:
        # "Is this a train?" / "Is that an apple?"
        m2 = re.match(r"^(this|that|these|those)\s+(a|an|the)\s+(.+)$", rest, re.I)
        if m2:
            det, art, noun = m2.group(1), m2.group(2).lower(), m2.group(3)
            be = "are" if aux in {"are", "were"} else "is"
            if aux == "was":
                be = "was"
            elif aux == "were":
                be = "were"
            pos = f"{capitalize_first(det)} {be} {art} {noun}."
        else:
            subj, pred = split_subject_predicate(rest)
            pred = _drop_duplicate_leading_aux(pred, aux)
            if not pred:
                return None
            if aux in {"are", "were"}:
                be = "are" if aux == "are" else "were"
            elif aux == "was":
                be = "was"
            else:
                be = "is"
            pos = format_subject_be(subj, pred, be)

    # ---- Have / Has / Had / Modals: keep auxiliary ----
    elif aux in {"has", "have", "had", "can", "could", "will", "would"}:
        subj, pred = split_subject_predicate(rest)
        pred = _drop_duplicate_leading_aux(pred, aux)
        if not pred:
            # "Has it snowed?" style — treat whole rest as after aux
            if rest.lower().startswith(("this ", "that ", "these ", "those ", "the ")):
                pos = f"{capitalize_first(rest)}."
            else:
                return None
        else:
            pos = f"{prefix_the(subj)} {aux} {pred}."

    if pos is None:
        return None
    if is_yes(answer):
        return pos
    if is_no(answer):
        return insert_not(pos)
    return None


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
    """Pattern: 'What is ...?' → natural declarative with the answer.

    Examples:
        'What is in front of the giraffes?' + 'tree'
            → 'A tree is in front of the giraffes.'
        'What is in the picture?' + 'clock'
            → 'The picture shows a clock.'
        'What is the giraffe standing behind?' + 'tree'
            → 'The giraffe is standing behind a tree.'
        'What is the car?' + 'taxi'
            → 'The car is a taxi.'  (fallback-style identity)
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

    ans_np = with_article(answer)

    # "What is in/on the picture/photo/image?" → "The picture shows a/an {answer}."
    media_m = re.match(
        r"^(?:in|on)\s+(?:the\s+)?(picture|photo|image|photograph|scene|shot)$",
        rest,
        re.I,
    )
    if media_m:
        media = media_m.group(1).lower()
        return f"The {media} shows {ans_np}."

    # "What is PREP_PHRASE?" → "A {answer} is PREP_PHRASE."
    rest_l = rest.lower()
    for prep in _LOCATION_HEADS:
        if rest_l == prep or rest_l.startswith(prep + " "):
            return f"{capitalize_first(ans_np)} is {rest}."

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
    return format_the_subject(subj, answer, "is")


def rule_what_brand_sport(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What brand/sport/room/animal/vehicle/food/drink (of X) (is/are ...)?'
    → 'The {kind} (of X) (that is ...) is {answer}.'

    Examples:
        'What animal is shown?' + 'dog' → 'The animal is a dog.'
        'What animal is laying next to the dog?' + 'giraffe'
            → 'The animal that is laying next to the dog is a giraffe.'
        'What brand of computer is in the image?' + 'dell'
            → 'The brand of computer is dell.'
    """
    m = re.match(
        r"^what (brand|sport|room|animal|vehicle|food|drink)\s*(.*)$",
        question,
        re.I,
    )
    if not m:
        return None
    kind = m.group(1).lower()
    head, aux, tail = _split_head_tail(m.group(2).strip())
    subj = f"{kind} {head}".strip() if head else kind
    if is_no(answer):
        return f"There is no {_plain_head_tail(subj, tail)}."
    noun = smart_article(answer)
    subj_full = _describe_with_tail(subj, aux, tail)
    return f"The {subj_full} is {noun}."


def rule_fallback(question: str, answer: str) -> str:
    """Age hich rule match nashod, in fallback estefade mishe."""
    q_clean = strip_question_mark(question)
    if is_yes(answer):
        return f"{capitalize_first(q_clean)} — yes."
    if is_no(answer):
        return f"{capitalize_first(q_clean)} — no."
    return f"{capitalize_first(q_clean)} The answer is {answer}."


# ---------------------------------------------------------------------------
# Rule list — tartib mohem hast: rule haye specific aval, fallback akhar
# ---------------------------------------------------------------------------

RULES: List[Tuple[str, RuleFn]] = [
    ("what_color", rule_what_color),
    ("how_many", rule_how_many),
    ("what_is_doing", rule_what_is_doing),
    ("what_kind_type", rule_what_kind_type),
    ("where", rule_where),
    ("who", rule_who),
    ("which", rule_which),
    ("is_there", rule_is_there),
    ("are_there", rule_are_there),
    ("is_are_yesno", rule_is_are_yesno),
    ("what_brand_sport", rule_what_brand_sport),
    ("what_is", rule_what_is),
]


def generate_caption(question: str, answer: str) -> Tuple[str, str]:
    """Az (soal, javab) yek caption + esm rule ro tolid kon.

    Returns:
        (caption, rule_name) — rule_name baraye debug/statistics.
    """
    q = strip_question_mark(question).lower()
    a = normalize_answer(answer)

    for rule_name, rule_fn in RULES:
        caption = rule_fn(q, a)
        if caption:
            return caption, rule_name

    return rule_fallback(q, a), "fallback"
