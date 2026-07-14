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


def insert_not(sentence: str) -> str:
    """Negation sade: 'The boy is wearing glasses.' → 'The boy is not wearing glasses.'"""
    s = sentence.rstrip(".")
    m = re.match(
        r"^(The|There|This|That|These|Those)\s+(.+?)\s+(is|are|was|were|has|have|can)\s+(.+)$",
        s,
        re.I,
    )
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} not {m.group(4)}."
    return f"It is not true that {s.lower()}."


def format_the_subject(subject: str, predicate: str, be: str) -> str:
    """Jomle besaz ba 'The X is/are Y' — double 'the' ro handle kon."""
    subj = subject.strip()
    if subj.startswith("the "):
        return f"The {subj[4:]} {be} {predicate}."
    return f"The {subj} {be} {predicate}."


def subject_verb_from_is(rest: str) -> Tuple[str, str]:
    """Az 'the boy wearing glasses' subject va predicate ro joda kon."""
    tokens = rest.split()
    if len(tokens) < 2:
        return rest, ""

    # Heuristic: 2-gram subject (the boy, the red car, ...)
    if tokens[0] == "the" and len(tokens) >= 3:
        subj = " ".join(tokens[:2])
        pred = " ".join(tokens[2:])
        return subj, pred

    subj = tokens[0]
    pred = " ".join(tokens[1:])
    return subj, pred


# ---------------------------------------------------------------------------
# Rule functions — har rule yek pattern soal ro handle mikone
# ---------------------------------------------------------------------------


def rule_what_color(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What color is/are the X?' → 'The X is/are {answer}.'"""
    m = re.match(r"^what color(?:s)? (?:is|are) (?:the )?(.+)$", question, re.I)
    if not m:
        return None
    obj = m.group(1).strip()
    verb = "are" if " are " in question.lower() else "is"
    return f"The {obj} {verb} {answer}."


def rule_how_many(question: str, answer: str) -> Optional[str]:
    """Pattern: 'How many X ...?' → 'There are {answer} X.'"""
    m = re.match(
        r"^how many (.+?)(?: are there| is there| are in| is in| in .+| on .+)?$",
        question,
        re.I,
    )
    if not m:
        return None
    obj = m.group(1).strip()
    ans = normalize_answer(answer)

    if is_no(answer) or ans in {"zero", "none"}:
        return f"There are no {obj}."

    if ans in {"one", "1"}:
        return f"There is one {obj}."
    return f"There are {ans} {obj}."


def rule_what_is_doing(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What is the X doing?' → 'The X is {answer}.'"""
    m = re.match(r"^what is (?:the )?(.+?) doing$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    act = answer if answer.endswith("ing") else answer
    return f"The {subj} is {act}."


def rule_what_kind_type(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What kind/type of X ...?' → 'This is a {answer}.' ya 'The X is ...'"""
    m = re.match(
        r"^what (?:kind|type) of (.+?)(?: is this| are these| is that| are those| is in .+)?$",
        question,
        re.I,
    )
    if not m:
        return None
    obj = m.group(1).strip()
    noun = with_article(answer)
    q_lower = question.lower()
    if "this" in q_lower or "that" in q_lower:
        return f"This is {noun}."
    if "these" in q_lower or "those" in q_lower:
        return f"These are {noun}."
    return f"The {obj} is {noun}."


def rule_where(question: str, answer: str) -> Optional[str]:
    """Pattern: 'Where is/are the X?' → 'The X is/are {answer}.'"""
    m = re.match(r"^where (?:is|are) (?:the )?(.+)$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    verb = "are" if " are " in question.lower() else "is"
    return f"The {subj} {verb} {answer}."


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
    """Pattern: 'Is/Are/Does/Do ...?' + yes/no → jomle-ye affirmative/negative."""
    m = re.match(r"^(is|are|was|were|does|do|did|can|could) (.+)$", question, re.I)
    if not m:
        return None

    aux = m.group(1).lower()
    rest = m.group(2).strip()

    # "Is this a train?" → "This is a train."
    m2 = re.match(r"^(this|that|these|those)\s+(.+)$", rest, re.I)
    if m2 and aux in {"is", "was"}:
        subj, pred = m2.group(1), m2.group(2)
        pos = f"{capitalize_first(subj)} is {pred}."
        if is_yes(answer):
            return pos
        if is_no(answer):
            return insert_not(pos)

    # "Is the boy wearing glasses?" → "The boy is wearing glasses."
    if aux in {"is", "are", "was", "were"}:
        subj, pred = subject_verb_from_is(rest)
        be = "are" if aux in {"are", "were"} else "is"
        pos = format_the_subject(subj, pred, be)
        if is_yes(answer):
            return pos
        if is_no(answer):
            return insert_not(pos)

    # "Does the pizza have pepperoni?" → "The pizza have pepperoni." (sade)
    if aux in {"does", "do", "did"}:
        subj, pred = subject_verb_from_is(rest)
        subj_clean = subj[4:] if subj.startswith("the ") else subj
        pos = f"The {subj_clean} {pred}."
        if is_yes(answer):
            return pos
        if is_no(answer):
            return insert_not(pos)

    return None


def rule_what_is(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What is the X?' → 'The X is {answer}.'"""
    m = re.match(r"^what is (?:the )?(.+?)(?: on .+| in .+| near .+)?$", question, re.I)
    if not m:
        return None
    subj = m.group(1).strip()
    if answer in YES | NO:
        return None
    return f"The {subj} is {answer}."


def rule_what_brand_sport(question: str, answer: str) -> Optional[str]:
    """Pattern: 'What brand/sport/... X?' → 'The {kind} X is {answer}.'"""
    m = re.match(r"^what (brand|sport|room|animal|vehicle|food|drink) (.+)$", question, re.I)
    if not m:
        return None
    kind, rest = m.group(1).lower(), m.group(2).strip()
    return f"The {kind} {rest} is {answer}."


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
