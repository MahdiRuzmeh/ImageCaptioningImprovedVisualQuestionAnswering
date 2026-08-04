"""Prompt ha baraye LLM fallback: chand Q+A ro toye yek request pack mikonim."""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Version string baraye metadata toye output JSON
PROMPT_VERSION = "v6_no_question_echo_no_answer_phrase"

SYSTEM_PROMPT = """\
You are given visual questions and their answers.

Your task is to write ONE short, natural image caption per pair using ONLY the \
information contained in the question and answer.

Rules:
- Produce a caption, NOT a question or an answer.
- Return exactly one short declarative sentence per pair.
- Do not output a question. Never end the sentence with "?", and never repeat, \
quote, or rephrase the question itself anywhere in the caption — describe what \
is in the image instead. E.g. question "Where is the logo?" answer "nowhere" -> \
"There is no logo visible.", NOT "Where is there no logo? No logo present."
- Never write the phrases "the answer is" or "the answer" — weave the answer \
directly into a natural sentence about the image instead.
- Do not use brackets, labels, explanations, or quotation marks.
- Preserve the meaning of both the question and the answer.
- Do not invent objects, actions, locations, or attributes that are not implied \
by the question.
- Do not add extra details not present in the answer (no invented units, times \
of day, brands, counts, or qualifiers). E.g. answer "1:50" -> "It is 1:50.", \
NOT "It's 1:50 PM."
- Do not mention "the image", "the photo", or "the picture" unless they appear \
naturally in the question.
- Use natural English that a human would write.
- Keep each caption under 15 words.
- If the answer is "yes" or "no", convert the question into a natural affirmative \
or negative statement.
- If the answer is anything else (an object, person, animal, color, number, \
attribute, name, ...), the caption MUST be affirmative/positive — never add \
"no", "not", "never", or any other negation word when the answer itself isn't \
"no"/"none". E.g. question "Who made the clock?" answer "rolex" -> \
"Rolex made the clock.", NOT "No clock was made by Rolex."
- If the answer is an object, person, animal, color, number, or attribute, \
integrate it naturally into the sentence.
- Return ONLY a JSON array of caption strings (same order as the inputs). \
No other text, no labels, no "Caption:" prefixes.\
"""

# Few-shot: natural captions from Q+A only (no invented facts)
_FEW_SHOT: List[Tuple[str, str, str]] = [
    (
        "What is the man holding?",
        "bat",
        "The man is holding a bat.",
    ),
    (
        "What color are the dishes?",
        "pink and yellow",
        "The dishes are pink and yellow.",
    ),
    (
        "Are there numbers on the clock face?",
        "no",
        "There are no numbers on the clock face.",
    ),
    (
        "What is in front of the giraffes?",
        "tree",
        "There is a tree in front of the giraffes.",
    ),
    (
        "What color is the person on the elephant in the back wearing?",
        "red",
        "The person on the elephant in the back is wearing red.",
    ),
    (
        "Does this photo show train tracks?",
        "yes",
        "This photo shows train tracks.",
    ),
    (
        "Are these wings strong?",
        "yes",
        "These wings are strong.",
    ),
]


def build_user_prompt(pairs: Sequence[Tuple[str, str]]) -> str:
    """Az list (question, answer) yek user prompt pack-shode misaze.

    Uses few-shot Q/A → Caption examples, then asks for a JSON array of
    captions for the new pairs (same order).

    Args:
        pairs: list of (question, answer) — size = batch-size
    """
    lines: List[str] = [
        "You are given a question and its answer about an image.",
        "Write one short, natural image caption using only the information "
        "in the question and answer.",
        "",
        "Examples:",
        "",
    ]
    for q, a, cap in _FEW_SHOT:
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
        lines.append(f"Caption: {cap}")
        lines.append("")

    lines.append("Now write the captions for the pairs below.")
    lines.append(
        "Return ONLY a JSON array of sentence strings "
        "(no keys, no Caption: prefix, same order):"
    )
    lines.append("")
    for i, (q, a) in enumerate(pairs, start=1):
        lines.append(f"{i}. Q: {q}")
        lines.append(f"   A: {a}")
    lines.append("")
    lines.append(
        "Remember: return exactly one short declarative sentence per pair. "
        "Do not output a question, and do not repeat or rephrase the question "
        "text inside the caption. Never write \"the answer is\" or \"the "
        "answer\". Do not use brackets, labels, explanations, or quotation "
        "marks. Preserve the meaning of both the question and the answer."
    )
    lines.append("")
    lines.append("JSON array:")
    return "\n".join(lines)


def chat_messages(pairs: Sequence[Tuple[str, str]]) -> List[dict]:
    """Messages list baraye Ollama /api/chat misaze.

    Args:
        pairs: packed (question, answer) batch for one LLM call.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(pairs)},
    ]
