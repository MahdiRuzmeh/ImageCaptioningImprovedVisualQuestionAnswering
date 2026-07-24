"""Prompt ha baraye LLM fallback: chand Q+A ro toye yek request pack mikonim."""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Version string baraye metadata toye output JSON
PROMPT_VERSION = "v3_natural_caption_qa_only"

SYSTEM_PROMPT = """\
You are given visual questions and their answers.

Your task is to write ONE short, natural image caption per pair using ONLY the \
information contained in the question and answer.

Rules:
- Produce a caption, NOT a question or an answer.
- Do not invent objects, actions, locations, or attributes that are not implied \
by the question.
- Do not mention "the image", "the photo", or "the picture" unless they appear \
naturally in the question.
- Use natural English that a human would write.
- Prefer simple declarative sentences.
- Keep each caption under 15 words.
- If the answer is "yes" or "no", convert the question into a natural affirmative \
or negative statement.
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
