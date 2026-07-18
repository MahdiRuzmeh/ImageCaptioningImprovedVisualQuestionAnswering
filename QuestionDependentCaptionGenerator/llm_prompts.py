"""Prompt ha baraye LLM fallback: chand Q+A ro toye yek request pack mikonim."""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Version string baraye metadata toye output JSON
PROMPT_VERSION = "v2_natural_caption"

SYSTEM_PROMPT = """\
You rewrite each VQA question+answer into ONE short, natural English image caption.
Write like a human describing a photo — fluent and grammatical, not a template.
Use the answer as the main fact; keep the useful subject/details from the question.
Do NOT invent facts beyond the answer. Do NOT start with "Caption:" or labels.
For yes/no: state the fact positively or with natural negation (not "Q — no").
Keep each sentence under 20 words.
Reply with ONLY a JSON array of strings, same order as the inputs. No other text.\
"""

# Few-shot: natural captions (no "Caption:" label)
_FEW_SHOT: List[Tuple[str, str, str]] = [
    (
        "What do these giraffes have in common?",
        "eating",
        "The giraffes are both eating.",
    ),
    (
        "Is this a motorcycle or bike?",
        "motorcycle",
        "This is a motorcycle.",
    ),
    (
        "What toppings are on the pizza?",
        "cheese and pepperoni",
        "The pizza is topped with cheese and pepperoni.",
    ),
    (
        "What is written on the boat?",
        "sea breeze",
        "The boat has \"sea breeze\" written on it.",
    ),
    (
        "How big is the plane?",
        "large",
        "It is a large plane.",
    ),
    (
        "Has the wall been painted recently?",
        "no",
        "The wall does not look freshly painted.",
    ),
    (
        "Could this photo be from a zoo?",
        "yes",
        "This photo could be from a zoo.",
    ),
]


def build_user_prompt(pairs: Sequence[Tuple[str, str]]) -> str:
    """Az list (question, answer) yek user prompt pack-shode misaze.

    Args:
        pairs: list of (question, answer) — size = batch-size
    """
    lines: List[str] = [
        "Examples of natural captions (do not copy these; do not print labels):",
    ]
    for i, (q, a, cap) in enumerate(_FEW_SHOT, start=1):
        lines.append(f"{i}. Question: {q}")
        lines.append(f"   Answer: {a}")
        lines.append(f"   -> {cap}")

    lines.append("")
    lines.append(
        "Convert the pairs below into natural captions. "
        "Return ONLY a JSON array of sentence strings (no keys, no Caption: prefix):"
    )
    for i, (q, a) in enumerate(pairs, start=1):
        lines.append(f"{i}. Question: {q}")
        lines.append(f"   Answer: {a}")
    lines.append("")
    lines.append("JSON array:")
    return "\n".join(lines)


def chat_messages(pairs: Sequence[Tuple[str, str]]) -> List[dict]:
    """Messages list baraye Ollama /api/chat misaze."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(pairs)},
    ]
