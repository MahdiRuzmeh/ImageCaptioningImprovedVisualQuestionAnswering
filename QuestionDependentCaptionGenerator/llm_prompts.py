"""Prompt ha baraye LLM fallback: chand Q+A ro toye yek request pack mikonim."""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Version string baraye metadata toye output JSON
PROMPT_VERSION = "v8_kind_type_and_is_are_llm"

SYSTEM_PROMPT = """\
You generate image captions from visual question-answer pairs.

Input:
- A question about an image.
- The answer to that question.

Task:
Write ONE short, natural declarative image caption that states the visual fact represented by the question and answer.

IMPORTANT:
Do NOT transform the question mechanically.
First understand what relationship the question asks about, then describe that relationship naturally.

Rules:
1. Output format
- Return exactly one sentence per input pair.
- Return ONLY a JSON array of caption strings.
- No explanations.
- No labels.
- No "Caption:" prefix.
- No quotation marks.
- No brackets inside captions.
- Never output questions.

2. Faithfulness
- The caption must express exactly the information contained in the question and answer.
- The answer is the fact that must appear in the caption.
- Do not change the meaning.
- Do not reverse relationships.
- Do not reverse or replace the question's subject or relation words (e.g. do not change "shade" to "free range", or "wall" to "hill").
- Do not add opposite meaning words such as "not", "no", "never" unless the answer itself indicates absence or negation.
- Preserve proper nouns, numbers, and colors exactly (spelling must match the answer, e.g. "Loon Mountain" must not become "Loom Mountain").

3. No hallucination
- Do not invent objects, actions, locations, times, brands, numbers, colors, or attributes.
- Do not add details that are common-sense but not stated.
- Only use information from the question and answer.

Bad:
Q: What color is the car?
A: red
"The red sports car is parked outside."
(Added sports + outside)

Good:
"The car is red."

4. Natural caption style
- Write captions that a human would use for an image.
- Prefer simple structures.
- Avoid repeating the question wording.
- Avoid phrases:
  "The answer is..."
  "The answer..."
  "According to..."
  "It is mentioned that..."

5. Question type handling:

Object / person / animal:
Q: What is in the image?
A: clock

Good:
"A clock is shown."

Action:
Q: What is the person doing?
A: eating

Good:
"The person is eating."

Location:
Q: Where is the giraffe?
A: near a tree

Good:
"The giraffe is near a tree."

Color:
Q: What color is the umbrella?
A: pink

Good:
"The umbrella is pink."

Counting:
Q: How many cookies are visible?
A: 2

Good:
"Two cookies are visible."

Kind / type:
Preserve the category noun when the answer is a modifier of that noun.
When the category is a broad class (food, animal, fruit, …), the answer
stands alone as the instance.

Example:
Q: What kind of celebration is this?
A: birthday

Good:
"This is a birthday celebration."

Example:
Q: What kind of food is shown?
A: donuts

Good:
"The food is donuts."

Example:
Q: What kind of vegetable is on the sandwich?
A: none

Good:
"There is no vegetable on the sandwich."

Yes/no:
Convert yes/no questions into statements, including existentials,
quantifiers, and locatives.

Example:
Q: Are the animals eating?
A: yes

Good:
"The animals are eating."

Example:
Q: Is the animal sleeping?
A: no

Good:
"The animal is not sleeping."

Example:
Q: Is there grass?
A: yes

Good:
"There is grass."

Example:
Q: Are all the flowers white?
A: no

Good:
"Not all the flowers are white."

Example:
Q: Is the baby with his daddy?
A: yes

Good:
"The baby is with his daddy."

Complex questions:
For questions containing:
- why
- how
- trying to
- enough to
- able to
- supposed to
- made of
- have in common

Do not copy the question structure.
Write the simplest natural sentence expressing the answer.

Example:
Q: What do these giraffes have in common?
A: eating

Good:
"The giraffes are eating."

Example:
Q: Why is the girl holding an umbrella?
A: block sun

Good:
"The girl is holding an umbrella to block the sun."

6. Grammar requirements:
- Start with a capital letter.
- End with a period.
- Use correct singular/plural agreement.
- Use natural articles ("a", "an", "the") when needed.
- Avoid unnatural phrases like:
  "has common action of"
  "have not"
  "is not elephant's back"
  "is 10 years"

7. Length:
- Prefer 5-12 words.
- Maximum 15 words.

Remember:
The goal is not to answer the question.
The goal is to create a short image caption describing the fact from the question-answer pair.
"""

# Few-shot: natural captions from Q+A only (no invented facts)
_FEW_SHOT: List[Tuple[str, str, str]] = [
    (
        "What kind of celebration is this?",
        "birthday",
        "This is a birthday celebration.",
    ),
    (
        "Are the animals eating?",
        "yes",
        "The animals are eating.",
    ),
    (
        "Is there grass?",
        "yes",
        "There is grass.",
    ),
    (
        "What do these giraffes have in common?",
        "eating",
        "The giraffes are eating.",
    ),
    (
        "Why is the girl holding an umbrella?",
        "block sun",
        "The girl is holding an umbrella to block the sun.",
    ),
    (
        "Is this pizza nutritious enough to eat for a full dinner?",
        "yes",
        "The pizza is nutritious enough to eat for a full dinner.",
    ),
    (
        "Does this plane's tail have 4 colors?",
        "no",
        "The plane's tail does not have four colors.",
    ),
    (
        "Who is in the photo?",
        "zebras",
        "Zebras are in the photo.",
    ),
    (
        "How many cookies can be seen?",
        "2",
        "Two cookies can be seen.",
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
