"""Enrich question-dependent captions with visual context from COCO human captions.

The base QD caption only restates (question, answer) as a sentence, so it carries
no information beyond the VQA answer. This step gives the LLM the image's 5 COCO
human captions as well and asks for one sentence that keeps the base fact and adds
one or two question-relevant visual details taken from those captions.

Every output is checked deterministically:

- answer kept: answer words / numbers present, and the same negation polarity as
  the base caption
- grounded: every content word is in the question, answer, base caption or COCO
  captions (no invented objects)
- no conflicting number for counting answers

If both attempts fail, the base caption is kept (``enrich_status=base_fallback``).

Usage (pilot)::

    python enrich_captions.py --start 0 --max-items 300 --out outputs/enriched_pilot.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from validation.checks import has_sentential_negation
from validation.tokens import content_words, numeric_equivalents, stem, words

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
DEFAULT_QD = SRC / "dataset" / "vqa_v2_question_dependent_captions_train2014.json"
# src/dataset/captions_train2014.json starts with a stray line → use the original copy
DEFAULT_COCO = SRC.parents[2] /"dataset" / "MSCOCO Captions" / "2014" / "captions_train2014.json"

PROMPT_VERSION = "enrich_v1_coco_context"

SYSTEM_PROMPT = """\
You enrich a short image caption with visual details.

Input:
- QUESTION: a question about an image.
- ANSWER: its correct answer.
- BASE: a caption that states the answer.
- DESCRIPTIONS: five human descriptions of the same image.

Write ONE natural sentence (8 to 25 words) that:
1. Keeps the fact of BASE exactly: the same answer words, numbers and colors, and the same negation. If BASE says "not" or "no", your sentence must say it too.
2. Adds one or two visual details from DESCRIPTIONS about the question's subject or what is directly around it (objects, place, activity, appearance).
3. Uses only facts found in BASE, QUESTION or DESCRIPTIONS. Never invent objects, colors, numbers or actions.
4. If a description disagrees with ANSWER, trust ANSWER and leave that detail out.
5. Never mention the image, the picture, the photo or the descriptions.

Return only the sentence, nothing else."""

FEW_SHOTS: Sequence[Tuple[dict, str]] = (
    (
        {
            "question": "What color is the umbrella?",
            "answer": "red",
            "base": "The umbrella is red.",
            "coco": [
                "A woman walking down a rainy street holding an umbrella.",
                "A person with a red umbrella on a wet sidewalk.",
                "A lady walks in the rain under her umbrella.",
                "Someone carrying an umbrella on a city street.",
                "A woman in a coat holds an umbrella on a rainy day.",
            ],
        },
        "A woman holds a red umbrella while walking down a wet, rainy street.",
    ),
    (
        {
            "question": "How many dogs are there?",
            "answer": "2",
            "base": "There are two dogs.",
            "coco": [
                "Two dogs lying on a couch together.",
                "A pair of dogs resting on a brown sofa.",
                "Two puppies sleeping on a couch.",
                "Dogs napping on the living room couch.",
                "Two small dogs curled up on a sofa.",
            ],
        },
        "Two dogs are lying together on a brown couch.",
    ),
    (
        {
            "question": "Is the man wearing a hat?",
            "answer": "no",
            "base": "The man is not wearing a hat.",
            "coco": [
                "A man riding a wave on top of a surfboard.",
                "A surfer in a black wetsuit riding a wave.",
                "A person surfing on a large wave in the ocean.",
                "A man on a surfboard in the water.",
                "Surfer catching a wave at the beach.",
            ],
        },
        "The man in a black wetsuit riding a wave on a surfboard is not wearing a hat.",
    ),
)

_YES_NO = {"yes", "no"}
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "dozen",
}
# Words the model may add without a source (glue, not visual content).
_FREE_WORDS = {
    "while", "near", "next", "beside", "behind", "front", "under", "above",
    "inside", "outside", "around", "along", "together", "another", "someone",
    "something", "person", "people", "where", "which", "sitting",
    "standing", "being", "has", "have", "large", "small", "big", "little",
    "sit", "sits", "stand", "stands",
    "but", "among", "amid", "various", "variety", "array", "assortment",
    "surrounded", "surrounding", "arranged", "alongside", "nearby", "alone",
    "present", "other", "others", "one", "also", "located", "seems", "seem",
}
BASE_KEEP_MIN = 0.6


def user_prompt(item: dict) -> str:
    desc = "\n".join(f"- {c.strip()}" for c in item["coco"])
    return (
        f"QUESTION: {item['question']}\nANSWER: {item['answer']}\n"
        f"BASE: {item['base']}\nDESCRIPTIONS:\n{desc}"
    )


def chat_messages(item: dict) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot, out in FEW_SHOTS:
        msgs.append({"role": "user", "content": user_prompt(shot)})
        msgs.append({"role": "assistant", "content": out})
    msgs.append({"role": "user", "content": user_prompt(item)})
    return msgs


def ollama_chat(host: str, model: str, messages: List[dict], temperature: float,
                timeout_s: float = 120.0) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 2048, "num_predict": 64},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return str((raw.get("message") or {}).get("content") or "")


def clean_sentence(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = text.splitlines()[0].strip() if text else ""
    text = re.sub(r"^(caption|sentence|output)\s*:\s*", "", text, flags=re.I)
    text = text.strip('"').strip()
    if text and text[-1] not in ".!":
        text += "."
    return text


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _number_tokens(text: str) -> set:
    toks = set(re.findall(r"\b\d+\b", text.lower()))
    toks |= {w for w in words(text) if w in _NUMBER_WORDS}
    out = set()
    for t in toks:
        out |= numeric_equivalents(t)
    return out - {"a", "an"}


def answer_kept(item: dict, caption: str) -> Optional[str]:
    """None if the caption still states the answer, else a reject reason."""
    ans = item["answer"].strip().lower()
    base = item["base"]
    if ans in _YES_NO:
        if has_sentential_negation(caption) != has_sentential_negation(base):
            return "polarity_changed"
        return None
    if has_sentential_negation(caption) and not has_sentential_negation(base):
        return "negation_added"
    cap_words = set(words(caption))
    cap_stems = {stem(w) for w in cap_words}
    cap_nums = _number_tokens(caption)
    for tok in re.findall(r"[a-z0-9']+", ans):
        if tok in {"a", "an", "the", "of", "and", "or", "in", "on"}:
            continue
        if tok.isdigit() or tok in _NUMBER_WORDS:
            if not (numeric_equivalents(tok) & (cap_nums | cap_words)):
                return f"answer_missing:{tok}"
        elif stem(tok) not in cap_stems and tok not in cap_words:
            return f"answer_missing:{tok}"
    return None


def number_conflict(item: dict, caption: str) -> Optional[str]:
    """Counting answers: the caption must not state a different number."""
    ans = item["answer"].strip().lower()
    if not (ans.isdigit() or ans in _NUMBER_WORDS):
        return None
    allowed = numeric_equivalents(ans) | _number_tokens(item["base"]) | _number_tokens(item["question"])
    extra = _number_tokens(caption) - allowed
    # "one" also means "a"; ignore it unless the answer is a count > 1 and "one" is the only number
    extra -= {"1", "one"}
    return f"number_conflict:{sorted(extra)[0]}" if extra else None


def grounding(item: dict, caption: str) -> Tuple[List[str], List[str]]:
    """(added content words, ungrounded content words)."""
    base_side = content_words(item["question"]) | content_words(item["answer"]) | content_words(item["base"])
    coco_side = set()
    for c in item["coco"]:
        coco_side |= content_words(c)
    free = {stem(w) for w in _FREE_WORDS}
    cap = content_words(caption)
    added = sorted(cap - base_side - free)
    ungrounded = sorted(w for w in added if w not in coco_side)
    return added, ungrounded


def check(item: dict, caption: str) -> Tuple[List[str], List[str]]:
    """(reject reasons, added words)."""
    reasons: List[str] = []
    n = len(caption.split())
    if n < 4 or n > 30:
        reasons.append(f"length:{n}")
    meta = set(re.findall(r"\b(?:image|picture|photo|description)s?\b", caption, re.I))
    if "?" in caption or {m.lower() for m in meta} - set(words(item["base"] + " " + item["question"])):
        reasons.append("format")
    for r in (answer_kept(item, caption), number_conflict(item, caption)):
        if r:
            reasons.append(r)
    # The question's subject/relation must survive (e.g. "in front of the giraffe")
    base = content_words(item["base"])
    if base and len(base & content_words(caption)) / len(base) < BASE_KEEP_MIN:
        reasons.append("base_fact_lost")
    added, ungrounded = grounding(item, caption)
    if ungrounded:
        reasons.append("ungrounded:" + ",".join(ungrounded[:4]))
    return reasons, added


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def enrich_one(item: dict, host: str, model: str) -> dict:
    attempts = []
    for temp in (0.0, 0.7):
        try:
            raw = ollama_chat(host, model, chat_messages(item), temp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            attempts.append({"caption": "", "reasons": [f"llm_error:{exc}"]})
            continue
        cap = clean_sentence(raw)
        reasons, added = check(item, cap)
        attempts.append({"caption": cap, "reasons": reasons})
        if not reasons:
            status = "enriched" if added else "base_no_detail"
            return {"caption": cap if added else item["base"], "status": status,
                    "added": added, "attempts": attempts}
    return {"caption": item["base"], "status": "base_fallback", "added": [], "attempts": attempts}


def load_coco(path: Path) -> Dict[int, List[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    caps: Dict[int, List[str]] = defaultdict(list)
    for a in data["annotations"]:
        caps[int(a["image_id"])].append(a["caption"].strip())
    return caps


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_QD)
    ap.add_argument("--coco", type=Path, default=DEFAULT_COCO)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M")
    ap.add_argument("--llm-parallel", type=int, default=4,
                    help="Concurrent Ollama requests (set OLLAMA_NUM_PARALLEL >= this on the server)")
    args = ap.parse_args(argv)

    with open(args.input, encoding="utf-8") as f:
        qd = json.load(f)
    rows = qd["annotations"][args.start:]
    if args.max_items is not None:
        rows = rows[: args.max_items]
    coco = load_coco(args.coco)
    print(f"[enrich] {len(rows)} rows from {args.input.name} (start {args.start}), "
          f"COCO captions for {len(coco)} images", flush=True)

    # Resume: one JSON line per finished row
    progress = args.out.with_suffix(".progress.jsonl")
    done: Dict[int, dict] = {}
    if progress.exists():
        with open(progress, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["question_id"]] = rec
        print(f"[enrich] resume: {len(done)} rows already done", flush=True)

    todo = [r for r in rows if r["question_id"] not in done]
    t0 = time.time()

    def job(row: dict) -> dict:
        item = {"question": row["question"], "answer": row["answer"],
                "base": row["caption"], "coco": coco.get(int(row["image_id"]), [])[:5]}
        if not item["coco"]:
            res = {"caption": row["caption"], "status": "no_coco", "added": [], "attempts": []}
        else:
            res = enrich_one(item, args.host, args.model)
        return {"question_id": row["question_id"], **res}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(progress, "a", encoding="utf-8") as pf, ThreadPoolExecutor(args.llm_parallel) as pool:
        for i, rec in enumerate(pool.map(job, todo), 1):
            done[rec["question_id"]] = rec
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"[enrich] {i}/{len(todo)}  {rate:.2f} rows/s", flush=True)

    out_rows = []
    for r in rows:
        rec = done[r["question_id"]]
        new = dict(r)
        new["caption_base"] = r["caption"]
        new["caption"] = rec["caption"]
        new["enrich_status"] = rec["status"]
        new["enrich_added"] = rec["added"]
        if rec["status"] == "base_fallback":
            new["enrich_rejects"] = [a["reasons"] for a in rec["attempts"]]
        out_rows.append(new)

    status = Counter(r["enrich_status"] for r in out_rows)
    info = dict(qd.get("info", {}))
    info["enrich"] = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "coco_captions": str(args.coco),
        "source": str(args.input),
        "start": args.start,
        "num_rows": len(out_rows),
        "status_counts": dict(status),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"info": info, "annotations": out_rows}, f, ensure_ascii=False, indent=1)
    print(f"[enrich] wrote {args.out}  status: {dict(status)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
