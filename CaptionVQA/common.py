"""Shared helpers: config, tokenizer, VQA v2 answer normalization and accuracy.

Finglish: answer normalization + accuracy = nashkhe-ye kootah-e evaluator-e
rasmi-ye VQA v2 (punctuation, number words → digit, article ha hazf, leave-one-out
roye 10 javab). Hame model ha (A/B/C va SimpleVQA) ba hamin tabe score mishan.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT.parent
CAPTIONER_ROOT = SRC_ROOT / "QuestionDependentImageCaptioner"

TOKEN_RE = re.compile(r"[a-z0-9']+")  # hamoon tokenizer-e captioner va SimpleVQA


def tok(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_config(path: str) -> Dict[str, Any]:
    """YAML load; path haye relative nesbat be CaptionVQA/ resolve mishan."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in list(cfg.items()):
        if isinstance(v, str) and (k.endswith("_json") or k.endswith("_dir") or k.endswith("_ckpt")):
            p = Path(v)
            cfg[k] = str(p if p.is_absolute() else (ROOT / p).resolve())
    return cfg


def add_captioner_to_path() -> None:
    if str(CAPTIONER_ROOT) not in sys.path:
        sys.path.insert(0, str(CAPTIONER_ROOT))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------------------
# VQA v2 answer normalization (official evaluator, condensed)
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't",
    "cant": "can't", "wont": "won't", "didnt": "didn't", "im": "i'm",
    "ive": "i've", "youre": "you're", "thats": "that's", "whats": "what's",
    "wasnt": "wasn't", "werent": "weren't", "hasnt": "hasn't", "havent": "haven't",
    "shouldnt": "shouldn't", "couldnt": "couldn't", "wouldnt": "wouldn't",
    "theyre": "they're", "theres": "there's", "lets": "let's",
}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_answer(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    out = ans
    for p in _PUNCT:
        if (p + " " in ans or " " + p in ans) or re.search(_COMMA_STRIP, ans) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    words = []
    for w in out.split():
        w = _NUMBERS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_score(pred: str, gt_answers: List[str]) -> float:
    """Official VQA v2 accuracy: mean over 10 leave-one-out subsets of min(1, match/3)."""
    pred = normalize_answer(pred)
    gts = [normalize_answer(a) for a in gt_answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(1 for g in others if g == pred) / 3.0))
    return sum(accs) / max(1, len(accs))


def soft_targets(gt_answers: List[str]) -> Dict[str, float]:
    """Training target per distinct answer (same formula as vqa_score)."""
    gts = [normalize_answer(a) for a in gt_answers]
    return {a: vqa_score(a, gt_answers) for a in set(gts)}
