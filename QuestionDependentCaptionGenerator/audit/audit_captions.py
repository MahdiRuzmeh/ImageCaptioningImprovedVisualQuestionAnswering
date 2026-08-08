"""Audit a generated captions JSON for Comments6 / QC failure patterns.

Usage (from QuestionDependentCaptionGenerator/):

    python audit_captions.py outputs/v2_question_dependent_captions_train2014.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return counts of known systematic failure patterns."""
    neg = re.compile(
        r"\b(no|not|never|none|nobody|nothing|neither|cannot|can't|no one)\b",
        re.I,
    )
    yes = {"yes", "yeah", "yep", "true", "maybe"}

    the_there = sum(
        1 for r in rows if re.search(r"\bThe there\b", r.get("caption") or "")
    )
    the_the = sum(
        1 for r in rows if re.search(r"\bThe the\b", r.get("caption") or "")
    )
    made_of_is = sum(
        1
        for r in rows
        if re.search(
            r"\bmade of is\b|\bused for is\b|\bdesigned for is\b",
            (r.get("caption") or ""),
            re.I,
        )
    )
    empty = sum(1 for r in rows if not str(r.get("caption") or "").strip())
    needs = sum(1 for r in rows if r.get("rule") == "needs_llm")
    yes_neg = 0
    for r in rows:
        if str(r.get("rule")) != "llm_fallback":
            continue
        if str(r.get("answer") or "").strip().lower() not in yes:
            continue
        if neg.search(r.get("caption") or ""):
            yes_neg += 1

    rules = Counter(str(r.get("rule")) for r in rows)
    return {
        "num_samples": len(rows),
        "the_there": the_there,
        "the_the": the_the,
        "made_of_is_collapses": made_of_is,
        "empty_captions": empty,
        "needs_llm": needs,
        "llm_yes_to_neg_suspects": yes_neg,
        "rule_counts": dict(rules.most_common()),
        "has_answer_consensus": sum(
            1 for r in rows if "answer_consensus" in r
        ),
        "has_answer_confidence_legacy": sum(
            1 for r in rows if "answer_confidence" in r
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()
    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("annotations") or []
    info = data.get("info") or {}
    report = audit(rows)
    report["info_ocr_excluded"] = info.get("ocr_excluded_count")
    report["info_dropped_empty"] = info.get("dropped_empty_count")
    report["info_subjective_excluded"] = info.get("subjective_excluded_count")
    print(json.dumps(report, indent=2))

    bad = (
        report["the_there"]
        + report["the_the"]
        + report["made_of_is_collapses"]
        + report["empty_captions"]
        + report["needs_llm"]
    )
    if bad:
        print(
            f"\nAUDIT FAIL: {bad} residual rule/empty issues "
            "(llm yes→neg suspects are advisory).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("\nAUDIT OK: no The-there / made-of-is / empty / needs_llm leftovers.")


if __name__ == "__main__":
    main()
