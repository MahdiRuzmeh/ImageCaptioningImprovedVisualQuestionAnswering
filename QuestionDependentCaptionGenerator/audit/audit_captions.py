"""Audit a generated captions JSON for known QC failure patterns.

Covers Comments6 (template collapses, leftovers) and Comments8: per-row
``visual_filter_source`` provenance, ``validation_flags``, and the new
``info`` counters, so a re-run can be reported with one command.

Usage (from QuestionDependentCaptionGenerator/):

    python audit/audit_captions.py outputs/vqa_v2_question_dependent_captions_train2014.json
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

    # Rules deleted after Comments8 — any row still carrying them means the
    # file predates the change.
    retired_rules = sum(
        1 for r in rows if str(r.get("rule")) in {"what_is", "yesno_modal_have"}
    )

    rules = Counter(str(r.get("rule")) for r in rows)
    sources = Counter(
        str(r.get("visual_filter_source") or "(none)") for r in rows
    )
    flags: Counter = Counter()
    flagged_rows = 0
    for r in rows:
        row_flags = r.get("validation_flags") or []
        if row_flags:
            flagged_rows += 1
            flags.update(str(f) for f in row_flags)

    return {
        "num_samples": len(rows),
        "the_there": the_there,
        "the_the": the_the,
        "made_of_is_collapses": made_of_is,
        "empty_captions": empty,
        "needs_llm": needs,
        "retired_rule_rows": retired_rules,
        "llm_yes_to_neg_suspects": yes_neg,
        "rule_counts": dict(rules.most_common()),
        "visual_filter_source_counts": dict(sources.most_common()),
        "rows_with_validation_flags": flagged_rows,
        "validation_flag_counts": dict(flags.most_common()),
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
    report["info_duplicate_count"] = info.get("duplicate_count")
    report["info_directly_visual"] = info.get("directly_visual_count")
    report["info_not_directly_visual"] = info.get(
        "not_directly_visual_count",
        info.get("subjective_excluded_count"),
    )
    report["info_validation_retry"] = info.get("validation_retry_count")
    report["info_validation_failure"] = info.get("validation_failure_count")
    report["info_validation_flagged"] = info.get("validation_flagged_count")
    report["info_rule_validation_rejects"] = info.get(
        "rule_validation_reject_count"
    )
    report["info_input_count"] = info.get("input_count")
    report["info_low_consensus_excluded"] = info.get(
        "low_consensus_excluded_count"
    )
    report["info_min_consensus"] = info.get("min_consensus")
    classifier = info.get("question_classifier") or {}
    report["info_classifier_prompt_version"] = classifier.get("prompt_version")
    report["info_fast_path_enabled"] = classifier.get("fast_path_enabled")
    report["info_classifier_label_counts"] = classifier.get("label_counts")

    # Accounting identity the professor checks: every input row is either
    # filtered out, dropped, or present in the file.
    accounted = sum(
        int(info.get(k) or 0)
        for k in (
            "ocr_excluded_count",
            "low_consensus_excluded_count",
            "duplicate_count",
            "not_directly_visual_count",
            "dropped_empty_count",
            "validation_failure_count",
        )
    ) + len(rows)
    report["accounting_input_vs_accounted"] = {
        "input_count": info.get("input_count"),
        "accounted": accounted,
        "difference": (int(info.get("input_count") or 0) - accounted),
    }
    print(json.dumps(report, indent=2))

    bad = (
        report["the_there"]
        + report["the_the"]
        + report["made_of_is_collapses"]
        + report["empty_captions"]
        + report["needs_llm"]
        + report["retired_rule_rows"]
    )
    if bad:
        print(
            f"\nAUDIT FAIL: {bad} residual rule/empty issues "
            "(llm yes→neg suspects and validation_flags are advisory).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(
        "\nAUDIT OK: no The-there / made-of-is / empty / needs_llm / "
        "retired-rule leftovers."
    )


if __name__ == "__main__":
    main()
