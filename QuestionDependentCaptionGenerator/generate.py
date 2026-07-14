"""CLI baraye generate kardan-e question-dependent captions az VQA v2.

Run az in folder:

    python generate.py --split train
    python generate.py --split val
    python generate.py --split train --max-items 1000   # smoke test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from caption_rules import generate_caption, mode_answer

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"


# ---------------------------------------------------------------------------
# Default paths — hamoon path hayi ke SimpleVQA estefade mikone
# ---------------------------------------------------------------------------

SPLIT_PATHS: Dict[str, Dict[str, Path]] = {
    "train": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_train2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_train2014_annotations.json",
        "output": DATASET_ROOT / "v2_question_dependent_captions_train2014.json",
    },
    "val": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_val2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_val2014_annotations.json",
        "output": DATASET_ROOT / "v2_question_dependent_captions_val2014.json",
    },
}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def load_vqa_pairs(
    questions_json: Path,
    annotations_json: Path,
    max_items: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Soal va javab haye VQA v2 ro load kon va baraye har sample caption besaz.

    Args:
        questions_json: path be v2_OpenEnded_*_questions.json
        annotations_json: path be v2_mscoco_*_annotations.json
        max_items: age set shode, faghat N sample aval ro process kon (smoke test)

    Returns:
        (rows, rule_counts) — rows list sample ha, rule_counts statistik rule ha.
    """
    with questions_json.open("r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with annotations_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    qmap = {int(x["question_id"]): x for x in questions}
    amap = {int(x["question_id"]): x for x in annotations}
    qids = sorted(set(qmap.keys()) & set(amap.keys()))
    if max_items is not None and max_items > 0:
        qids = qids[:max_items]

    rows: List[Dict[str, Any]] = []
    rule_counts: Counter = Counter()

    for qid in qids:
        q = qmap[qid]
        ann = amap[qid]
        answers = [x["answer"] for x in ann["answers"]]
        ans = mode_answer(answers)
        caption, rule = generate_caption(q["question"], ans)
        rule_counts[rule] += 1

        rows.append(
            {
                "question_id": qid,
                "image_id": int(q["image_id"]),
                "question": q["question"],
                "answer": ans,
                "caption": caption,
                "rule": rule,
            }
        )

    return rows, rule_counts


def write_output_json(
    output_path: Path,
    rows: List[Dict[str, Any]],
    rule_counts: Counter,
    questions_json: Path,
    annotations_json: Path,
) -> None:
    """Natije ro be JSON file save kon."""
    payload = {
        "info": {
            "description": "VQA v2 question-dependent captions (rule-based Q+A → statement)",
            "source_questions": str(questions_json),
            "source_annotations": str(annotations_json),
            "num_samples": len(rows),
            "rule_counts": dict(rule_counts),
        },
        "annotations": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_stats(rows: List[Dict[str, Any]], rule_counts: Counter, output_path: Path) -> None:
    """Statistik rule ha ro chap kon ta befahmim cheghadr fallback darim."""
    total = len(rows)
    print(f"Wrote {total} captions -> {output_path}")
    for rule, count in rule_counts.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {rule}: {count} ({pct:.1f}%)")

    print("\nSample captions:")
    for row in rows[:5]:
        print(f"  Q: {row['question']}")
        print(f"  A: {row['answer']}")
        print(f"  C: {row['caption']}  [{row['rule']}]")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon."""
    parser = argparse.ArgumentParser(
        description="Generate question-dependent captions from VQA v2 (rule-based)."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Kodom split VQA v2 (train ya val)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Override path be questions JSON",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Override path be annotations JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override path be output JSON",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Faghat N sample aval (baraye smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — caption ha ro generate kon va save kon."""
    args = parse_args()
    paths = SPLIT_PATHS[args.split]

    questions_json = Path(args.questions) if args.questions else paths["questions"]
    annotations_json = Path(args.annotations) if args.annotations else paths["annotations"]
    output_path = Path(args.output) if args.output else paths["output"]

    if not questions_json.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_json}")
    if not annotations_json.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_json}")

    rows, rule_counts = load_vqa_pairs(
        questions_json,
        annotations_json,
        max_items=args.max_items,
    )
    write_output_json(
        output_path,
        rows,
        rule_counts,
        questions_json,
        annotations_json,
    )
    print_stats(rows, rule_counts, output_path)


if __name__ == "__main__":
    main()
