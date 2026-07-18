"""CLI baraye generate kardan-e question-dependent captions az VQA v2.

Run az in folder:

    python generate.py --split train
    python generate.py --split val
    python generate.py --split train --max-items 1000   # smoke test
    python generate.py --split val --llm --batch-size 10 \\
        --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 50

Output default: ./outputs/v2_question_dependent_captions_{train,val}2014.json

Resume (Ctrl+C bad):
    hamoon command ro dobare bezan — az checkpoint edame mide.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from caption_rules import generate_caption, mode_answer
from llm_client import OllamaClient, run_batches_concurrent
from llm_prompts import PROMPT_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent
# VQA raw data az ../dataset; caption JSON inja save mishe
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Default paths — input az dataset/, output dakhele in folder
# ---------------------------------------------------------------------------

SPLIT_PATHS: Dict[str, Dict[str, Path]] = {
    "train": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_train2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_train2014_annotations.json",
        "output": OUTPUT_ROOT / "v2_question_dependent_captions_train2014.json",
    },
    "val": {
        "questions": DATASET_ROOT / "v2_OpenEnded_mscoco_val2014_questions.json",
        "annotations": DATASET_ROOT / "v2_mscoco_val2014_annotations.json",
        "output": OUTPUT_ROOT / "v2_question_dependent_captions_val2014.json",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chunked(
    items: Sequence[Any],
    size: int,
) -> List[List[Any]]:
    """List ro be batch haye size N chop mikone."""
    n = max(1, int(size))
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def recount_rules(rows: List[Dict[str, Any]]) -> Counter:
    """Az rows, Counter rule ha ro dobare hesab kon."""
    return Counter(str(r.get("rule", "unknown")) for r in rows)


def load_output_payload(output_path: Path) -> Optional[Dict[str, Any]]:
    """Output JSON ro load kon; age corrupt bashe None."""
    if not output_path.is_file():
        return None
    try:
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_existing_llm_map(output_path: Path) -> Dict[int, Dict[str, Any]]:
    """Age output ghablan hast, llm_fallback ha ro baraye resume load kon.

    Returns:
        map question_id -> row (faghat rule=llm_fallback)
    """
    data = load_output_payload(output_path)
    if not data:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in data.get("annotations") or []:
        if str(row.get("rule")) != "llm_fallback":
            continue
        try:
            qid = int(row["question_id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[qid] = row
    return out


def count_vqa_overlap(
    questions_json: Path,
    annotations_json: Path,
    max_items: Optional[int] = None,
) -> int:
    """Tedad sample moshtarak questions∩annotations (ba max_items)."""
    with questions_json.open("r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with annotations_json.open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    qids = set(int(x["question_id"]) for x in questions) & set(
        int(x["question_id"]) for x in annotations
    )
    n = len(qids)
    if max_items is not None and max_items > 0:
        n = min(n, max_items)
    return n


def try_load_checkpoint_rows(
    output_path: Path,
    expected_n: int,
) -> Optional[List[Dict[str, Any]]]:
    """Age checkpoint size == expected, rows ro az file bardar (resume sari)."""
    data = load_output_payload(output_path)
    if not data:
        return None
    rows = data.get("annotations")
    if not isinstance(rows, list) or len(rows) != expected_n:
        return None
    # Minimal field check
    for row in rows[:3]:
        if "question_id" not in row or "rule" not in row:
            return None
    return list(rows)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def load_vqa_pairs(
    questions_json: Path,
    annotations_json: Path,
    max_items: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Soal va javab haye VQA v2 ro load kon va ba rule caption besaz.

    Args:
        questions_json: path be v2_OpenEnded_*_questions.json
        annotations_json: path be v2_mscoco_*_annotations.json
        max_items: age set shode, faghat N sample aval (smoke test)

    Returns:
        (rows, rule_counts) — rows + statistik rule ha.
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


def merge_llm_resume(
    rows: List[Dict[str, Any]],
    resume_map: Dict[int, Dict[str, Any]],
) -> int:
    """llm_fallback haye save-shode ro roye rows restore kon.

    Returns:
        tedad row ke restore shod.
    """
    restored = 0
    for row in rows:
        if row["rule"] != "fallback":
            continue
        prev = resume_map.get(int(row["question_id"]))
        if prev is None:
            continue
        cap = str(prev.get("caption") or "").strip()
        if not cap:
            continue
        row["caption"] = cap
        row["rule"] = "llm_fallback"
        restored += 1
    return restored


def apply_llm_fallbacks(
    rows: List[Dict[str, Any]],
    *,
    client: OllamaClient,
    batch_size: int,
    workers: int,
    checkpoint_every: int,
    output_path: Path,
    questions_json: Path,
    annotations_json: Path,
    llm_meta: Dict[str, Any],
    resume_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Counter:
    """Row haye rule=fallback ro ba packed LLM caption update mikone.

    Resume: age question_id toye resume_map bashe, LLM call nemikone.
    Checkpoint: har N batch (default 1), JSON ro atomic save mikone.
    """
    resume_map = resume_map or {}
    restored = merge_llm_resume(rows, resume_map)
    if restored:
        print(f"Resume: {restored} llm_fallback az checkpoint restore shod")

    pending_idx: List[int] = [
        i for i, r in enumerate(rows) if r["rule"] == "fallback"
    ]
    already = sum(1 for r in rows if r["rule"] == "llm_fallback")
    print(
        f"LLM fallback: {len(pending_idx)} pending, {already} already done "
        f"(batch-size={batch_size}, workers={workers}, "
        f"checkpoint-every={checkpoint_every})"
    )

    if not pending_idx:
        print("Hichi pending nist — LLM pass skip.")
        return recount_rules(rows)

    pairs = [(rows[i]["question"], rows[i]["answer"]) for i in pending_idx]
    indexed = list(zip(pending_idx, pairs))
    batches_idx = chunked(indexed, batch_size)
    batches_pairs: List[List[Tuple[str, str]]] = [
        [(q, a) for _, (q, a) in batch] for batch in batches_idx
    ]

    done_batches = 0
    total_batches = len(batches_pairs)

    def _on_batch(batch_i: int, caps: List[Optional[str]]) -> None:
        nonlocal done_batches
        for j, cap in enumerate(caps):
            row_i = batches_idx[batch_i][j][0]
            if cap is None:
                continue
            rows[row_i]["caption"] = cap
            rows[row_i]["rule"] = "llm_fallback"
        done_batches += 1
        still = sum(1 for r in rows if r["rule"] == "fallback")
        print(
            f"  LLM batch {batch_i + 1}/{total_batches} done "
            f"({done_batches}/{total_batches} completed, "
            f"{still} fallback left)"
        )
        if checkpoint_every > 0 and done_batches % checkpoint_every == 0:
            counts = recount_rules(rows)
            write_output_json(
                output_path,
                rows,
                counts,
                questions_json,
                annotations_json,
                llm_meta=llm_meta,
            )
            print(f"  checkpoint saved -> {output_path}")

    run_batches_concurrent(
        client,
        batches_pairs,
        workers=workers,
        on_batch_done=_on_batch,
    )
    return recount_rules(rows)


def write_output_json(
    output_path: Path,
    rows: List[Dict[str, Any]],
    rule_counts: Counter,
    questions_json: Path,
    annotations_json: Path,
    llm_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Natije ro atomic be JSON file save kon (crash-safe)."""
    info: Dict[str, Any] = {
        "description": "VQA v2 question-dependent captions (rule-based Q+A → statement)",
        "source_questions": str(questions_json),
        "source_annotations": str(annotations_json),
        "num_samples": len(rows),
        "rule_counts": dict(rule_counts),
    }
    if llm_meta:
        info["description"] = (
            "VQA v2 question-dependent captions (rules + optional LLM fallback)"
        )
        info["llm"] = llm_meta

    payload = {
        "info": info,
        "annotations": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp file bad rename — file nime-kar corrupt nashe
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_path.stem + "_",
        suffix=".tmp.json",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, output_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def print_stats(
    rows: List[Dict[str, Any]],
    rule_counts: Counter,
    output_path: Path,
) -> None:
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

    llm_rows = [r for r in rows if r["rule"] == "llm_fallback"][:3]
    if llm_rows:
        print("Sample llm_fallback captions:")
        for row in llm_rows:
            print(f"  Q: {row['question']}")
            print(f"  A: {row['answer']}")
            print(f"  C: {row['caption']}  [{row['rule']}]")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon (rule + optional LLM)."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate question-dependent captions from VQA v2 "
            "(rules + optional LLM). Ctrl+C safe: checkpoint + resume."
        )
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
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Baraye rule=fallback az Ollama/Mistral caption begir",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Chand Q+A toye yek LLM request (default 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Esm model Ollama (default mistral)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API request be hamoon Ollama (default 1, 8GB safe)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Base URL Ollama",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Har chand LLM batch output JSON save beshe "
            "(1=har batch, 50 ya 100=kamtar I/O; default 1)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Checkpoint ghabli ro ignore kon (az aval LLM)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — rule caption + optional LLM fallback + resume."""
    args = parse_args()
    paths = SPLIT_PATHS[args.split]

    questions_json = Path(args.questions) if args.questions else paths["questions"]
    annotations_json = (
        Path(args.annotations) if args.annotations else paths["annotations"]
    )
    output_path = Path(args.output) if args.output else paths["output"]

    if not questions_json.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_json}")
    if not annotations_json.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_json}")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    expected_n = count_vqa_overlap(
        questions_json, annotations_json, max_items=args.max_items
    )

    rows: Optional[List[Dict[str, Any]]] = None
    rule_counts: Counter

    # Resume sari: age output size match bashe, dobare rule run nakon
    if args.llm and not args.no_resume:
        rows = try_load_checkpoint_rows(output_path, expected_n)
        if rows is not None:
            rule_counts = recount_rules(rows)
            print(
                f"Loaded checkpoint ({len(rows)} rows) az {output_path} "
                f"— rules skip, LLM az ja-monde edame."
            )

    if rows is None:
        rows, rule_counts = load_vqa_pairs(
            questions_json,
            annotations_json,
            max_items=args.max_items,
        )

    llm_meta: Optional[Dict[str, Any]] = None
    if args.llm:
        llm_meta = {
            "model": args.model,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "host": args.ollama_host,
            "prompt_version": PROMPT_VERSION,
        }
        # Merge llm_fallback az file (age rules-rebuild shode bashe).
        # Age rows mostaghim az checkpoint load shode, merge no-op safe hast.
        if args.no_resume:
            print("--no-resume: checkpoint llm_fallback merge nemishe")
            resume_map: Dict[int, Dict[str, Any]] = {}
        else:
            resume_map = load_existing_llm_map(output_path)

        client = OllamaClient(host=args.ollama_host, model=args.model)
        try:
            rule_counts = apply_llm_fallbacks(
                rows,
                client=client,
                batch_size=args.batch_size,
                workers=args.workers,
                checkpoint_every=args.checkpoint_every,
                output_path=output_path,
                questions_json=questions_json,
                annotations_json=annotations_json,
                llm_meta=llm_meta,
                resume_map=resume_map,
            )
        except KeyboardInterrupt:
            # Ctrl+C: last state ro save kon ta resume beshe
            print("\nInterrupted — saving checkpoint...")
            rule_counts = recount_rules(rows)
            write_output_json(
                output_path,
                rows,
                rule_counts,
                questions_json,
                annotations_json,
                llm_meta=llm_meta,
            )
            still = sum(1 for r in rows if r["rule"] == "fallback")
            print(
                f"Checkpoint saved -> {output_path} "
                f"({still} fallback left). Dobare hamoon command ro bezan."
            )
            raise SystemExit(130) from None

    write_output_json(
        output_path,
        rows,
        rule_counts,
        questions_json,
        annotations_json,
        llm_meta=llm_meta,
    )
    print_stats(rows, rule_counts, output_path)


if __name__ == "__main__":
    main()
