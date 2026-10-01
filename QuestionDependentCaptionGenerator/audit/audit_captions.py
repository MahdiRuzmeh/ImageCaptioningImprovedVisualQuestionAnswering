"""Score GoldAuditor caption labels with the production caption validator.

Uses the same ``score_rows_keep_all`` / fast+LLM path as ``generate.py``
(``validate_rows`` logic) so gold tuning matches production behavior.

Usage (from QuestionDependentCaptionGenerator/):

    python audit/audit_captions.py
    python audit/audit_captions.py --llm --batch-size 10
    python audit/audit_captions.py audit/GoldAuditor/caption_audit_manual.json --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from llm_client import OllamaClient  # noqa: E402
from validation import ValidationConfig, score_rows_keep_all  # noqa: E402

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_GOLD = AUDIT_DIR / "GoldAuditor" / "caption_audit_manual.json"


def load_gold(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load gold caption JSON (``info`` + ``records``)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    info = data.get("info")
    records = data.get("records")
    if not isinstance(info, dict):
        raise ValueError("info must be an object")
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    return dict(info), list(records)


def predicted_good(row: Dict[str, Any]) -> bool:
    """True when validator predicts a good caption (aligns with manual PASS)."""
    fast = str(row.get("fast_validator_label") or "").upper()
    if fast == "PASS":
        return True
    if fast == "FAIL":
        return False
    # UNKNOWN → use LLM judge when present; otherwise treat as not-good.
    llm = str(row.get("llm_judge_label") or "").upper()
    return llm == "PASS"


def row_agreement(row: Dict[str, Any]) -> Optional[bool]:
    """True/False when manual vs validator can be compared; else None."""
    gold = str(row.get("manual_label") or "").strip().upper()
    if gold not in {"PASS", "FAILED"}:
        return None
    return (gold == "PASS") == predicted_good(row)


def annotate_agreement(records: Sequence[Dict[str, Any]]) -> int:
    """Set ``agreement`` on each row; return disagreement count."""
    disagreements = 0
    for row in records:
        agreed = row_agreement(row)
        if agreed is None:
            row.pop("agreement", None)
            continue
        row["agreement"] = agreed
        if not agreed:
            disagreements += 1
    return disagreements


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score GoldAuditor caption_audit_manual.json with the production "
            "fast+LLM caption validator."
        ),
    )
    parser.add_argument(
        "gold_json",
        nargs="?",
        type=Path,
        default=DEFAULT_GOLD,
        help=f"Gold captions JSON (default: {DEFAULT_GOLD})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write scored JSON here (default: <stem>_scored.json next to input)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input gold JSON instead of writing a sibling scored file",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run batched LLM judge on UNKNOWN captions (requires Ollama)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"LLM judge batch size (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_HOST,
        help=f"Ollama API base URL (default {DEFAULT_HOST})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    gold_path = args.gold_json.resolve()
    if not gold_path.is_file():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 1

    info, records = load_gold(gold_path)
    # Drop prior scoring blob if re-running on a scored file.
    info.pop("validator_scoring", None)
    info.pop("disagreement_count", None)

    config = ValidationConfig(llm_batch_size=args.batch_size)
    client = None
    if args.llm:
        client = OllamaClient(host=args.ollama_host, model=args.model)

    print(
        f"Gold caption audit: {gold_path.name} n={len(records)} "
        f"llm={bool(args.llm)}",
        flush=True,
    )
    scored, _stats = score_rows_keep_all(
        records,
        config=config,
        client=client,
        use_llm=bool(args.llm),
    )
    disagreement_count = annotate_agreement(scored)
    print(
        f"Disagreements (manual_label vs fast/llm): {disagreement_count}",
        flush=True,
    )

    out_info = dict(info)
    out_info["disagreement_count"] = disagreement_count

    if args.in_place:
        out_path = gold_path
    elif args.output is not None:
        out_path = args.output.resolve()
    else:
        # If input is already *_scored.json, overwrite that scored file.
        out_path = gold_path.with_name(f"{gold_path.stem}_scored.json")
        if gold_path.stem.endswith("_scored"):
            out_path = gold_path

    payload = {"info": out_info, "records": scored}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(scored)} scored records -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
