"""Standalone CLI to re-validate an existing captions JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_client import OllamaClient
from validation import (
    VALIDATOR_VERSION,
    ValidationConfig,
    ValidationLogWriter,
    validate_rows,
    validation_log_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone validation."""
    parser = argparse.ArgumentParser(
        description="Re-validate question-dependent captions JSON (fast + LLM judge).",
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to captions JSON (outputs/*.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write filtered annotations here (default: overwrite input)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run batched LLM judge on UNKNOWN captions (requires Ollama)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Fast validator only; UNKNOWN rows are dropped",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="LLM judge batch size (default: 10)",
    )
    parser.add_argument(
        "--overlap-fail",
        type=float,
        default=0.30,
        help="Overlap ratio below this → fast FAIL (default: 0.30)",
    )
    parser.add_argument(
        "--overlap-pass",
        type=float,
        default=0.50,
        help="Overlap ratio at/above this → fast PASS candidate (default: 0.50)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum caption word count (default: 3)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=30,
        help="Maximum caption word count (default: 30)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b-instruct-q4_K_M",
        help="Ollama model for LLM judge",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m validation.cli``."""
    args = parse_args(argv)
    input_path = args.input_json.resolve()
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = list(payload.get("annotations") or [])
    info = dict(payload.get("info") or {})

    config = ValidationConfig(
        min_words=args.min_words,
        max_words=args.max_words,
        overlap_fail_threshold=args.overlap_fail,
        overlap_pass_threshold=args.overlap_pass,
        llm_batch_size=args.batch_size,
    )

    use_llm = args.llm and not args.no_llm
    client = None
    if use_llm:
        client = OllamaClient(host=args.ollama_host, model=args.model)

    out_path = (args.output or input_path).resolve()
    log_path = validation_log_path(out_path)
    log_writer = ValidationLogWriter(log_path)

    kept, failed, stats = validate_rows(
        rows,
        config=config,
        client=client,
        use_llm=use_llm,
        log_writer=log_writer,
    )
    log_writer.close()
    sidecar = log_writer.write_failed_sidecar(out_path)

    info["num_samples"] = len(kept)
    info["validation"] = {
        "validator_version": VALIDATOR_VERSION,
        **config.__dict__,
        **stats.to_dict(),
        "validation_log": str(log_path),
    }
    if sidecar:
        info["validation"]["validation_failed_sidecar"] = str(sidecar)

    out_payload = {"info": info, "annotations": kept}
    out_path.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Validated {len(rows)} rows: kept {len(kept)}, failed {len(failed)} "
        f"(fast pass={stats.fast_pass_count}, fail={stats.fast_fail_count}, "
        f"unknown={stats.fast_unknown_count}; "
        f"llm pass={stats.llm_pass_count}, fail={stats.llm_fail_count})"
    )
    print(f"Log -> {log_path}")
    if sidecar:
        print(f"Failed sidecar -> {sidecar}")
    print(f"Output -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
