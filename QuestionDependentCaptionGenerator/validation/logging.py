"""Validation trace records and JSONL / sidecar writers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CaptionTraceEntry:
    """One caption attempt at a pipeline stage."""

    stage: str
    caption: str
    source: str = ""


@dataclass
class ValidationTrace:
    """Full validation record for one Q+A row."""

    question_id: int
    image_id: int
    question: str
    answer: str
    rule: str
    captions_trace: List[CaptionTraceEntry] = field(default_factory=list)
    fast_verdict: str = ""
    fast_reasons: List[str] = field(default_factory=list)
    llm_verdict: Optional[str] = None
    final_verdict: str = ""
    validation_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the validation log JSON schema."""
        return {
            "question_id": self.question_id,
            "image_id": self.image_id,
            "question": self.question,
            "answer": self.answer,
            "rule": self.rule,
            "captions_trace": [
                {"stage": e.stage, "caption": e.caption, "source": e.source}
                for e in self.captions_trace
            ],
            "fast_verdict": self.fast_verdict,
            "fast_reasons": list(self.fast_reasons),
            "llm_verdict": self.llm_verdict,
            "final_verdict": self.final_verdict,
            "validation_flags": list(self.validation_flags),
        }


class ValidationLogWriter:
    """Append one JSON object per line to ``*_validation_log.jsonl``."""

    def __init__(self, path: Path) -> None:
        """Open (overwrite) the validation log at ``path``."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.count = 0
        self.failed: List[Dict[str, Any]] = []

    def write(self, trace: ValidationTrace) -> None:
        """Append one validation record."""
        record = trace.to_dict()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1
        if trace.final_verdict == "FAIL":
            self.failed.append(record)

    def close(self) -> None:
        """Close the log file and write the failed sidecar if any failures."""
        try:
            self._fh.close()
        except Exception:
            pass

    def write_failed_sidecar(self, output_path: Path) -> Optional[Path]:
        """Write ``*_validation_failed.json`` next to the captions JSON."""
        if not self.failed:
            return None
        sidecar = output_path.with_name(output_path.stem + "_validation_failed.json")
        payload = {
            "description": "Captions that failed validation (fast or LLM judge)",
            "count": len(self.failed),
            "records": self.failed,
        }
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return sidecar


def validation_log_path(output_path: Path) -> Path:
    """Path for ``{stem}_validation_log.jsonl`` next to a captions JSON."""
    return output_path.with_name(output_path.stem + "_validation_log.jsonl")
