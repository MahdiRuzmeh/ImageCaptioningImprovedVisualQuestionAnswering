"""Unit tests for the two-layer caption validator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validation import (
    ValidationConfig,
    ValidationLogWriter,
    ValidationTrace,
    fast_validate,
    compute_overlap_ratio,
)
from validation.fast_validator import FastVerdict
from validation.logging import CaptionTraceEntry
from validation.overlap import overlap_verdict


class TestFastValidator(unittest.TestCase):
    """Fast PASS / FAIL / UNKNOWN cases."""

    def test_fail_empty(self) -> None:
        r = fast_validate("What color?", "red", "")
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertIn("empty_caption", r.reasons)

    def test_fail_brackets(self) -> None:
        r = fast_validate("What color?", "red", "The car is (red).")
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertIn("contains_brackets", r.reasons)

    def test_fail_question_mark(self) -> None:
        r = fast_validate("What color?", "red", "Is the car red?")
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertIn("contains_question_mark", r.reasons)

    def test_fail_too_short(self) -> None:
        r = fast_validate("What color?", "red", "Red.")
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertIn("too_short", r.reasons)

    def test_fail_counterexample_echo(self) -> None:
        """High overlap must not PASS when caption echoes the question."""
        r = fast_validate(
            "How many flags do you see",
            "1",
            "one flag do you see",
        )
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertIn("echoes_question", r.reasons)

    def test_pass_color_rule(self) -> None:
        r = fast_validate(
            "What color are the dishes?",
            "pink and yellow",
            "The dishes are pink and yellow.",
        )
        self.assertEqual(r.verdict, FastVerdict.PASS)

    def test_unknown_borderline_overlap(self) -> None:
        cfg = ValidationConfig(overlap_fail_threshold=0.30, overlap_pass_threshold=0.90)
        r = fast_validate(
            "What are the animals doing?",
            "eating",
            "The animals are eating.",
            config=cfg,
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)


class TestOverlap(unittest.TestCase):
    """Overlap ratio and digit/word equivalence."""

    def test_digit_word_equivalence_in_verbatim(self) -> None:
        r = fast_validate(
            "How many cookies can be seen?",
            "2",
            "Two cookies can be seen.",
        )
        self.assertEqual(r.verdict, FastVerdict.PASS)

    def test_overlap_ratio_bounded(self) -> None:
        ratio = compute_overlap_ratio(
            "What color is the car?",
            "The car is red.",
        )
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_overlap_bands(self) -> None:
        cfg = ValidationConfig(overlap_fail_threshold=0.30, overlap_pass_threshold=0.50)
        band, _ = overlap_verdict(
            "What color is the car?",
            "The car is red.",
            cfg,
        )
        self.assertIn(band, ("fail", "pass", "borderline"))


class TestValidationLog(unittest.TestCase):
    """Validation log schema."""

    def test_trace_serializes_captions_trace(self) -> None:
        trace = ValidationTrace(
            question_id=1,
            image_id=2,
            question="Q?",
            answer="a",
            rule="how_many",
            captions_trace=[
                CaptionTraceEntry(
                    stage="generation",
                    caption="Two cookies can be seen.",
                    source="rule",
                ),
                CaptionTraceEntry(
                    stage="retry_1",
                    caption="There are two cookies.",
                    source="llm_fallback",
                ),
            ],
            fast_verdict="UNKNOWN",
            fast_reasons=["relation_low"],
            llm_verdict="PASS",
            final_verdict="PASS",
        )
        d = trace.to_dict()
        self.assertEqual(len(d["captions_trace"]), 2)
        self.assertEqual(d["captions_trace"][0]["stage"], "generation")

    def test_log_writer_failed_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "captions_train2014.json"
            log_path = Path(tmp) / "captions_train2014_validation_log.jsonl"
            writer = ValidationLogWriter(log_path)
            writer.write(
                ValidationTrace(
                    question_id=9,
                    image_id=1,
                    question="Q",
                    answer="no",
                    rule="llm_fallback",
                    fast_verdict="FAIL",
                    final_verdict="FAIL",
                )
            )
            writer.close()
            sidecar = writer.write_failed_sidecar(out)
            self.assertIsNotNone(sidecar)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["count"], 1)


if __name__ == "__main__":
    unittest.main()
