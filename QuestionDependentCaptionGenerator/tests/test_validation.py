"""Unit tests for the two-layer caption validator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from caption_rules import DIGIT_TO_WORD, is_ocr_question
from validation import (
    ValidationConfig,
    ValidationLogWriter,
    ValidationTrace,
    fast_validate,
    compute_overlap_ratio,
)
from validation.checks import FLAG_OVERLAP_TOO_LOW, quantifier_incomplete
from validation.fast_validator import FastVerdict
from validation.llm_validator import JudgeItem, LlmVerdict, parse_judge_response
from validation.logging import CaptionTraceEntry
from validation.overlap import overlap_verdict
from validation.pipeline import validate_rows
from validation.tokens import numeric_equivalents


class TestFastValidator(unittest.TestCase):
    """Fast FAIL / UNKNOWN cases (no PASS)."""

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
        """High overlap must not UNKNOWN-clean when caption echoes the question."""
        r = fast_validate(
            "How many flags do you see",
            "1",
            "one flag do you see",
        )
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertIn("echoes_question", r.reasons)

    def test_possessive_apostrophe_not_quotes(self) -> None:
        r = fast_validate(
            "What is the job of the boy?",
            "pitcher",
            "The boy's job is to pitch the ball.",
        )
        self.assertNotEqual(r.verdict, FastVerdict.FAIL)
        self.assertNotIn("contains_quotes", r.reasons)

    def test_question_negation_not_spurious(self) -> None:
        r = fast_validate(
            "How many people are not in the water in this picture?",
            "14",
            "Fourteen people are not in the water.",
        )
        self.assertNotEqual(r.verdict, FastVerdict.FAIL)
        self.assertNotIn("spurious_negation", r.reasons)

    def test_snowboarding_stem_match(self) -> None:
        r = fast_validate(
            "What sport is being played?",
            "snowboarding",
            "Snowboarders are playing.",
        )
        self.assertNotIn("answer_mismatch", r.reasons)

    def test_overcast_contrastive_negation_unknown(self) -> None:
        """Contrastive 'not sunny but overcast' is not a hard FAIL."""
        r = fast_validate(
            "Is it overcast or sunny?",
            "overcast",
            "It is not sunny but overcast outside.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)
        self.assertNotIn("spurious_negation", r.reasons)

    def test_ones_on_top_paraphrase_unknown(self) -> None:
        """Missing filler 'ones' is not a hard answer_mismatch FAIL."""
        r = fast_validate(
            "Which bananas are newer?",
            "ones on top",
            "The newer bananas are on top.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)
        self.assertNotIn("answer_mismatch", r.reasons)

    def test_unknown_borderline_overlap(self) -> None:
        cfg = ValidationConfig(overlap_fail_threshold=0.30, overlap_pass_threshold=0.90)
        r = fast_validate(
            "What are the animals doing?",
            "eating",
            "The animals are eating.",
            config=cfg,
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)

    def test_overlap_too_low_is_unknown_not_fail(self) -> None:
        cfg = ValidationConfig(overlap_fail_threshold=0.90, overlap_pass_threshold=0.95)
        r = fast_validate(
            "Is this a cloudy day?",
            "no",
            "No clouds are present.",
            config=cfg,
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)
        self.assertIn(FLAG_OVERLAP_TOO_LOW, r.flags)

    def test_answer_forty_accepts_word(self) -> None:
        r = fast_validate(
            "How many people are watching?",
            "40",
            "Forty people are watching.",
        )
        self.assertNotEqual(r.verdict, FastVerdict.FAIL)
        self.assertNotIn("answer_mismatch", r.reasons)

    def test_answer_one_accepts_a(self) -> None:
        r = fast_validate(
            "How many dogs are there?",
            "1",
            "There is a dog.",
        )
        self.assertNotEqual(r.verdict, FastVerdict.FAIL)
        self.assertNotIn("answer_mismatch", r.reasons)

    def test_answer_one_accepts_one(self) -> None:
        r = fast_validate(
            "How many dogs are there?",
            "1",
            "There is one dog.",
        )
        self.assertNotEqual(r.verdict, FastVerdict.FAIL)
        self.assertNotIn("answer_mismatch", r.reasons)

    def test_quantifier_hard_mismatch_fail(self) -> None:
        r = fast_validate(
            "Are both giraffes standing?",
            "no",
            "Both giraffes are standing.",
        )
        self.assertEqual(r.verdict, FastVerdict.FAIL)
        self.assertTrue(
            "quantifier_mismatch" in r.reasons or "polarity_mismatch" in r.reasons
        )

    def test_quantifier_incomplete_unknown(self) -> None:
        r = fast_validate(
            "Are both giraffes standing?",
            "yes",
            "The giraffes are standing.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)
        self.assertIn("quantifier_incomplete", r.flags)

    def test_any_existence_not_quantifier_incomplete(self) -> None:
        self.assertFalse(
            quantifier_incomplete(
                "Are there any flowers on the shower curtain?",
                "There are flowers on the shower curtain.",
            )
        )
        r = fast_validate(
            "Are there any flowers on the shower curtain?",
            "yes",
            "There are flowers on the shower curtain.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)
        self.assertNotIn("quantifier_incomplete", r.flags)

    def test_unknown_cake_layers_omitted_noun(self) -> None:
        r = fast_validate(
            "How many layers are in this cake?",
            "7",
            "There are seven layers.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)

    def test_unknown_animals_eating_action_paraphrase(self) -> None:
        r = fast_validate(
            "What are the animals doing?",
            "eating",
            "The animals are eating.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)


class TestLexicalJudgeOverride(unittest.TestCase):
    """Lexical safety net for LLM judge false SUSPICIOUS / meaning traps."""

    def test_faithful_restatement(self) -> None:
        from validation.checks import lexical_caption_looks_faithful

        self.assertTrue(
            lexical_caption_looks_faithful(
                "What color are the headlights?",
                "yellow",
                "The headlights are yellow.",
            )
        )

    def test_solitude_trap(self) -> None:
        from validation.checks import (
            lexical_caption_looks_faithful,
            solitude_quantifier_trap,
        )

        self.assertTrue(
            solitude_quantifier_trap(
                "Is this elephant all alone?",
                "no",
                "Not all the elephants are alone.",
            )
        )
        self.assertFalse(
            lexical_caption_looks_faithful(
                "Is this elephant all alone?",
                "no",
                "Not all the elephants are alone.",
            )
        )

    def test_neither_trap_blocks_lexical_pass(self) -> None:
        from validation.checks import (
            lexical_caption_looks_faithful,
            neither_stronger_than_no_trap,
        )

        self.assertTrue(
            neither_stronger_than_no_trap(
                "Are both men smiling?",
                "no",
                "Neither man is smiling.",
            )
        )
        self.assertFalse(
            lexical_caption_looks_faithful(
                "Are both men smiling?",
                "no",
                "Neither man is smiling.",
            )
        )

    def test_spurious_denial_blocks_lexical_pass(self) -> None:
        from validation.checks import lexical_caption_looks_faithful

        self.assertFalse(
            lexical_caption_looks_faithful(
                "What kind of food is shown?",
                "pizza",
                "The food shown is not pizza.",
            )
        )


class TestOverlap(unittest.TestCase):
    """Overlap ratio and digit/word equivalence."""

    def test_digit_word_equivalence_in_verbatim(self) -> None:
        r = fast_validate(
            "How many cookies can be seen?",
            "2",
            "Two cookies can be seen.",
        )
        self.assertEqual(r.verdict, FastVerdict.UNKNOWN)
        self.assertNotIn("answer_mismatch", r.reasons)

    def test_digit_map_covers_forty(self) -> None:
        self.assertEqual(DIGIT_TO_WORD["40"], "forty")
        eqs = numeric_equivalents("40")
        self.assertIn("forty", eqs)

    def test_numeric_equivalents_include_articles_for_one(self) -> None:
        eqs = numeric_equivalents("1")
        self.assertIn("one", eqs)
        self.assertIn("a", eqs)
        self.assertIn("an", eqs)

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


class TestOcrFilter(unittest.TestCase):
    """Expanded first-level OCR detector."""

    def test_what_number_bus(self) -> None:
        self.assertTrue(is_ocr_question("What number bus is this?"))

    def test_what_name_is_on(self) -> None:
        self.assertTrue(is_ocr_question("What name is on the cake?"))


class TestLlmJudgeParse(unittest.TestCase):
    """LLM judge PASS / SUSPICIOUS parsing."""

    def test_parse_suspicious(self) -> None:
        items = [JudgeItem(index=0, question="Q", answer="a", caption="C")]
        raw = '[{"id": 0, "verdict": "SUSPICIOUS"}]'
        results = parse_judge_response(raw, items)
        self.assertEqual(results[0].verdict, LlmVerdict.SUSPICIOUS)

    def test_legacy_fail_maps_to_suspicious(self) -> None:
        items = [JudgeItem(index=0, question="Q", answer="a", caption="C")]
        raw = '[{"id": 0, "verdict": "FAIL"}]'
        results = parse_judge_response(raw, items)
        self.assertEqual(results[0].verdict, LlmVerdict.SUSPICIOUS)

    def test_parse_preserves_local_ids(self) -> None:
        items = [
            JudgeItem(index=0, question="Q0", answer="a", caption="C0"),
            JudgeItem(index=1, question="Q1", answer="a", caption="C1"),
        ]
        raw = '[{"id": 0, "verdict": "PASS"}, {"id": 1, "verdict": "SUSPICIOUS"}]'
        results = parse_judge_response(raw, items)
        self.assertEqual(results[0].verdict, LlmVerdict.PASS)
        self.assertEqual(results[1].verdict, LlmVerdict.SUSPICIOUS)

    def test_validate_rows_keeps_suspicious_without_llm(self) -> None:
        rows = [
            {
                "question_id": 1,
                "image_id": 1,
                "question": "Are both giraffes standing?",
                "answer": "yes",
                "caption": "The giraffes are standing.",
                "rule": "llm_fallback",
            }
        ]
        kept, failed, stats = validate_rows(rows, use_llm=False, client=None)
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(kept), 1)
        self.assertIn("suspicious", kept[0]["validation_flags"])
        self.assertEqual(kept[0]["caption_status"], "Need to Manual validate")
        self.assertEqual(stats.llm_suspicious_count, 1)

    def test_validate_rows_unknown_without_llm_is_manual(self) -> None:
        """Well-formed captions are UNKNOWN → SUSPICIOUS when no LLM."""
        rows = [
            {
                "question_id": 1,
                "image_id": 1,
                "question": "What color are the dishes?",
                "answer": "pink and yellow",
                "caption": "The dishes are pink and yellow.",
                "rule": "color",
            }
        ]
        kept, failed, stats = validate_rows(rows, use_llm=False, client=None)
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["caption_status"], "Need to Manual validate")
        self.assertEqual(stats.fast_unknown_count, 1)
        self.assertEqual(stats.llm_suspicious_count, 1)


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

    def test_log_writer_suspicious_sidecar(self) -> None:
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
                    fast_verdict="UNKNOWN",
                    llm_verdict="SUSPICIOUS",
                    final_verdict="SUSPICIOUS",
                    validation_flags=["suspicious"],
                )
            )
            writer.close()
            sidecar = writer.write_suspicious_sidecar(out)
            self.assertIsNotNone(sidecar)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["count"], 1)


if __name__ == "__main__":
    unittest.main()
