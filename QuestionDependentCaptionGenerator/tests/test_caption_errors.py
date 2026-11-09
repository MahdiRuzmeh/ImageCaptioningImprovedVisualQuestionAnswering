"""Unit tests for caption_errors (no Ollama needed).

Run from QuestionDependentCaptionGenerator/:  python -m unittest tests.test_caption_errors
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from caption_errors import judge_captions as jc
from caption_errors.judge import (
    GRAMMAR, OK, WRONG, build_messages, parse_reply, TransportError,
)
from caption_errors.regenerate import build_messages as regen_messages
from caption_errors.regenerate import parse_reply as regen_parse
from reclassify_captions import load_progress


class TestJudgeParse(unittest.TestCase):
    def test_object_reply(self) -> None:
        raw = json.dumps({"items": [
            {"id": 1, "verdict": "ok", "reason": "none"},
            {"id": 2, "verdict": "WRONG", "reason": "meaning_changed"},
            {"id": 3, "verdict": "GRAMMAR_ISSUE_ONLY", "reason": "added_facts"}]})
        self.assertEqual(parse_reply(raw, 3),
                         [(OK, None), (WRONG, "meaning_changed"), (GRAMMAR, None)])

    def test_bare_array_fenced_and_aliases(self) -> None:
        raw = '```json\n[{"id":1,"verdict":"PASS"},{"id":2,"verdict":"grammar only"}]\n```'
        self.assertEqual(parse_reply(raw, 2), [(OK, None), (GRAMMAR, None)])

    def test_wrong_without_reason_defaults(self) -> None:
        raw = '{"items":[{"id":1,"verdict":"WRONG","reason":"???"}]}'
        self.assertEqual(parse_reply(raw, 1), [(WRONG, "meaning_changed")])

    def test_missing_and_unknown_ids(self) -> None:
        raw = '{"items":[{"id":2,"verdict":"ok"},{"id":3,"verdict":"maybe"}]}'
        self.assertEqual(parse_reply(raw, 3), [None, (OK, None), None])

    def test_broken_json(self) -> None:
        self.assertEqual(parse_reply("I think they are fine", 2), [None, None])

    def test_by_position_when_no_ids(self) -> None:
        raw = '[{"verdict":"ok"},{"verdict":"WRONG","reason":"unrelated"}]'
        self.assertEqual(parse_reply(raw, 2), [(OK, None), (WRONG, "unrelated")])

    def test_prompt_numbers_items(self) -> None:
        msgs = build_messages([("Q1?", "a1", "C1."), ("Q2?", "a2", "C2.")])
        user = msgs[1]["content"]
        self.assertIn("1. QUESTION: Q1?", user)
        self.assertIn("2. QUESTION: Q2?", user)
        self.assertIn("exactly 2 objects", user)


class TestRegenerate(unittest.TestCase):
    def test_prompt_has_reason_but_not_wrong_caption(self) -> None:
        msgs = regen_messages([("What is leaning against the wall?", "motorcycle",
                                "The motorcycle is leaning against the door.",
                                "meaning_changed")])
        user = msgs[1]["content"]
        self.assertIn("What is leaning against the wall?", user)
        self.assertIn("changes a word or relation", user)
        self.assertNotIn("against the door", user)

    def test_parse(self) -> None:
        self.assertEqual(regen_parse('{"captions": ["the lid is open"]}', 1),
                         ["The lid is open."])
        self.assertEqual(regen_parse('["A.", "B."]', 1), [None])
        self.assertEqual(regen_parse("nope", 2), [None, None])


def _args(**kw) -> argparse.Namespace:
    base = dict(retries=0, retry_backoff=0.0, regen_rounds=2, regen_batch_size=1,
                regen_temperature=0.3)
    base.update(kw)
    return argparse.Namespace(**base)


ROWS = [
    {"question_id": 1, "question": "How many dogs are there?", "answer": "2",
     "caption": "There are two dogs.", "caption_status": "Ready to Use"},
    {"question_id": 2, "question": "What is leaning against the wall?",
     "answer": "motorcycle", "caption": "The motorcycle is leaning against the door.",
     "caption_status": "Ready to Use"},
    {"question_id": 3, "question": "How many people in all?", "answer": "3",
     "caption": "There are three people in alls.", "caption_status": "Need to Manual validate"},
]


class TestProcessBatch(unittest.TestCase):
    def test_wrong_is_regenerated_and_fixed(self) -> None:
        judge_calls = [
            [(OK, None), (WRONG, "meaning_changed"), (GRAMMAR, None)],
            [(OK, None)],                      # judge of the new caption
        ]
        with mock.patch.object(jc, "judge_items", side_effect=judge_calls), \
             mock.patch.object(jc, "regenerate",
                               return_value=["The motorcycle is leaning against the wall."]):
            recs = jc.process_batch(ROWS, client=None, args=_args())
        by = {r["question_id"]: r for r in recs}
        self.assertEqual(by[1]["verdict"], OK)
        self.assertNotIn("regen_rounds", by[1])
        self.assertEqual(by[3]["verdict"], GRAMMAR)        # grammar only: untouched
        self.assertNotIn("regen_rounds", by[3])
        self.assertEqual(by[2]["verdict"], OK)
        self.assertEqual(by[2]["original_reason"], "meaning_changed")
        self.assertEqual(by[2]["regen_rounds"], 1)
        self.assertEqual(by[2]["caption"], "The motorcycle is leaning against the wall.")

    def test_still_wrong_after_two_rounds(self) -> None:
        judge_calls = [[(WRONG, "unrelated")], [(WRONG, "meaning_changed")],
                       [(WRONG, "meaning_changed")]]
        with mock.patch.object(jc, "judge_items", side_effect=judge_calls), \
             mock.patch.object(jc, "regenerate",
                               side_effect=[["A cat is there."], ["A dog is there."]]):
            recs = jc.process_batch([ROWS[1]], client=None, args=_args())
        self.assertEqual(recs[0]["verdict"], WRONG)
        self.assertEqual(recs[0]["regen_rounds"], 2)
        self.assertEqual(recs[0]["caption"], "A dog is there.")

    def test_unreadable_verdict_is_not_recorded(self) -> None:
        with mock.patch.object(jc, "judge_items", return_value=[None, (OK, None), None]):
            recs = jc.process_batch(ROWS, client=None, args=_args())
        self.assertEqual([r["question_id"] for r in recs], [2])

    def test_transport_error_propagates(self) -> None:
        with mock.patch.object(jc, "judge_items", side_effect=TransportError("x")):
            with self.assertRaises(TransportError):
                jc.process_batch(ROWS, client=None, args=_args())


class TestOutputs(unittest.TestCase):
    def test_build_row_fields(self) -> None:
        fixed = {"verdict": OK, "reason": None, "original_verdict": WRONG,
                 "original_reason": "meaning_changed", "regen_rounds": 1,
                 "caption": "The motorcycle is leaning against the wall."}
        row = jc.build_row(ROWS[1], fixed)
        self.assertEqual(row["caption"], "The motorcycle is leaning against the wall.")
        self.assertEqual(row["caption_original"], ROWS[1]["caption"])
        self.assertEqual(row["caption_original_verdict_reason"], "meaning_changed")
        self.assertEqual(row["caption_quality_verdict"], OK)
        self.assertIsNone(row["verdict_reason"])
        self.assertEqual(row["caption_status"], "Ready to Use")
        keys = list(row)
        self.assertEqual(keys.index("caption_quality_verdict"), keys.index("caption") + 1)

        grammar = {"verdict": GRAMMAR, "reason": None, "original_verdict": GRAMMAR,
                   "original_reason": None}
        g = jc.build_row(ROWS[2], grammar)
        self.assertEqual(g["caption"], ROWS[2]["caption"])
        self.assertEqual(g["caption_status"], "Ready to Use")
        self.assertNotIn("caption_original", g)

        still = {"verdict": WRONG, "reason": "unrelated", "original_verdict": WRONG,
                 "original_reason": "unrelated", "regen_rounds": 2, "caption": None}
        s = jc.build_row(ROWS[1], still)
        self.assertEqual(s["caption"], ROWS[1]["caption"])     # no usable new caption
        self.assertEqual(s["caption_status"], "Need to Manual validate")

        self.assertIsNone(jc.build_row(ROWS[0], None)["caption_quality_verdict"])

    def test_summary_and_resume_with_torn_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x_caption_check_progress.jsonl"
            meta = {k: "v" for k in jc.HEADER_KEYS}
            jc.check_progress_header(p, meta)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({"question_id": 1, "verdict": OK, "reason": None,
                                    "original_verdict": OK, "original_reason": None}) + "\n")
                f.write('{"question_id": 2, "verd')            # killed mid-write
            done = load_progress(p)
            self.assertEqual(set(done), {1})
            s = jc.summarize(ROWS, done)
            self.assertEqual((s["rows_judged"], s["rows_left"], s["complete"]), (1, 2, False))
            with self.assertRaises(SystemExit):
                jc.check_progress_header(p, dict(meta, model="other"))


if __name__ == "__main__":
    unittest.main()
