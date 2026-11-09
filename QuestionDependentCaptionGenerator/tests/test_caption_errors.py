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
from caption_errors import regenerate as rg
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
    base = dict(retries=0, retry_backoff=0.0, regen_rounds=2, regen_batch_size=5,
                regen_temperature=0.3, no_regen_single_fallback=False)
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
             mock.patch.object(jc, "regenerate_checked",
                               return_value=[("The motorcycle is leaning against the wall.", None)]):
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
             mock.patch.object(jc, "regenerate_checked",
                               side_effect=[[("A cat is there.", None)],
                                            [("A dog is there.", None)]]):
            recs = jc.process_batch([ROWS[1]], client=None, args=_args())
        self.assertEqual(recs[0]["verdict"], WRONG)
        self.assertEqual(recs[0]["regen_rounds"], 2)
        self.assertEqual(recs[0]["caption"], "A dog is there.")

    def test_rejected_regen_goes_to_next_round_without_judge(self) -> None:
        judge_calls = [[(WRONG, "unrelated")], [(OK, None)]]
        with mock.patch.object(jc, "judge_items", side_effect=judge_calls) as j,              mock.patch.object(jc, "regenerate_checked",
                               side_effect=[[(None, "unparsed")],
                                            [("The motorcycle is leaning against the wall.", None)]]):
            recs = jc.process_batch([ROWS[1]], client=None, args=_args())
        self.assertEqual(j.call_count, 2)                  # no judge call for the reject
        self.assertEqual(recs[0]["verdict"], OK)
        self.assertEqual(recs[0]["regen_rounds"], 2)
        self.assertEqual(recs[0]["history"][0]["detail"], "unparsed")

    def test_unreadable_verdict_is_not_recorded(self) -> None:
        with mock.patch.object(jc, "judge_items", return_value=[None, (OK, None), None]):
            recs = jc.process_batch(ROWS, client=None, args=_args())
        self.assertEqual([r["question_id"] for r in recs], [2])

    def test_transport_error_propagates(self) -> None:
        with mock.patch.object(jc, "judge_items", side_effect=TransportError("x")):
            with self.assertRaises(TransportError):
                jc.process_batch(ROWS, client=None, args=_args())


PACK = [
    ("What activity is this boy doing?", "surfing", "x", "unrelated"),
    ("Which way is the one way sign pointing?", "right", "x", "answer_contradicted"),
    ("What color tie does the man have?", "green", "x", "answer_contradicted"),
]


class TestRegenerateChecked(unittest.TestCase):
    def test_pack_rejects_are_redone_one_by_one(self) -> None:
        packed = ["The one way sign points right.",            # belongs to item 2
                  "The one way sign is pointing right.",
                  "The man does not have a green tie."]        # spurious negation
        # near-duplicates: the contamination check rejects BOTH (it can't tell
        # which one is the original), so items 1, 2 and 3 are redone alone
        singles = [["The boy is surfing."], ["The one way sign is pointing right."],
                   ["The man has a green tie."]]
        with mock.patch.object(rg, "regenerate",
                               side_effect=[packed] + singles) as gen:
            out = rg.regenerate_checked(None, PACK, batch_size=5)
        self.assertEqual(out, [("The boy is surfing.", None),
                               ("The one way sign is pointing right.", None),
                               ("The man has a green tie.", None)])
        self.assertEqual(gen.call_count, 4)                    # 1 pack + 3 singles

    def test_no_single_fallback(self) -> None:
        with mock.patch.object(rg, "regenerate",
                               return_value=["The boy is surfing.", None,
                                             "The man has a green tie."]) as gen:
            out = rg.regenerate_checked(None, PACK, batch_size=5,
                                        single_fallback=False)
        self.assertEqual(out[1], (None, "unparsed"))
        self.assertEqual(gen.call_count, 1)

    def test_check_caption(self) -> None:
        self.assertEqual(rg.check_caption("What color tie?", "green",
                                          "The man does not have a green tie."),
                         "spurious_negation")
        self.assertIsNone(rg.check_caption("Is it raining?", "no", "It is not raining."))
        self.assertEqual(rg.check_caption("Q?", "a", None), "unparsed")


class TestSuspicious(unittest.TestCase):
    def check(self, q, a, c):
        from caption_errors.suspicious import suspicious_reasons
        return suspicious_reasons(q, a, c)

    def test_extra_words(self) -> None:
        self.assertEqual(self.check("What is hung up?", "clothes",
                                    "Clothes are hung up to dry."), ["extra_words:dry"])
        self.assertEqual(self.check("Are these cows standing up?", "yes",
                                    "Both cows are standing up."), ["extra_words:both"])
        self.assertEqual(self.check("What is leaning against the wall?", "motorcycle",
                                    "The motorcycle is leaning against the door."),
                         ["extra_words:door"])

    def test_clean_captions(self) -> None:
        for q, a, c in [("How many cookies can be seen?", "2", "Two cookies can be seen."),
                        ("How many fish?", "0", "No fish can be seen."),
                        ("Is the dog on the sofa?", "no", "The dog is not on the sofa."),
                        ("What color is the man's hat?", "red", "The man's hat is red."),
                        ("Is it raining?", "yes", "It is raining.")]:
            self.assertEqual(self.check(q, a, c), [], c)

    def test_polarity_flip(self) -> None:
        self.assertEqual(self.check("Is it raining?", "yes", "It is not raining."),
                         ["polarity_flip"])
        self.assertEqual(self.check("Is the dog on the sofa?", "no",
                                    "The dog is on the sofa."), ["polarity_flip"])
        self.assertEqual(self.check("What color tie does the man have?", "green",
                                    "The man does not have a green tie."),
                         ["polarity_flip"])

    def test_missing_words_do_not_count(self) -> None:
        self.assertEqual(self.check("What is the largest object in the water?", "ship",
                                    "A ship is in the water."), [])


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

        # step 3 runs on the FINAL caption, also for rows not judged yet
        self.assertFalse(row["suspicious"])                  # fixed caption is clean
        self.assertEqual(s["suspicious_reasons"], ["extra_words:door"])
        self.assertTrue(jc.build_row(ROWS[1], None)["suspicious"])

    def test_scope_manual(self) -> None:
        self.assertEqual([r["question_id"] for r in ROWS if jc.in_scope(r, "manual")], [3])
        self.assertEqual(len([r for r in ROWS if jc.in_scope(r, "all")]), 3)
        done = {3: {"verdict": GRAMMAR, "reason": None, "original_verdict": GRAMMAR,
                    "original_reason": None}}
        s = jc.summarize(ROWS, done, "manual")
        self.assertEqual((s["rows_in_scope"], s["rows_left"], s["complete"],
                          s["complete_all"], s["rows_left_all"]), (1, 0, True, False, 2))
        self.assertFalse(jc.summarize(ROWS, done, "all")["complete"])

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
