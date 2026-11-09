# Caption error check (`caption_errors/`)

Finds captions whose **meaning** is wrong for their question + answer and
regenerates only those. Captions with only grammar problems are kept as they are.

## Fields added to every item

| Field | Values |
|---|---|
| `caption_quality_verdict` | `"ok"`, `"GRAMMAR_ISSUE_ONLY"`, `"WRONG"`, or `null` (not judged yet) |
| `verdict_reason` | for `"WRONG"`: `answer_missing`, `answer_contradicted`, `meaning_changed`, `unrelated`, `added_facts`; otherwise `null` |
| `suspicious` | `true` when the code check (step 3) thinks the FINAL caption may still be wrong |
| `suspicious_reasons` | e.g. `["extra_words:dry"]`, `["polarity_flip"]`; `[]` when not suspicious |

Regenerated rows also get:

| Field | Meaning |
|---|---|
| `caption` | the new caption (when one passed the format checks) |
| `caption_original` | the caption before regeneration |
| `caption_original_verdict_reason` | why the original was `"WRONG"` |
| `regen_rounds` | 1 or 2 |

`caption_status` is set to `Ready to Use` for `"ok"`, `"GRAMMAR_ISSUE_ONLY"` and
fixed rows, and to `Need to Manual validate` for rows still `"WRONG"` after 2
rounds. Those rows keep their last caption.

`info.caption_error_fix.progress` gives rows judged / left, how many were
regenerated / fixed / still wrong, and `complete` (true only when every row is judged).

## The three steps

1. **7B judge** on every caption: `ok` / `GRAMMAR_ISSUE_ONLY` / `WRONG`.
2. **Regenerate** the `WRONG` ones (packs of 5, checked, judged again, 2 rounds).
3. **Code check** (`suspicious.py`, no LLM) on the FINAL caption of every row:
   - `extra_words`: a content word that is in neither the question nor the
     answer ("to dry", "in the sky", "both", "door" for "wall");
   - `polarity_flip`: the yes/no meaning looks flipped, or a "not" that a
     non-yes/no answer never implied.

   Words left out of the caption do not count. Rows marked here are written
   to `<stem>_suspicious.json` for a later pass with a bigger model. The check
   never changes a caption or verdict.

   Test set (298 captions): the code check marks 16 of the 19 wrong captions
   the 7B (j2) passed, and 48 of the 213 correct ones. On the current file
   (before the run) it marks 14.3% of rows.

## How a batch is processed (steps 1 and 2)

```
10 captions --judge--> ok / GRAMMAR_ISSUE_ONLY ............ kept unchanged
                    \-> WRONG + reason --regenerate (reason as hint)
                                          --fast_validate (format, polarity, ...)
                                          --judge again
                                          ok?  -> fixed
                                          no   -> round 2, then Need to Manual validate
```

- Judge (`judge.py`): text only, 10 items per request, Ollama structured output
  (JSON schema). Unreadable items are retried one by one. A row with no readable
  verdict is left unjudged and retried later; it is never counted as `"ok"`.
- Regeneration (`regenerate.py`): uses the generation rules of
  `llm_prompts.SYSTEM_PROMPT`, without the few-shot captions (the 3B model had
  copied them ~900 times). The wrong caption itself is not shown: given
  "Wrong caption: X", the 3B model wrote "not X".
- Packs of 5 (`--regen-batch-size`), same as generation. Like
  `llm_client.OllamaClient.captions_with_retry`, every new caption is checked
  against its pack (`check_caption`: `fast_validate` hard rejects including
  `batch_contamination`, plus `spurious_negation`). Each reject is asked again
  on its own (`--no-regen-single-fallback` turns that off), and only captions
  that pass go to the judge.

  Test on the 85 WRONG gold captions with the 3B (2026-11-09, my grading of the new captions):

  | Setting | Correct new captions | Speed |
  |---|---|---|
  | batch 1 | ~76 / 85 (89%) | 0.80 captions/s |
  | batch 5, no checks | ~50 / 85 (59%): captions moved to other pairs, unwanted "not" | 3.27 captions/s |
  | **batch 5 + checks + single redo** (default) | ~78 / 85 (92%) | 1.16 captions/s |

## Files

| File | Role |
|---|---|
| `judge.py` | judge prompt, reply parser, small Ollama client |
| `regenerate.py` | regeneration prompt + parser |
| `judge_captions.py` | resumable run over the whole captions JSON |
| `../audit/audit_caption_errors.py` | scores the judge against the hand-labelled set |
| `../audit/DevAuditor/caption_error_audit_manual.json` | 298 hand-labelled real rows (`manual_verdict`, `manual_reason`; `borderline: true` = check first) |
| `../kaggle/kaggle_judge_runner.py` | Kaggle runner (2 GPUs, time limit, resume), see `../kaggle/README.md` |

## Commands (from `QuestionDependentCaptionGenerator/`)

```bash
# score the judge on the hand-labelled set (your PC: your 3B)
python audit/audit_caption_errors.py --model qwen2.5:3b-instruct-q4_K_M

# smoke test on the first 200 rows; the JSON in the output dir still holds every row
python -m caption_errors.judge_captions --input outputs/vqa_v2_question_dependent_captions_train2014.json \
    --output-dir outputs_caption_check --model qwen2.5:3b-instruct-q4_K_M --max-items 200

# only rows whose caption_status is "Need to Manual validate" (a later
# --scope all run in the same output dir skips them)
python -m caption_errors.judge_captions --input ... --output-dir outputs_caption_check --scope manual

# rebuild the captions JSON from the progress file only (no LLM)
python -m caption_errors.judge_captions --input ... --output-dir outputs_caption_check --write-only

python -m unittest tests.test_caption_errors
```

## Resuming

Progress goes to `<output-dir>/<stem>_caption_check_progress.jsonl` after every
batch. Run the same command again to continue. The first line stores model,
prompt versions, `regen_rounds` and input name, and a run with different
settings refuses to use the file. On every exit the captions JSON and
`<stem>_caption_check_state.json` are written.

Exit codes: `0` all rows judged, `3` rows left, `4` model not replying,
`5` time limit (`--deadline-unix`), `130` Ctrl+C.

## Calibration (2026-11-09)

| Model | WRONG precision | WRONG recall | Notes |
|---|---|---|---|
| qwen2.5:3b-instruct-q4_K_M (PC) | 0.56 | 0.85 | flags many fine captions: not good enough for the full run |
| qwen2.5:7b-instruct-q4_K_M (Kaggle) | – | – | run `run.mode = "calibrate"` |

Target before the full run: recall ≥ 0.90, precision ≥ 0.80.

In the hand-labelled sample, 13 of 199 `Ready to Use` rows (6.5%) and 72 of 99
`Need to Manual validate` rows (73%) are `"WRONG"`. For the full file
(276,210 + 7,497 rows) that suggests roughly 18k + 5.5k ≈ **23k wrong captions**.
This is a rough estimate from a small sample.
