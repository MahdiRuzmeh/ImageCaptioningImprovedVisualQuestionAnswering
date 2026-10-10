# Audit tools (`audit/`)

DevAuditor scorers for tuning the **production** caption validator and question
classifier without re-running the full corpus. Both scripts mirror the same
code paths used by [`generate.py`](../generate.py).

Manual sets live under [`DevAuditor/`](DevAuditor/):

| File | Purpose |
|------|---------|
| `DevAuditor/caption_audit_manual.json` | Human `manual_label` PASS / FAILED for captions |
| `DevAuditor/classifier_audit_manual.json` | Human `manual_label` DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL |

Each record has a `source` field:

| Value | Meaning |
|-------|---------|
| `natural` | Item came from the LLM pipeline outputs (main captions / suspicious / NDV sidecars) |
| `synthetic` | Human-authored for testing and tuning (default when not found in pipeline outputs) |

Tuning loop: score set → compare to `manual_label` → adjust validator /
classifier → re-score. Only then optionally re-run full `generate.py`.

---

## Caption scorer (`audit_captions.py`)

Runs the same fast + optional LLM judge path as production
(`validation.pipeline.score_rows_keep_all`, same logic as `validate_rows`).

**Does not drop** FAIL rows — every gold record is kept and labeled so metrics
stay aligned with `manual_label`.

### Usage

From `QuestionDependentCaptionGenerator/`:

```bash
python audit/audit_captions.py
python audit/audit_captions.py --llm --batch-size 10
python audit/audit_captions.py audit/DevAuditor/caption_audit_manual.json --in-place
```

| Arg | Meaning |
|-----|---------|
| `gold_json` | Optional path (default: `audit/DevAuditor/caption_audit_manual.json`) |
| `--llm` | Run batched LLM judge on UNKNOWN (requires Ollama) |
| `--batch-size` | LLM batch size (default `10`) |
| `--model` / `--ollama-host` | Ollama settings (same defaults as `generate.py`) |
| `--output` | Explicit output path |
| `--in-place` | Overwrite the gold file (default writes `<stem>_scored.json`) |

### Fields written (per record)

| Field | Values |
|-------|--------|
| `manual_label` | Preserved (`PASS` / `FAILED`) |
| `source` | Preserved (`natural` / `synthetic`) |
| `fast_validator_label` | `FAIL` / `UNKNOWN` |
| `llm_judge_label` | `PASS` / `SUSPICIOUS` when LLM ran; omitted otherwise |
| `caption_status` | `Ready to Use` (LLM PASS) / `Need to Manual validate` (SUSPICIOUS) |
| `agreement` | `true` / `false` — manual vs validator prediction |

Console prints how many items disagree: ``manual_label`` vs validator
prediction (good = UNKNOWN + ``llm_judge_label`` PASS; FAIL is not-good).
Output ``info`` stores only ``disagreement_count`` (plus the original gold
metadata). Filter disagreements in the scored JSON with ``"agreement": false``.

---

## Classifier scorer (`classify_questions.py`)

Runs the same blacklist→NDV | UNKNOWN→parallel chat-turn LLM cascade as
[`question_classifier.py`](../question_classifier.py) / `generate.py`
(`v19_chat_turn_fewshots`).

**Does not drop** rows — every gold record is scored in place.

### Usage

```bash
python audit/classify_questions.py
python audit/classify_questions.py --batch-size 10 --llm-parallel 4
python audit/classify_questions.py audit/DevAuditor/classifier_audit_manual.json --in-place
```

| Arg | Meaning |
|-----|---------|
| `gold_json` | Optional path (default: `audit/DevAuditor/classifier_audit_manual.json`) |
| `--batch-size` | UNKNOWN items flushed together to parallel `classify_batch` (default `10`) |
| `--llm-parallel` | Max concurrent classifier Ollama requests (default `4`; set `OLLAMA_NUM_PARALLEL` >= this) |
| `--host` / `--model` | Ollama settings |
| `--no-blacklist-drop` / `--no-fast-path` | Disable blacklist auto-NDV; all rows → parallel LLM |
| `--output` / `--in-place` | Same as caption scorer |

### Fields written (per record)

| Field | Values |
|-------|--------|
| `manual_label` | Preserved |
| `source` | Preserved (`natural` / `synthetic`) |
| `classifier_label` | `DIRECTLY_VISUAL` / `NOT_DIRECTLY_VISUAL` |
| `visual_filter_source` | `blacklist` (hard NDV) / `llm_classifier` |
| `detail` / `non_visual_reason` | Optional gate / LLM detail |
| `agreement` | `true` / `false` — manual vs `classifier_label` |

Console prints how many items disagree: ``manual_label`` vs
``classifier_label``. Output ``info`` stores only ``disagreement_count``
(plus the original gold metadata). Filter disagreements with
``"agreement": false``.

---

## Prerequisites

- Ollama running locally when using `--llm` (captions) or when any gold
  classifier rows need LLM confirm
- Default model: `qwen2.5:3b-instruct-q4_K_M`
