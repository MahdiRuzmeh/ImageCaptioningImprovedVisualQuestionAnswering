# Audit tools (`audit/`)

Tools for inspecting caption quality and the binary question classifier.

---

## Caption Auditor (`audit_captions.py`)

LLM-based sample auditor for question-dependent captions produced by
[`generate.py`](../generate.py).

Randomly samples `test_item_count` non-empty captions from a dataset JSON,
sends them to Ollama in batches, and writes a structured audit report with
PASS/FAIL labels.

## Prerequisites

- Ollama running locally
- Model available: `qwen2.5:3b-instruct-q4_K_M` (default)

## Usage

From `QuestionDependentCaptionGenerator/`:

```bash
python audit/audit_captions.py outputs/vqa_v2_question_dependent_captions_train2014.json 50
python audit/audit_captions.py outputs/vqa_v2_question_dependent_captions_train2014.json 50 --batch-size 10
```

| Arg | Meaning |
|-----|---------|
| `caption_file_path` | Captions dataset JSON (`info` + `annotations`) |
| `test_item_count` | How many captions to sample (capped at pool size) |
| `--batch-size` | Items per Ollama call (default `10`) |

Fixed defaults: seed `42`, model `qwen2.5:3b-instruct-q4_K_M`,
host `http://localhost:11434`.

Output path (next to the input file):

`{stem}_llm_caption_audit_k{k}_seed42.json`

## LLM labels

| Field | Values |
|-------|--------|
| `label` | `PASS` / `FAIL` |
| `error_type` | `NONE` / `GRAMMAR_ERROR` / `HALLUCINATION` / `WRONG_CAPTION` |
| `reason` | Short explanation from the model |

- **PASS** — grammatical, expresses the answer, no facts beyond Q+A
- **GRAMMAR_ERROR** — broken / unnatural wording
- **HALLUCINATION** — extra objects, attributes, actions, etc.
- **WRONG_CAPTION** — missing / wrong / contradictory answer

Parse/LLM batch failures are **retried once as a batch**. Items that still fail
are kept as FAIL in the report and appended to
`audit/llm_caption_audit_errors.jsonl`. Truncated batch JSON is salvaged when
complete objects can still be extracted.

## Output schema

```json
{
  "info": {
    "source": "...",
    "test_item_count": 50,
    "seed": 42,
    "batch_size": 10,
    "model": "qwen2.5:3b-instruct-q4_K_M",
    "host": "http://localhost:11434",
    "pool_size": 390,
    "num_audited": 50,
    "label_counts": {"PASS": 40, "FAIL": 10},
    "error_type_counts": {"NONE": 40, "HALLUCINATION": 5}
  },
  "records": [
    {
      "question": "...",
      "answer": "...",
      "caption": "...",
      "label": "PASS",
      "reason": "...",
      "error_type": "NONE"
    }
  ]
}
```

---

## Classifier retest (`reclassify_questions.py`)

Re-runs [`question_classifier.py`](../question_classifier.py) on a
`*_not_directly_visual.json` sidecar so you can debug false
`NOT_DIRECTLY_VISUAL` drops.

For each item it records Fast Path / suspect diagnostics, calls
`classify_one`, and marks whether the new label **flipped** vs the prior
sidecar label.

### Usage

From `QuestionDependentCaptionGenerator/`:

```bash
# All dropped questions
python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json

# Single question_id
python audit/reclassify_questions.py outputs/vqa_v2_question_dependent_captions_train2014_not_directly_visual.json 9002
```

| Arg | Meaning |
|-----|---------|
| `not_directly_visual_path` | Sidecar JSON (`info` + `annotations`) |
| `question_id` | Optional: reclassify only this id (default: all) |
| `--batch-size` | `llm_confirm` items per Ollama call (default `10`) |

Optional: `--host`, `--model` (defaults match the pipeline classifier).

### Output

Always under `outputs/`:

- All items: `{stem}_reclassify.json`
- One id: `{stem}_reclassify_qid{question_id}.json`

Each record includes `prior_label`, `exempt_match`, `blacklist_match`,
`gate` (`fast_path` / `default_visual` / `llm_confirm`), `new_label`,
`non_visual_reason`, `detail`, and `flipped`. Only `llm_confirm` items call
Ollama, packed in batches (`classify_batch`; salvage with `classify_one` on
parse failure).
