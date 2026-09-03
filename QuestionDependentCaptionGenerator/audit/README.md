# Caption Auditor (`audit/`)

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
