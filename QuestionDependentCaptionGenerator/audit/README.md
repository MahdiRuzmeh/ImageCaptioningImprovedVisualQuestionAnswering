# Caption Auditor (`audit/`)

LLM-based sample auditor for question-dependent captions produced by
[`generate.py`](../generate.py).

Randomly samples `test_item_count` non-empty captions from a dataset JSON,
sends them to Ollama in batches, and writes a structured audit report with
PASS/FAIL labels plus automatic caption **precision** / **recall**.

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

Parse failures are fail-closed (`label=FAIL`, `error_type=NONE`, reason notes the parse/LLM error).

## Precision and recall

Automatic Q+A grounding metrics (no human gold labels):

- \(C\) = content words in caption
- \(A\) = content words in answer
- \(S\) = content words in question ∪ answer

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Recall** | \(\|C \cap A\| / \|A\|\) | How much of the answer appears in the caption |
| **Precision** | \(\|C \cap S\| / \|C\|\) | How much of the caption is supported by Q+A |

Empty answer → recall `1.0` if caption non-empty else `0.0`.  
Empty caption content → precision `0.0`.

Reported per record and as `info.mean_precision` / `info.mean_recall`.

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
    "error_type_counts": {"NONE": 40, "HALLUCINATION": 5},
    "mean_precision": 0.91,
    "mean_recall": 0.88
  },
  "records": [
    {
      "question": "...",
      "answer": "...",
      "caption": "...",
      "label": "PASS",
      "reason": "...",
      "error_type": "NONE",
      "precision": 1.0,
      "recall": 1.0
    }
  ]
}
```
