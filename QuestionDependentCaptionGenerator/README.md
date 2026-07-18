# QuestionDependentCaptionGenerator

Generator baraye sakht-e **question-dependent caption** az VQA v2.

Har sample: `(soal, javab)` → caption mesl `"The car is red."`

Pipeline:

1. VQA questions + annotations ro load mikone
2. Rule engine try mikone (`caption_rules.py`)
3. Age match nashod (`fallback`) va `--llm` on bashe → Ollama/Mistral ba **packed batch**

## Files

| File | Kar |
|------|-----|
| `caption_rules.py` | Rule engine + helper ha |
| `generate.py` | CLI: rules + optional LLM fallback |
| `llm_prompts.py` | Packed prompt (chand Q+A toye yek request) |
| `llm_client.py` | Ollama HTTP client + concurrent workers |

## Data (pishfarz)

- Input (VQA): `../dataset/v2_OpenEnded_mscoco_*_questions.json` + `v2_mscoco_*_annotations.json`
- Output (in this folder): `outputs/v2_question_dependent_captions_{train,val}2014.json`

## Run (rules only)

```bash
cd QuestionDependentCaptionGenerator
python generate.py --split train
python generate.py --split val
python generate.py --split train --max-items 1000   # smoke test
```

## Run (rules + LLM fallback)

Pishniaz: Ollama run bashe va model pull shode bashe.

```bash
cd QuestionDependentCaptionGenerator

# smoke
python generate.py --split train --llm --batch-size 10 --max-items 200 \
  --model qwen2.5:3b-instruct-q4_K_M

# full val — checkpoint har 50 batch (kamtar disk I/O)
python generate.py --split val --llm --batch-size 10 --workers 1 \
  --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 50

# checkpoint har 100 batch
python generate.py --split train --llm --batch-size 10 --workers 1 \
  --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 100
```

### LLM CLI args

| Arg | Default | Meaning |
|-----|---------|---------|
| `--llm` | off | Baraye unmatched ha az Ollama caption begir |
| `--batch-size` | `10` | Chand Q+A toye **yek** LLM prompt |
| `--model` | `mistral` | Esm model Ollama |
| `--workers` | `1` | Concurrent API request (hamoon yek model) |
| `--ollama-host` | `http://localhost:11434` | Base URL Ollama |
| `--checkpoint-every` | `1` | Har N batch JSON save (`1`, `50`, `100`, …) |
| `--no-resume` | off | Checkpoint ghabli ro ignore kon |
| `--output` | `outputs/...` | Override path output JSON |

### Resume / checkpoint

- `--checkpoint-every N` → har N LLM batch output save (atomic write).
- `Ctrl+C` → hatman yek checkpoint save, bad exit.
- Dobare **hamoon command** → az ja-monde edame (`llm_fallback` skip).
- Redo az aval: file toye `outputs/` ro pak kon.

```bash
# start / continue (same command + same --checkpoint-every optional)
python generate.py --split val --llm --batch-size 10 --workers 1 \
  --model qwen2.5:3b-instruct-q4_K_M --checkpoint-every 50
```

### 8GB VRAM notes

- Yek model load mishe (na chand copy).
- Asli-tarin speedup = `--batch-size` (pack).
- `--workers 1` safe-tarine; `--workers 2` faghat age OOM nashod.
- Age concurrent mikhay, Ollama side: `OLLAMA_NUM_PARALLEL` ba `--workers` align bashe.

## Output row

```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question": "What color is the car?",
  "answer": "red",
  "caption": "The car is red.",
  "rule": "what_color"
}
```

`rule` mishe yeki az: rule name ha (`what_color`, …), `fallback` (template), ya `llm_fallback` (az Ollama).

`info.llm` (age `--llm`): `model`, `batch_size`, `workers`, `host`, `prompt_version`.

## Notes

- Javab = mode answer (10 annotator) — hamoon logic `SimpleVQA/train.py`
- `rule_counts` to `info` baraye statistik
- Baraye train captioner: dataset loader `(image_id, question, caption)` lazem hast
