# QuestionDependentCaptionGenerator

Generator baraye sakht-e **question-dependent caption** az VQA v2.

Har sample: `(soal, javab)` → caption mesl `"The car is red."`

Pipeline:

1. VQA questions + annotations ro load mikone (`input_count`)
2. OCR-dependent Q/A pair ha (`is_ocr_question`) — soal hayi ke javab-eshun faghat az ru-ye reading-e text/adad-e ru-ye tasvir mishe fahmid (sign, logo, brand, plate, jersey number, clock) — kollan hazf mishan, chon `SimpleImageCaptioner` OCR nadare va nemitune in target ha ro yad begire; count-esh dar `info.ocr_excluded_count` save mishe
3. Duplicate `(image_id, question, answer)` rows drop mishan (`info.duplicate_count`)
4. Optional `--classify-questions`: binary `DIRECTLY_VISUAL` / `NOT_DIRECTLY_VISUAL`. A **regex fast-path** (`_ALWAYS_VISUAL_RE`) auto-accepts obviously visual patterns (color, count, spatial, sport/game, material, which, doing, animal, visible expression, etc.) without calling the LLM — typically ~60-70% of questions. Only ambiguous questions go through Qwen (prompt `v4_sport_action_material`). Non-visual drops go to sidecar `*_not_directly_visual.json` (faghat baraye captioner train — VQA2 eval dastkhord nashavad)
5. Rule engine try mikone (`caption_rules.py`) — faghat pattern haye daghigh va motmaen
6. Age hich rule match nakone, row `rule="needs_llm"` va `caption=""` mishe
7. Age `--llm` on bashe → Ollama ba packed batch + **two-tier validator** (lexical → optional Qwen PASS/FAIL) + **1 regenerate** then drop

## Files

| File | Kar |
|------|-----|
| `caption_rules.py` | Rule engine + helper ha |
| `generate.py` | CLI: rules + optional LLM fallback |
| `llm_prompts.py` | Packed prompt (chand Q+A toye yek request) |
| `llm_client.py` | Ollama HTTP client + concurrent workers + two-tier validator |
| `question_classifier.py` | Binary DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL filter (regex fast-path + LLM) |
| `audit/audit_captions.py` | Post-hoc QC audit on a captions JSON (optional) |

Progress logs (flush): VQA load, rules scan, classify `i/N`, and
`LLM batch k/N calling Ollama...` **before** each batch (so long waits are visible).

## Data (pishfarz)

- Input (VQA): `../dataset/v2_OpenEnded_mscoco_*_questions.json` + `v2_mscoco_*_annotations.json`
- Output (in this folder): `outputs/v2_question_dependent_captions_{train,val}2014.json`

## Rule engine

`caption_rules.py` faghat pattern hayi ro handle mikone ke bedoon POS tagging motmaen split mishan. Har chizi ke motmaen nist (mesalan subject-e ba article-e nامعین mesle `"a military person"`, ya `which`/`where`/`what brand/sport/room/animal/vehicle/food/drink` — ke NP-eshun azad-tar az un chizi hast ke bе rule split beshe) → `needs_llm`, va SLM (`--llm`) jaygozin-esh mikone. **Hich catch-all fallback rule vojud nadare** — age rule match nashe, caption khali mimoone ta LLM por-esh kone.

### OCR filter (`is_ocr_question`)

Ghabl az rule engine, har Q/A pair check mishe ke aya javab-esh faghat az reading-e text/adad-e ru-ye tasvir be dast miyad (sign says, brand, logo, license plate, jersey/bus number, clock/watch time). `SimpleImageCaptioner` (Stage 1) yek Faster R-CNN region-feature + LSTM captioner-e bedoon OCR hast — faghat pooled visual features mibine, glyph nemikhoone. Pas caption-e sahih baraye in soal ha (mesal: `"The sign says 3M."`) ye target-e yad-nagereftani baraye un mishe, va in pipeline hazfeshun mikone ta signal-e training kasif nashe.

In ye **heuristic** hast, na ground truth (VQA v2 field-e explicit-e "requires OCR" nadare) — do signal combine mishe:

1. Regex-e ru-ye khod-e matn-e soal (`_OCR_QUESTION_RE` toye `caption_rules.py`): "what does ... say", "what is written", "what word(s)", "what letter(s)", "license number/plate", "what brand", "what logo", "what number is on/the/...", "what is the number on...", "what time is it/does".
2. `question_type` (az annotations file, na questions file) — chand prefix-e OCR-heavy (`what does the`, `what brand`, `what number is`, `what time`) tanha-shun ham kafi'e, hata age regex match nakone.

Amdan conservative: prefix haye mobham mesle `what is the name` (mitune "what is the name of this fruit" — OCR nist — ya "what is the name on the jersey" — OCR hast) az list kenar gozashte shode ta soal haye ma'mooli-e visual bishtar-az-hadd filter nashan.

Excluded item ha kollan az `rows` biroon mimoonan (na `needs_llm`, na rule-e digei) — count-eshoon toye stdout print mishe va dar `info.ocr_excluded_count` save mishe.

### Yes/No rule ha (jaye ye `is_are_yesno`-e omoomi)

Be jaye ye rule-e omoomi, chand sub-rule-e narrow darim — har kodoom faghat baraye ye shape-e daghigh:

| Rule | Pattern | Mesal |
|------|---------|-------|
| `yesno_is_anyone` | `Is/Are anyone ...?` | "Is anyone wearing wrist protection?" + yes → "Someone is wearing wrist protection." |
| `yesno_is_everyone` | `Is/Are everyone/everybody ...?` | "Is everyone wearing a hat?" + no → "Not everyone is wearing a hat." |
| `yesno_are_any` | `Are any of ...?` | "Are any of the animals eating?" + yes → "At least one of the animals is eating." |
| `yesno_are_all` | `Is/Are all ...?` | "Are all the flowers white?" + no → "Not all the flowers are white." |
| `yesno_are_both` | `Are both ...?` | "Are both giraffes standing?" + no → "Not both giraffes are standing." |
| `yesno_does_do` | `Does/Do/Did ...?` | **Routed to LLM** (rule kept, not applied) |
| `yesno_modal_have` | `Can/Could/Will/Would/Has/Have/Had ...?` | "Could this photo be from a zoo?" + yes → "This photo could be from a zoo." |
| `yesno_is_this_a` | `Is/Are this/that a/an/the X?` | "Is this a horse?" + no → "This is not a horse." |
| `yesno_is_are_possessive` | `Is/Are the X's Y ...?` | "Is the zebra's tail up?" + no → "The zebra's tail is not up." |
| `yesno_is_are_coordinated` | `Is/Are the X and Y ...?` | "Are the clock and owl made ...?" + no → "The clock and owl are not made ..." |
| `yesno_is_are_predicate` | Simple `Is/Are` + subject + predicate | "Are the animals eating?" + yes → "The animals are eating." (complex / everyone/anyone → LLM or dedicated rule) |
| `is_there` | `Is there (a/an/any) X?` | "Is there any window in the room?" + no → "There is no window in the room." (`any` as whole word — no `ny` bug) |

A subject led by an indefinite article (`"a"`/`"an"`, e.g. `"Is a military person in the picture?"`) can't be split into a head noun without POS tagging, so those rules return `None` and defer to the SLM instead of guessing.

### Routing (`caption_generation_strategy`)

Some categories are too fragile for deterministic rewrite. Helpers in `caption_rules.py`:

| Helper | Behavior |
|--------|----------|
| `should_use_llm_for_does_do` | **Always** LLM for Does/Do/Did (rule kept but never applied) |
| `is_complex_is_are_question` | LLM when predicate has `trying to` / `enough to` / `able to` / `supposed to` / `going to` / `have in common` / `why`, is very long, or has multiple verbs |
| `should_use_llm_for_who` | LLM for non-`Who is/are` (e.g. `Who made...`) or uncertain answers |
| `can_generate_safe_rule_caption` | Rejects broken templates (`The in the...`, `the answer is...`) |
| `caption_generation_strategy` | Returns `"rule"` or `"llm"` |

Simple Is/Are (`Are the animals eating?`) stay rule-based.

### How-many rule (narrowed)

`rule_how_many` faghat 2 shape ro handle mikone:

- `How many <noun> are/is there?`
- `How many <noun> are/is in/on ...?` (location dropped az output)

Har chizi dige (`"...can you see eating?"`, `"...are standing?"`, `"...can be seen?"`) → `needs_llm`. Count agreement: count=1 singularizes (`"windows"` → `"window"`); count>1 / zero pluralizes (`"light post"` → `"light posts"`).

## Run (rules only)

```bash
cd QuestionDependentCaptionGenerator
python generate.py --split train
python generate.py --split val
python generate.py --split train --max-items 1000   # smoke test
```

Rows-e `rule="needs_llm"` toye in mode `caption=""` mimoonan — bayad `--llm` bezani ta por beshan.

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
| `--llm` | off | Baraye `needs_llm` rows az Ollama caption begir |
| `--batch-size` | `10` | Chand Q+A toye **yek** LLM prompt |
| `--model` | `mistral` | Esm model Ollama |
| `--workers` | `1` | Concurrent API request (hamoon yek model) |
| `--ollama-host` | `http://localhost:11434` | Base URL Ollama |
| `--checkpoint-every` | `1` | Har N LLM batch JSON save (`1`, `50`, `100`, …) |
| `--classifier-checkpoint-every` | `50` | Har N classified question classifier checkpoint save |
| `--no-resume` | off | Ignore classifier + LLM checkpoints (fresh start) |
| `--output` | `outputs/...` | Override path output JSON |

### Output validator + retry (two-tier)

Har caption-e LLM, ghabl az accept:

**Tier 1 — lexical / cheap**

1. **Format** (`caption_format_is_valid`): ye jomle-ye declarative — na khali, na `?`, na bracket/quote, na chand jomle, na "the answer".
2. **Polarity / spurious negation**
3. **Relation preserve**: content-word-haye asli-e soal (≤4 stem → hame) bayad toye caption bashan (shade≠free range, wall≠hill).
4. **Answer grounding**: proper noun / number / color / short answer → **verbatim** (Loon≠Loom); digar answers ≥50% token.
5. **Unsupported facts**: caption nabayad content-word-e jadid-e ghalabe ezafe kone.
6. **Batch contamination**

**Tier 2 — Qwen semantic judge** (faghat sample-haye mashkuk):

> Given QUESTION, ANSWER and CAPTION, return PASS only if the caption correctly expresses the answer to the question and adds no unsupported factual information. Otherwise return FAIL.

**Retry policy:** FAIL → **1** regenerate (`single_retries=1`) → FAIL dobare → drop. Counts: `info.validation_retry_count`, `info.validation_failure_count`.

**Final salvage:** leftover `needs_llm` **batched** (default 1 round, no per-item retry).

Low `answer_consensus` rows are **kept** (for later down-weight experiments).

### Resume / checkpoint

**Classifier (`--classify-questions`):**

- Progress saved every `--classifier-checkpoint-every N` classifications (default 50) to `{output_stem}_classifier_checkpoint.json`.
- `Ctrl+C` during classification → checkpoint saved; rerun same command to continue.
- When classification completes, checkpoint is marked `complete` and removed after final output write.

**LLM (`--llm`):**

- `--checkpoint-every N` → har N LLM batch output save (atomic write).
- `Ctrl+C` → hatman yek checkpoint save, bad exit.
- Dobare **hamoon command** → az ja-monde edame (`llm_fallback` skip + classifier checkpoint if needed).
- Full fast-resume skips reload/classifier when output JSON row count matches `post_filter_count` / `directly_visual_count`.

**Start fresh:** delete output + sidecar files, or pass `--no-resume`.

```bash
# start / continue (same command)
python generate.py --split train --llm --classify-questions \
  --model qwen2.5:3b-instruct-q4_K_M --batch-size 10 \
  --checkpoint-every 100 --classifier-checkpoint-every 50
```

### LLM failure log

With `--llm`, every leftover `rule=needs_llm` is explained in a sidecar log:

`outputs/v2_question_dependent_captions_{split}2014.json.llm_failures.log`

Typical reasons:

| reason | Meaning |
|--------|---------|
| `connection_error` | Ollama not reachable |
| `http_error` | Ollama HTTP error (model missing, …) |
| `parse_length_mismatch` | Model did not return N captions as JSON array |
| `parse_json_error` / `parse_no_json_array` | Response was not valid JSON |
| `answer_mismatch` | Caption omitted the answer tokens |
| `spurious_negation` | Answer isn't yes/no, but caption added "no"/"not"/... (meaning-flip hallucination, e.g. "No clock was made by Rolex." for answer "rolex") |
| `contains_question_mark` | Caption is a question, or echoes/repeats the question, instead of a statement |
| `contains_brackets` / `contains_quotes` | Caption has `[]`/`{}` or quotation marks (echoed formatting) |
| `contains_answer_phrase` | Caption literally says "the answer"/"the answer is" instead of a natural sentence |
| `multiple_sentences` / `double_period` | More than one sentence, or a stray ".." |
| `too_short` / `empty_caption` | Caption has fewer than 2 words, or is empty |
| `relation_mismatch` | Subject/relation words from the question missing in caption |
| `unsupported_facts` | Caption invents content not in Q+A |
| `semantic_fail` | Tier-2 Qwen judge returned FAIL |
| `empty_response` / `timeout` | Model returned nothing / timed out |

If `--llm` finishes with any `needs_llm` left, the process exits with code `1` and prints the log path. Fix the top reason and re-run the same command.

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
  "answer_count": 8,
  "answer_consensus": 0.8,
  "caption": "The car is red.",
  "rule": "what_color"
}
```

`rule` mishe yeki az: rule name ha (`what_color`, `how_many`, `yesno_are_all`, …), `needs_llm` (hanuz LLM nagerefte — `caption` khali), ya `llm_fallback` (LLM tolid karde).

`answer_count` = chand ta az 10 annotator dagigan hamun mode answer ro dadan; `answer_consensus` = `answer_count / total_annotators` (rounded). In annotator agreement ast, na model confidence — ba'dan mitune baraye loss weighting estefade beshe.

`info.llm` (age `--llm`): `model`, `batch_size`, `workers`, `host`, `prompt_version`, `validation`.

Accounting fields (bayad jam beshan):

| Field | Meaning |
|-------|---------|
| `input_count` | Q/A ids scanned at load |
| `ocr_excluded_count` | Regex OCR prefilter |
| `duplicate_count` | Dedup drops |
| `post_filter_count` | Rows kept after OCR/dedup/classifier (for resume matching) |
| `directly_visual_count` | Classifier kept (ya hame rows age classify off) |
| `not_directly_visual_count` | Classifier dropped |
| `dropped_empty_count` | Empty/short (excluding counted validation failures) |
| `validation_retry_count` | Per-item regenerations |
| `validation_failure_count` | Final validation drops |
| `num_samples` | Final annotations length |

Identity: `input ≈ ocr + duplicate + not_directly_visual + num_samples + dropped_empty + validation_failure` (va `directly_visual ≈ num_samples + dropped_empty + validation_failure`).

## QC validators (LLM)

Beyond format checks, accepted LLM captions must pass Tier-1 relation / verbatim / unsupported-facts checks and, when suspicious, Tier-2 PASS/FAIL. Prefer `--batch-size` ≤ 10.

## DIRECTLY_VISUAL filter

**Off by default.** Bedoon flag, `question_classifier.py` call nemishe.

`DIRECTLY_VISUAL` = javab mostaghim az zaher-e tasvir-e sabet, **bedoon OCR**, **bedoon knowledge-e biruni**, **bedoon nazar-e shakhsi**. Visible sport/game/activity identity, material, which+attribute, doing, animal, and visible expression are DIRECTLY_VISUAL; sport **rules** / professionalism are NOT. Prompt version: `v4_sport_action_material`.

### Flags

| Flag | Chi mikone |
|------|------------|
| `--classify-questions` | Har soal → regex fast-path ya Qwen binary `DIRECTLY_VISUAL` / `NOT_DIRECTLY_VISUAL`. Drop + write sidecar. Resume via `*_classifier_checkpoint.json`. |
| `--classifier-checkpoint-every` | `50` | Save classifier progress every N questions |
| `--drop-subjective-candidates` | Offline: regex candidates drop (bedoon Qwen). |
| `--classifier-model` | Model Ollama baraye classifier; default = `--model` |

```bash
python generate.py --split train --llm --classify-questions \
  --model qwen2.5:3b-instruct-q4_K_M --batch-size 10
```

Sidecar: `outputs/v2_question_dependent_captions_{split}2014_not_directly_visual.json` — baraye tahlil-e ba'di. In filter **faghat** baraye dataset-e train-e Captioner ast; VQA2 asli baraye eval dastkhord nashavad.

Counts: `info.directly_visual_count`, `info.not_directly_visual_count`, `info.question_classifier.label_counts` (includes `FAST_PATH_VISUAL` count).

## Tests + audit

`audit/audit_captions.py` **ekhtiari** hast.

```bash
cd QuestionDependentCaptionGenerator
python audit/audit_captions.py outputs/v2_question_dependent_captions_train2014.json
```

## Re-pilot (before full 443k)

```bash
python generate.py --split train --llm --max-items 25000 --batch-size 10 \
  --classify-questions --model qwen2.5:3b-instruct-q4_K_M \
  --checkpoint-every 50 --output outputs/pilot_25k.json
python audit/audit_captions.py outputs/pilot_25k.json
```

## Notes

- Javab = mode answer (10 annotator) — hamoon logic `SimpleVQA/train.py`
- `rule_counts` to `info` baraye statistik
- OCR-e-mahvar soal ha (`is_ocr_question`) kollan az `rows` hazf mishan — `info.ocr_excluded_count`
- Duplicate rows — `info.duplicate_count`
- Low-consensus samples **hazf nemishan**; `answer_consensus` negah dashte mishe
- Baraye train captioner: faghat rows-e `caption` gheyr-khali
