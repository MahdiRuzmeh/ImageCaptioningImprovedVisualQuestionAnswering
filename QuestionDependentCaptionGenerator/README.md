# QuestionDependentCaptionGenerator

Generator baraye sakht-e **question-dependent caption** az VQA v2.

Har sample: `(soal, javab)` → caption mesl `"The car is red."`

Pipeline:

1. VQA questions + annotations ro load mikone
2. OCR-dependent Q/A pair ha (`is_ocr_question`) — soal hayi ke javab-eshun faghat az ru-ye reading-e text/adad-e ru-ye tasvir mishe fahmid (sign, logo, brand, plate, jersey number, clock) — kollan hazf mishan, chon `SimpleImageCaptioner` OCR nadare va nemitune in target ha ro yad begire; count-esh dar `info.ocr_excluded_count` save mishe
3. Duplicate `(image_id, question, answer)` rows (az annotator haye mokhtalef ke soal-e eyni neveshtan) drop mishan, faghat avalin occurrence mimoone
4. Rule engine try mikone (`caption_rules.py`) — faghat pattern haye daghigh va motmaen (color, how-many-e sade, is/are-e narrow, ...)
5. Age hich rule match nakone, row `rule="needs_llm"` va `caption=""` mishe (hich template-e sakhtegi sakhte nemishe)
6. Age `--llm` on bashe → Ollama/Mistral ba **packed batch** captioning-e in row ha ro anjam mide, ba format validator + retry

## Files

| File | Kar |
|------|-----|
| `caption_rules.py` | Rule engine + helper ha |
| `generate.py` | CLI: rules + optional LLM fallback |
| `llm_prompts.py` | Packed prompt (chand Q+A toye yek request) |
| `llm_client.py` | Ollama HTTP client + concurrent workers + output validator |

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
| `yesno_are_any` | `Are any of ...?` | "Are any of the animals eating?" + yes → "At least one of the animals is eating." |
| `yesno_are_all` | `Is/Are all ...?` | "Are all the flowers white?" + no → "Not all the flowers are white." |
| `yesno_are_both` | `Are both ...?` | "Are both giraffes standing?" + no → "Not both giraffes are standing." |
| `yesno_does_do` | `Does/Do/Did + subject + verb ...?` | "Does this photo show train tracks?" + yes → "This photo shows train tracks." |
| `yesno_modal_have` | `Can/Could/Will/Would/Has/Have/Had ...?` | "Could this photo be from a zoo?" + yes → "This photo could be from a zoo." |
| `yesno_is_this_a` | `Is/Are this/that a/an/the X?` | "Is this a horse?" + no → "This is not a horse." |
| `yesno_is_are_possessive` | `Is/Are the X's Y ...?` | "Is the zebra's tail up?" + no → "The zebra's tail is not up." |
| `yesno_is_are_coordinated` | `Is/Are the X and Y ...?` | "Are the clock and owl made ...?" + no → "The clock and owl are not made ..." |
| `yesno_is_are_predicate` | `Is/Are/Was/Were + subject + predicate` | "Is the stove light on?" + yes → "The stove light is on." |

A subject led by an indefinite article (`"a"`/`"an"`, e.g. `"Is a military person in the picture?"`) can't be split into a head noun without POS tagging, so those rules return `None` and defer to the SLM instead of guessing.

### How-many rule (narrowed)

`rule_how_many` faghat 2 shape ro handle mikone:

- `How many <noun> are/is there?`
- `How many <noun> are/is in/on ...?` (location dropped az output)

Har chizi dige (`"...can you see eating?"`, `"...are standing?"`, `"...can be seen?"`) → `needs_llm`. Count=1 ham "of"-tail ro singularize mikone: `"kinds of animals"` → `"kind of animal"`.

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
| `--checkpoint-every` | `1` | Har N batch JSON save (`1`, `50`, `100`, …) |
| `--no-resume` | off | Checkpoint ghabli ro ignore kon |
| `--output` | `outputs/...` | Override path output JSON |

### Output validator + retry

Har caption-e LLM, ghabl az accept shodan, 2 check migzarune:

1. **Format** (`caption_format_is_valid` toye `llm_client.py`): ye jomle-ye ساده-ye declarative bashe — na khali, na soal (`?`), na bracket (`[]`/`{}`), na quotation mark, na do ta "." ro ham ("..") , na chand jomle (mesal-e rad-shode: `"This is a home. It is not a restaurant."`), na "the answer is"/"the answer" (meta-phrase-e ghalat).
2. **Content**: javab bayad toye caption bashe (`answer_in_caption`) va caption nabayad ye negation-e ghalat ezafe kone (`has_spurious_negation`). `answer_in_caption` chand relaxation dare:
   - digit/word adad ha moadel ham hastan (`"2"` va `"two"` yeki hesab mishan).
   - inflection-e sade (light stem, na real lemmatizer) — `"stands"` va `"standing"` yeki hesab mishan, `"dogs"` va `"dog"` ham (suffix `-ing`/`-ed`/`-es`/`-s` pak mishe age >=3 harf bemune).
   - lazem nist hame token-e javab ain-e caption bashan — **>=50%** token-e javab (whole-word/stem match) kafi'e (mesal: answer `"holding it"` + caption `"He is holding the dog."` → `"holding"` match, `"it"` na, 1/2=50% → accepted).

Age har kodoom fail beshe, hamun item ta **3 bar** (`single_retries=3`) dobare az LLM darkhast mishe (batch attempt + per-item retries), ghabl az inke `needs_llm` bemune.

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
  "answer_confidence": 0.8,
  "caption": "The car is red.",
  "rule": "what_color"
}
```

`rule` mishe yeki az: rule name ha (`what_color`, `how_many`, `yesno_are_all`, …), `needs_llm` (hanuz LLM nagerefte — `caption` khali), ya `llm_fallback` (LLM tolid karde).

`answer_count` = chand ta az 10 annotator dagigan hamun mode answer ro dadan; `answer_confidence` = `answer_count / total_annotators` (rounded).

`info.llm` (age `--llm`): `model`, `batch_size`, `workers`, `host`, `prompt_version`.

`info.ocr_excluded_count`: chand ta OCR-dependent Q/A pair kollan hazf shod ghabl az caption generation (see [OCR filter](#ocr-filter-is_ocr_question)).

## Notes

- Javab = mode answer (10 annotator) — hamoon logic `SimpleVQA/train.py`
- `rule_counts` to `info` baraye statistik
- OCR-e-mahvar soal ha (`is_ocr_question`) kollan az `rows` hazf mishan ghabl az dedup/rule — count-eshoon `info.ocr_excluded_count` va stdout
- Duplicate `(image_id, question, answer)` rows (mesal: do ta annotator-e mokhtalef literally hamun soal ro neveshtan) dar `load_vqa_pairs` drop mishan — count-esh toye stdout print mishe
- Baraye train captioner: dataset loader `(image_id, question, caption)` lazem hast — faghat rows-e `caption` gheyr-khali estefade kon (yani `rule != "needs_llm"`)
