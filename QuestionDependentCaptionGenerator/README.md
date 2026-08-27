# QuestionDependentCaptionGenerator

Generator baraye sakht-e **question-dependent caption** az VQA v2.

Har sample: `(soal, javab)` → caption mesl `"The car is red."`

Pipeline:

1. VQA questions + annotations ro load mikone (`input_count`)
2. OCR-dependent Q/A pair ha (`is_ocr_question`) — soal hayi ke javab-eshun faghat az ru-ye reading-e text/adad-e ru-ye tasvir mishe fahmid (sign, logo, brand, plate, jersey number, clock) — kollan hazf mishan, chon `SimpleImageCaptioner` OCR nadare va nemitune in target ha ro yad begire; count-esh dar `info.ocr_excluded_count` save mishe
3. Duplicate `(image_id, question, answer)` rows drop mishan (`info.duplicate_count`)
4. Optional `--classify-questions`: binary `DIRECTLY_VISUAL` / `NOT_DIRECTLY_VISUAL`. The gate is a **conservative whitelist** (`_FAST_PATH_VISUAL_RE`: colour / count / existence / spatial / animal|sport|room|food|… / do-you-see / end-anchored doing|holding|wearing) — match + no suspect marker → `fast_path`; else UNKNOWN → Qwen (`v7_expanded_fast_path`). Har row field-e `visual_filter_source` (`fast_path` ya `llm_classifier`) migire. `--no-fast-path` hame ro be LLM mifreste. Non-visual drops go to sidecar `*_not_directly_visual.json` (faghat baraye captioner train — VQA2 eval dastkhord nashavad)
5. Rule engine try mikone (`caption_rules.py`) — faghat pattern haye daghigh va motmaen
6. Age hich rule match nakone, row `rule="needs_llm"` va `caption=""` mishe
7. Age `--llm` on bashe → Ollama ba packed batch + **two-tier validator** (high-precision lexical reject → Qwen PASS/FAIL) + **1 regenerate** then drop
8. Validator ru **hame** caption ha (rule ham) run mishe; moshkel haye mashkuk be jaye drop, `validation_flags` migiran. Har retry (validator ya generation) toye `*_validation_audit.jsonl` sabt mishe

## Files

| File | Kar |
|------|-----|
| `caption_rules.py` | Rule engine + helper ha |
| `generate.py` | CLI: rules + optional LLM fallback |
| `llm_prompts.py` | Packed prompt (chand Q+A toye yek request) |
| `llm_client.py` | Ollama HTTP client + concurrent workers + two-tier validator |
| `question_classifier.py` | Binary DIRECTLY_VISUAL / NOT_DIRECTLY_VISUAL filter (conservative Fast Path whitelist; everything else goes to the LLM) |
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

1. Regex-e ru-ye khod-e matn-e soal (`_OCR_QUESTION_RE` toye `caption_rules.py`): "what does ... say", "what is written", "what word(s)", "what letter(s)/initials on", "license number/plate", "what brand", "what logo", "what number is on/the/...", "what is the number on...", "number on the shirt", "shirt/jersey/bus/train/room/gate number", "name of the street" / "street name", "written/printed/engraved/stamped on ...", "what time is it/does".
2. `question_type` (az annotations file, na questions file) — chand prefix-e OCR-heavy (`what does the`, `what brand`, `what number is`, `what time`) tanha-shun ham kafi'e, hata age regex match nakone.

Amdan conservative: prefix haye mobham mesle `what is the name` (mitune "what is the name of this fruit" — OCR nist — ya "what is the name on the jersey" — OCR hast) az list kenar gozashte shode ta soal haye ma'mooli-e visual bishtar-az-hadd filter nashan; shakl-e bare-esh `_NON_VISUAL_SUSPECT_RE`-e classifier ro trigger mikone, pas LLM ta'in mikone.

Tartib-e filter ha sabet ast: **OCR → consensus → dedup → rules → classifier**, pas ye soal-e OCR hich vaght be Fast Path nemirese.

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
| `yesno_is_this_a` | `Is/Are this/that a/an/the X?` | "Is this a horse?" + no → "This is not a horse." |
| `yesno_is_are_possessive` | `Is/Are the X's Y ...?` | "Is the zebra's tail up?" + no → "The zebra's tail is not up." |
| `yesno_is_are_coordinated` | `Is/Are the X and Y ...?` | "Are the clock and owl made ...?" + no → "The clock and owl are not made ..." |
| `yesno_is_are_predicate` | Simple `Is/Are` + subject + predicate | "Are the animals eating?" + yes → "The animals are eating."; locative "Is the baby with his daddy?" + yes → "The baby is with his daddy."; PP leftover must be adjectival/participle (`squishy`) — bare-noun leftovers (`tourists`) and everyone/anyone → LLM or a dedicated rule |
| `is_there` | `Is there (a/an/any) X?` | "Is there any window in the room?" + no → "There is no window in the room." (`any` as whole word — no `ny` bug) |

A subject led by an indefinite article (`"a"`/`"an"`, e.g. `"Is a military person in the picture?"`) can't be split into a head noun without POS tagging, so those rules return `None` and defer to the SLM instead of guessing.

### Rule haye hazf-shode (Comments8)

Do rule kollan pak shodan, chon template-eshun grammar ro kharab mikard:

| Rule-e hazf-shode | Chera | Alan |
|-------------------|-------|------|
| `yesno_modal_have` (`Can/Could/Will/Would/Has/Have/Had ...?`) | Auxiliary ro jabeja mikard: "This photo be could ...", "The plane fly will ..." | Hame be LLM (`needs_llm`) |
| `what_is` (`What is ...?`) | Sub-type haye ziad (`What is it called?`, `What is it for?`, `What is the weather like?`) bedoon parser-e vagheie mishkanand | Hame be LLM (`needs_llm`) |

`what_is_doing` hamchenan rule-e jodast va kar mikone.

### Routing (`caption_generation_strategy`)

Some categories are too fragile for deterministic rewrite. Helpers in `caption_rules.py`:

| Helper | Behavior |
|--------|----------|
| `should_use_llm_for_does_do` | **Always** LLM for Does/Do/Did (rule kept but never applied) |
| `is_complex_is_are_question` | LLM when predicate has `trying to` / `enough to` / `able to` / `supposed to` / `going to` / `have in common` / `why`, is very long, or has multiple verbs |
| `should_use_llm_for_who` | LLM for non-`Who is/are` (e.g. `Who made...`) or uncertain answers |
| `can_generate_safe_rule_caption` | Rejects broken templates (`The in the...`, `the answer is...`, `with his is not trunk`) |
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
| `--min-consensus` | `0.0` (off) | Drop Q/A pair-hayi ke `answer_consensus` kamtar az in dare |
| `--no-fast-path` | off | Fast Path whitelist ro khamoosh kon — hame soal ha be LLM classifier miran (kondtar; baraye andaze-giri-e false positive-haye Fast Path) |
| `--no-resume` | off | Ignore classifier + LLM checkpoints (fresh start) |
| `--output` | `outputs/...` | Override path output JSON |

### Output validator + retry (two-tier)

Validator ru **hame** caption ha run mishe — rule-based ham mesl-e LLM (Comments8 band-e 7). Rule caption-i ke hard check ro rad kone, be `needs_llm` tabdil mishe ta LLM az no benevisad (`info.rule_validation_reject_count`).

Ghaide-ye asli: regex faghat vaghti reject mikone ke **ghat'i** ghalat bashe. Har chi faghat mashkuk ast → `validation_flags` roye row (row toye dataset mimoone) + escalate be Tier-2.

**Tier 1 — lexical / cheap (high precision reject)**

1. **Format** (`caption_format_is_valid`): ye jomle-ye declarative — na khali, na `?`, na bracket/quote, na chand jomle, na "the answer".
2. **Echo**: caption nabayad faghat khod-e soal ro tekrar kone.
3. **Polarity-e ghat'i motanaghez**: `yes` + caption-e negative (ya shoru' ba "No") → reject; `no` + caption-i ke sarih migeh "Yes" → reject. `no` + paraphrase-e bedoon negation (`Was this taken during the day?` + no → `It is taken at night.`) **reject nemishe** — flag `no_answer_without_negation` migire.
4. **Spurious negation**: javab-e non-yes/no + negation-e jomle-i. `no` toye ye noun phrase (`no parking sign`) negation hesab **nemishe**.
5. **Answer grounding-e verbatim**: proper noun / number / color / short answer bayad ayn-esh toye caption bashe (Loon≠Loom) → reject. Javab-haye tulani-tar → flag `answer_partial_match`.
6. **Batch contamination**.

**Flag ha (reject nemishan)**

| Flag | Ma'ni |
|------|-------|
| `relation_low` | Overlap-e stem-haye soal kam ast (`RELATION_MIN_RATIO = 0.5`) — mesl-e `bodies of water` vs `body of water` |
| `unsupported_facts_suspect` | Caption content word-e ezafe dare |
| `no_answer_without_negation` | Javab `no` vali caption negation nadare (mitune paraphrase-e dorost bashe) |
| `answer_partial_match` | Kamtar az 50% token-haye javab-e chand-kalame-i toye caption |

Note ru relation ratio (hala faghat flag): **≥50%** az stem-haye *required*-e soal. `required` = content stem-ha **menha-ye** do goruh ke javab jaye-shun ro migire:
   - **wh-category NP** — javab *jaye* esm-e daste ro migire, pas kalame-ye daste ejbari nist:
     - `What **animal** is this?` + dog → `This is a dog.`
     - `What **season** is it?` + summer → `It is summer outside.`
     - `What **sport** is shown here?` + skateboarding → `A skateboarding competition can be seen.`
     - `What **mode of transportation** is pictured?` + car → `A car is pictured.`
     Faghat shekl-e mostaghim-e `what/which/whose <NP>`; age ba'd az wh-word fe'l biyad (`What **is** on the table?`) hich chi hazf nemishe, pas table→chair hanooz reject mishe. Fe'l-haye depiction (`shown`/`pictured`/`seen`) stopword hastan ta synonym-e "shown" vs "seen" caption-e dorost ro drop nakone.
   - **either/or alternatives** — `Is the sun to the right **or** left of this flower?` hich vaght nemitune har do ro dashte bashe. Entekhab-e branch-e ghalat ba `answer_in_caption` gir mikhore.
   
   Stemming do bar suffix strip mikone, pas `buildings` → `building` → `build` ba `building` → `build` match mishe (ghablan `buildings`/`building` stem-e mokhtalef midadan va caption-e dorost `relation_mismatch` migereft).

**Tier 2 — Qwen semantic judge** (sample-haye mashkuk + har row-e flag-dar):

> Given QUESTION, ANSWER and CAPTION, return PASS only if the caption correctly expresses the answer to the question and adds no unsupported factual information. Otherwise return FAIL.

Ghazavat-e semantic **faghat** kar-e in judge ast, na regex.

**Retry policy:** FAIL → **1** regenerate (`single_retries=1`) → FAIL dobare → drop. Counts: `info.validation_retry_count`, `info.validation_failure_count`, `info.validation_flagged_count`.

**Final salvage:** leftover `needs_llm` **batched**, va har leftover ye single-item retry ham migire (`single_retries=1`), pas ye parse failure-e batch bedoon test-e tanha drop nemishe.

### Retry audit log (`*_validation_audit.jsonl`)

Ba `--llm`, har item-i ke retry shode — mohem nist accept shode ya na — ye record migire dar `outputs/v2_question_dependent_captions_{split}2014_validation_audit.jsonl`:

```json
{
  "question_id": 262148000,
  "question": "Is the sky blue?",
  "answer": "no",
  "stage": "main",
  "retry_kind": "validator",
  "first_caption": "Yes, the sky is blue.",
  "failure_reason": "polarity_mismatch",
  "retry_caption": "The sky is not blue.",
  "final_result": "accepted"
}
```

- `retry_kind`: `validator` (caption sakhte shod vali rad shod) ya `generation` (`parse_*` / `timeout` / `empty_response` / connection error).
- `final_result`: `accepted` ya `dropped`.
- `stage`: `main` ya `salvage`.

Ghablan faghat `validation_retry_count` bud, pas retry-e movafagh hich asari nemigozasht.

### Answer-consensus filter (`--min-consensus`)

**Off by default** (`0.0`) — low `answer_consensus` rows are kept for later down-weight experiments.

Ba `--min-consensus 0.4` har pair-i ke mode answer-esh kamtar az 40% annotator agreement dare drop mishe: vaghti khod-e adam-ha roye yek javab tavafogh nadaran, oon caption target-e ghabel-e etemadi baraye train nist. Tartib-e filter-ha sabet ast: **OCR → consensus → dedup** (consensus ghabl az dedup, ta pair-e drop-shode jaye dedup ro nagire).

Drop-ha kamel toye sidecar save mishan: `outputs/..._low_consensus.json` (ba `info.min_consensus`). Count → `info.low_consensus_excluded_count`, threshold → `info.min_consensus`.

Ru VQA v2 train ~11% pair-ha zir-e 0.4 hastan va **hame-shun** non-yes/no hastan (javab-e binary ba 10 annotator riyazi-yan nemitune zir-e 0.5 bere), pas in filter sahm-e yes/no ro dar dataset bala mibare — in trade-off ro dar nazar begir.

```bash
python generate.py --split train --llm --min-consensus 0.4 \
  --model qwen2.5:3b-instruct-q4_K_M --batch-size 10
```

Resume: age `--min-consensus` ba run-e ghabli fargh dashte bashe, checkpoint qabool nemishe va rows az no sakhte mishan (payam-e `Ignoring checkpoint: it was built with --min-consensus ...`).

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
| `echoes_question` | Caption just repeats the question |
| `polarity_mismatch` | `yes` answer with a negated caption, or `no` answer that explicitly says "Yes" |
| `semantic_fail` | Tier-2 Qwen judge returned FAIL |
| `empty_response` / `timeout` | Model returned nothing / timed out |

`relation_mismatch` and `unsupported_facts` are **no longer reject reasons** — they became the `relation_low` / `unsupported_facts_suspect` flags.

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
  "rule": "what_color",
  "visual_filter_source": "fast_path"
}
```

`rule` mishe yeki az: rule name ha (`what_color`, `how_many`, `yesno_are_all`, …), `needs_llm` (hanuz LLM nagerefte — `caption` khali), ya `llm_fallback` (LLM tolid karde).

`visual_filter_source` faghat ba `--classify-questions` neveshte mishe: `fast_path` (whitelist match kard, bedoon LLM) ya `llm_classifier` (Qwen label dad). Row-haye sidecar-e `*_not_directly_visual.json` ham hamin field ro daran, pas mishe did kodum filter chi ro rad karde.

`validation_flags` (age vojood dashte bashe) list-e moshkel-haye mashkuk ast; oon row ha toye dataset **mimoonan**.

`answer_count` = chand ta az 10 annotator dagigan hamun mode answer ro dadan; `answer_consensus` = `answer_count / total_annotators` (rounded). In annotator agreement ast, na model confidence — ba'dan mitune baraye loss weighting estefade beshe.

`info.llm` (age `--llm`): `model`, `batch_size`, `workers`, `host`, `prompt_version`, `validation`.

Accounting fields (bayad jam beshan):

| Field | Meaning |
|-------|---------|
| `input_count` | Q/A ids scanned at load |
| `ocr_excluded_count` | Regex OCR prefilter |
| `low_consensus_excluded_count` | `--min-consensus` drops (`0` when off) |
| `min_consensus` | Threshold used for this run |
| `duplicate_count` | Dedup drops |
| `post_filter_count` | Rows kept after OCR/dedup/classifier (for resume matching) |
| `directly_visual_count` | Classifier kept (ya hame rows age classify off) |
| `not_directly_visual_count` | Classifier dropped |
| `dropped_empty_count` | Empty/short (excluding counted validation failures) |
| `validation_retry_count` | Per-item regenerations |
| `validation_failure_count` | Final validation drops |
| `validation_flagged_count` | Rows **kept** with `validation_flags` |
| `rule_validation_reject_count` | Rule captions rejected by the validator and sent to the LLM |
| `num_samples` | Final annotations length |

Identity: `input ≈ ocr + low_consensus + duplicate + not_directly_visual + num_samples + dropped_empty + validation_failure` (va `directly_visual ≈ num_samples + dropped_empty + validation_failure`). `validation_flagged_count` va `rule_validation_reject_count` in identity ro **avaz nemikonan** — flag row ro drop nemikone va rule reject faghat row ro be LLM mifreste.

`info.llm.validation` = `{single_retries, salvage_single_retries, tier, validator_version, relation_min_ratio}` — `validator_version` har bar ke ghavanin-e accept/reject-e Tier-1 avaz beshan bump mishe, pas har output JSON mige ba kodum validator sakhte shode.

## QC validators (LLM)

Beyond format checks, accepted LLM captions must pass Tier-1 relation / verbatim / unsupported-facts checks and, when suspicious, Tier-2 PASS/FAIL. Prefer `--batch-size` ≤ 10.

## DIRECTLY_VISUAL filter

**Off by default.** Bedoon flag, `question_classifier.py` call nemishe.

`DIRECTLY_VISUAL` = soal ba negah kardan be tasvir javab dade mishe (object, rang, tedad, position, action, material, room/scene, sport/activity, weather, sen-e taghribi, expression) — **default hamin hast**.

`NOT_DIRECTLY_VISUAL` vaghti ke javab yeki az in ha ro lazem dare: **reading-e text-e ru-ye tasvir (OCR)**, **nazar/salighe-ye shakhsi**, **knowledge-e biruni** (ghavanin-e sport, legality, species facts, sazande/brand, gheymat, seda-ye heyvan), ya **ghezavat/ghasd** (`Is this safe?`, `Is this suitable?`, `Does this animal want to eat?`, `Is this place in a particular country?`).

### Fast Path = whitelist (Comments8 band-e 1)

Ghablan gate **visual-by-default** bud: har soal-i ke `_NON_VISUAL_SUSPECT_RE` ro match nemikard bedoon hich LLM call `FAST_PATH_VISUAL` mishod. Natije: ru 4000 sample, 3453 az 3623 soal fast-path shodan va classifier faghat 170 soal did — pas soal-haye safety / intention / country be file-e nahayi resid.

Hala Fast Path ye **whitelist** ast (`_FAST_PATH_VISUAL_RE`): match → `DIRECTLY_VISUAL` (`visual_filter_source=fast_path`); else → UNKNOWN → Qwen. Faghat in shape-ha (va bedoon marker-e `_NON_VISUAL_SUSPECT_RE`) fast-path mishan:

- colour: `what color/colour/colors …`
- count: `how many …`, `number of …`
- existence: `is/are there …`, `(do|can|…) you see …`
- scene / object class: `what animal(s)|shape|sport|game|activity|room|scene|place|food(s)|fruit(s)|dish …`
- sky: `is the sky …`
- spatial: `what is under|over|above|below|behind|beside|next to|in front of …`, plus plain `Is the cat on the table?` (end-anchored)
- action / attire (end-anchored): `What is the man doing/holding/wearing?`

**Nist** fast-path (mimune UNKNOWN → LLM): bare `what is/are/do/does`, `what kind/type`, `is he/she`, `where is`, `could this`, `does this look/appear`, `who is`, …

Baghie be Qwen miran (`v7_expanded_fast_path`). `Could this photo be from a zoo?` mitune bâz ham visual label bekhore, vali **hich vaght** fast-path nemishe.

`made of` suspect **nist** (material-e visible) vali `who made` hast (maker/brand knowledge). `text` / `says` / `words` OCR-suspect hastan. `can be seen` / `can you see` / `next to` / `on the right` / `trash can` / `city bus` / `can you spot` exempt hastan.

Hazine: LLM call-haye classifier az ~170 be ~2200 ru hamun 4000 sample mire (va ~3600 ba `--no-fast-path`) — kondtar, vali daghigh-tar; hamin trade-off khaste shode bud.

### Flags

| Flag | Chi mikone |
|------|------------|
| `--classify-questions` | Faghat whitelist-e Fast Path bedoon LLM label mikhore; baghie be Qwen miran. Drop + write sidecar. Resume via `*_classifier_checkpoint.json`. |
| `--no-fast-path` | Whitelist ro kollan khamoosh mikone — hame soal ha be Qwen miran (hame row ha `visual_filter_source = "llm_classifier"`). Baraye moghayese-ye ba/bedoon Fast Path. |
| `--classifier-checkpoint-every` | `50` | Save classifier progress every N questions |
| `--drop-subjective-candidates` | Offline: regex candidates drop (bedoon Qwen). |
| `--classifier-model` | Model Ollama baraye classifier; default = `--model` |

```bash
python generate.py --split train --llm --classify-questions \
  --model qwen2.5:3b-instruct-q4_K_M --batch-size 10
```

Sidecar: `outputs/v2_question_dependent_captions_{split}2014_not_directly_visual.json` — baraye tahlil-e ba'di. In filter **faghat** baraye dataset-e train-e Captioner ast; VQA2 asli baraye eval dastkhord nashavad.

Counts: `info.directly_visual_count`, `info.not_directly_visual_count`, `info.question_classifier.label_counts` (`FAST_PATH_VISUAL` = tedad-e row-haye `visual_filter_source == "fast_path"`), `info.question_classifier.fast_path_enabled`.

Note: `prompt_version` (`v7_expanded_fast_path`) avaz shode va checkpoint ba `fast_path_enabled` key mikhore, pas checkpoint-e ghadimi (masalan `v6_conservative_fast_path`) roye resume invalid hast — pak-esh kon ya `--no-resume` bede ta row-haye ghablan drop-shode dobare label bekhoran.

## Tests + audit

`audit/audit_captions.py` **ekhtiari** hast.

```bash
cd QuestionDependentCaptionGenerator
python audit/audit_captions.py outputs/v2_question_dependent_captions_train2014.json
```

Report shamel: `visual_filter_source_counts`, `rows_with_validation_flags` + `validation_flag_counts`, counter-haye jadid-e `info` (`validation_flagged_count`, `rule_validation_reject_count`), `fast_path_enabled`, va `accounting_input_vs_accounted` (bayad `difference: 0` bashe). Age row-i hanooz `rule` = `what_is` / `yesno_modal_have` dashte bashe (`retired_rule_rows`), yani file ghadimi'e va audit FAIL mide.

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
- Low-consensus samples default **hazf nemishan**; ba `--min-consensus T` drop mishan → `info.low_consensus_excluded_count` + sidecar `*_low_consensus.json`
- Baraye train captioner: faghat rows-e `caption` gheyr-khali
