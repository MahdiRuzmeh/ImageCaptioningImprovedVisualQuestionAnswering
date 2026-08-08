# QuestionDependentCaptionGenerator — Memari / Architecture

In sanad memari-e module-e **QuestionDependentCaptionGenerator** ro tozih mide (Finglish + English).  
This document explains the architecture of **QuestionDependentCaptionGenerator**.

**Hadaf / Goal:** Az joft-e `(soal, javab)` toye VQA v2 yek **caption-e soal-mehvar** besazim ke ba'dan target-e train-e Captioner bashe.

| Soal / Question | Javab / Answer | Caption |
|-----------------|----------------|---------|
| What color is the car? | red | The car is red. |
| Are the animals eating? | yes | The animals are eating. |
| Is there grass? | yes | There is grass. |

**Asl-e tarahi / Design principle:**  
`precision` mohem-tar az `coverage`-e Rule hast. Rule faghat vaghti sakhtar kamelan moshakhas-e ejra mishe; baghiye mire be LLM (Ollama). Hadaf-e pazhuhesh Rule-based boodan nist; hadaf **supervision-e ba-keyfiat va ghabel-e baz-tolid** hast.

---

## 1. Naghsh dar pipeline-e payan-name / Role in the thesis

```mermaid
flowchart LR
  VQA[VQA v2 Q+A] --> Gen[QuestionDependentCaptionGenerator]
  Gen --> Caps[Caption JSON]
  Caps --> CapTrain[Train SimpleImageCaptioner]
  CapTrain --> Downstream[VQA experiments]
```

Captioner-e Stage 1 faghat visual feature mibine va **OCR nadare**. Pas in generator:

1. Soal-haye text-reading (**OCR-dependent**) ro hazf mikone.
2. Caption-hayi misaze ke az rooye image + question ghabile yadgiri bashan.

---

## 2. Sakhtar-e file-ha / Module layout

```mermaid
flowchart TB
  subgraph cli [CLI]
    GEN[generate.py]
  end
  subgraph core [Core]
    RULES[caption_rules.py]
    PROMPT[llm_prompts.py]
    LLM[llm_client.py]
    CLS[question_classifier.py]
  end
  subgraph qc [QC]
    AUDIT[audit_captions.py]
    TESTS[tests/]
  end
  subgraph data [Data]
    VQA[(VQA JSON)]
    OUT[(outputs/*.json)]
  end
  VQA --> GEN
  GEN --> RULES
  GEN --> CLS
  GEN --> LLM
  LLM --> PROMPT
  GEN --> OUT
  AUDIT --> OUT
  TESTS --> RULES
  TESTS --> LLM
```

```
QuestionDependentCaptionGenerator/
├── generate.py              # CLI + orchestration-e kol-e pipeline
├── caption_rules.py         # filter-e OCR + motor-e Rule
├── llm_prompts.py           # packed prompt + few-shot
├── llm_client.py            # Ollama client + validator + retry
├── question_classifier.py   # filter-e soal-haye subjective / OCR
├── audit_captions.py        # audit-e keyfiat rooye JSON
├── tests/                   # unit test rooye bug-haye shenakhte-shode
├── architecture/            # hamin docs
└── outputs/                 # caption JSON (+ failure log)
```

| File | Kar / Responsibility |
|------|----------------------|
| `generate.py` | Load-e VQA, filter-ha, Rule, LLM, drop-e empty, save-e JSON |
| `caption_rules.py` | `is_ocr_question`, ghavanin-e rewrite, `generate_caption` |
| `llm_prompts.py` | System prompt-e version-dar (`PROMPT_VERSION`) |
| `llm_client.py` | Chat API, parse, polarity / grounding / contamination |
| `question_classifier.py` | Candidate-e regex → label-e Qwen-e 4-class |
| `audit_captions.py` | Shomaresh-e bug-haye baghimande bad az generate |

---

## 3. Jaryan-e dade entaha-be-entaha / End-to-end data flow

```mermaid
flowchart TD
  load[Load questions ∩ annotations] --> mode[Mode answer + answer_consensus]
  mode --> ocr[OCR filter]
  ocr -->|drop| ocrDrop[ocr_excluded_count]
  ocr --> dedup[Drop duplicate rows]
  dedup --> subjOpt{--classify-questions?}
  subjOpt -->|yes| cand[Subjective candidates regex]
  cand --> clf[Qwen 4-way classify]
  clf -->|non-VISUAL| subjDrop[Drop subjective / OCR]
  clf -->|VISUAL| rules
  subjOpt -->|no| rules[Rule engine]
  rules -->|safe match| row[Row ba caption]
  rules -->|na-motmaen| needs[needs_llm + caption khali]
  needs --> llmOpt{--llm?}
  llmOpt -->|yes| batch[Ollama batch max 10]
  batch --> val[Post-gen validators]
  val -->|reject| retry[Single-item retry]
  val -->|accept| llmRow[llm_fallback]
  retry -->|fail| needs
  llmOpt -->|no| keepEmpty[needs_llm mimone]
  row --> drop
  llmRow --> drop
  keepEmpty --> drop[Drop empty / needs_llm]
  drop --> out[Write outputs/*.json]
```

### Marhale-ha be tartib / Stages

1. **Mode answer** — Az 10 annotator, javab-e aksariat; save-e `answer_count` va `answer_consensus` (= count/10). In **tavaghof-e annotator** hast, na confidence-e model.
2. **Hazf-e OCR** — Ba regex + chand `question_type`. In sample-ha aslan vared-e rows nemishan.
3. **Dedup** — Faghat avalin `(image_id, question, answer)`.
4. **Ekhtiari: classifier-e soal** — Faghat rooye candidate-haye keyword; negah-dashtan-e `VISUAL`.
5. **Motor-e Rule** — Avalin ghanun-e daghigh; vagarna `needs_llm`.
6. **Ekhtiari: LLM** — batch + validate + retry.
7. **Hazf-e sakht** — Caption-e khali / kheili kootah / `needs_llm` toye file-e nahayi neveshte nemishe.

---

## 4. Memari-e motor-e Rule / Rule engine

**Bedoon catch-all.** Age parse motmaen nabashe → `None` → LLM.

### Flowchart-e tasmim-e Rule / Rule decision flow

```mermaid
flowchart TD
  q[Question + Answer] --> strat{caption_generation_strategy}
  strat -->|llm| needsLlm[needs_llm]
  strat -->|rule| tryRules[Try RULES be tartib]
  tryRules --> match{Rule match + safe?}
  match -->|yes| cap[Caption + rule name]
  match -->|no next| tryRules
  match -->|hich kodom| needsLlm
  tryRules --> families[Families: color / how_many / is_there / yesno_* / what_is / ...]
```

### Family-haye ghanun / Rule families

Ghavanin-e khastar-tar aval ejra mishan (rang, tedad, type, who, vojud-i, yes/no-e takhasosi, ba'd predicate va what-is).

| Family | Mesal | Note |
|--------|-------|------|
| Attribute | `what_color`, `what_kind_type` | Faghat pattern-e ghat'i |
| Existential | `is_there`, `are_there` | Shamel-e esm bedoon article (`Is there grass?`) |
| Yes/No takhasosi | anyone / any / all / both / this_a / … | Shape-e narrow |
| Yes/No omoomi | `yesno_is_are_predicate` | NP-haye morakab-e na-motmaen → LLM |
| Wh- | `what_is`, `who` | Aval `made of` / `used for` |
| Hamishe LLM | Does/Do/Did, Is/Are-e pichide | Az `caption_generation_strategy` |

### Tor-e imeni / Safety net

`can_generate_safe_rule_caption` template-haye kharab ro rad mikone:

- `The there …`, `The the …`
- `… made of is …`
- `the answer is …`

### Routing

| Function | Asar |
|----------|------|
| `should_use_llm_for_does_do` | Hamishe LLM |
| `is_complex_is_are_question` | Jomle-haye pichide → LLM |
| `should_use_llm_for_who` | Who-e gheyr-sade → LLM |
| `caption_generation_strategy` | `"rule"` vs `"llm"` |

---

## 5. Memari-e LLM fallback

```mermaid
flowchart TD
  needs[needs_llm rows] --> pack[Pack batch size leq 10]
  pack --> ollama[Ollama chat]
  ollama --> parse[Parse JSON array]
  parse --> fmt[Format check]
  fmt --> pol[Yes polarity check]
  pol --> ground[Answer / question grounding]
  ground --> contam[Batch contamination check]
  contam -->|ok| accept[llm_fallback]
  contam -->|fail| single[Single-item retry x3]
  single -->|ok| accept
  single -->|fail| salvage[Final salvage]
  salvage -->|fail| log[llm_failures.log]
  log --> dropLater[Drop az output-e nahayi]
```

### Prompt (`llm_prompts.py`)

- Version-dar baraye reproducibility.
- Yek jomle-ye ezhari, vafadar be javab, bedoon tekrar-e soal va bedoon "the answer is".
- Chand few-shot + batch-e shomare-dar → array-e JSON az caption-ha.

### Validator-ha (`llm_client.py`)

| Check | Chi rad mishe |
|-------|---------------|
| Format | Khali, kootah, soal, chand-jomle, bracket |
| Yes polarity | Javab yes vali caption manfi-ye vazeh (magar khod-e soal manfi bashe) |
| Grounding | Naboodan-e token-haye javab; baraye yes/no overlap ba kalame-haye soal |
| Spurious negation | Javab gheyr-e yes/no vali caption manfi-ye sakhtagi |
| Contamination | Jabeja shodan-e caption beyn-e item-haye yek batch |

**Batch size-e pishfarz 10 hast** ta khatar-e gati shodan-e sample-ha kam bemone.

---

## 6. Laye-haye filter-e keyfiat / Quality gates

```mermaid
flowchart LR
  inQ[VQA Q+A] --> g1[OCR filter]
  g1 --> g2[Subjective classifier]
  g2 --> g3[Rule safety]
  g3 --> g4[LLM validators]
  g4 --> g5[Empty drop]
  g5 --> clean[Clean caption set]
```

| Gate | Zaman | Natije |
|------|-------|--------|
| OCR | Hamishe | Hazf-e soal-haye text-reading |
| Subjective classifier | Ba `--classify-questions` | Hazf-e shakhsi / zehni / OCR-e classifier |
| Rule safety | Hamishe | Template-e kharab → LLM |
| LLM validators | Ba `--llm` | Rad / regenerate |
| Empty drop | Hengam-e neveshtan | Hich target-e khali vared-e train nemishe |

---

## 7. Ghalab-e khorooji / Output schema

```json
{
  "info": {
    "num_samples": 1205,
    "ocr_excluded_count": 75,
    "dropped_empty_count": 676,
    "subjective_excluded_count": 43,
    "rule_counts": { "...": "..." },
    "llm": { "model": "...", "batch_size": 10, "prompt_version": "..." },
    "question_classifier": { "model": "...", "prompt_version": "..." }
  },
  "annotations": [
    {
      "question_id": 9001,
      "image_id": 9,
      "question": "What color are the dishes?",
      "answer": "pink and yellow",
      "answer_count": 3,
      "answer_consensus": 0.3,
      "caption": "The dishes are pink and yellow.",
      "rule": "what_color"
    }
  ]
}
```

`answer_consensus` = tavaghof-e annotator-ha (mesalan 3 az 10 → `0.3`). Ba'dan mitune baraye vazn-dehi-e loss estefade beshe.

---

## 8. Amaliat va baz-tolidpaziri / Operations

| Mozu | Raftar |
|------|--------|
| Resume | Hamoon dastur-e `--llm` az checkpoint edame mide |
| Checkpoint | Save-e atomic har N batch; Ctrl+C ham save mikone |
| Failure log | `*.json.llm_failures.log` ba dalil-e khata |
| Audit | `python audit_captions.py outputs/....json` |
| Tests | `python -m unittest tests.test_qc_fixes -v` |

### Pilot-e pishnahadi ghabl az kol-e train (~443 hezar)

```bash
python generate.py --split train --llm --max-items 25000 --batch-size 10 \
  --classify-questions --model qwen2.5:3b-instruct-q4_K_M \
  --checkpoint-every 50 --output outputs/pilot_25k.json
python audit_captions.py outputs/pilot_25k.json
```

Bad az barresi-ye dasti-ye sample-haye Rule va LLM, version-e generator ro freeze konid.

---

## 9. Jam-bandi-ye dalayel-e tarahi / Design rationale

1. **Rule baraye ghat'iyat, LLM baraye ebham** — ta khata-ye dasturi-ye systematic kam beshe.
2. **Filter ghabl az supervision** — OCR va soal-e zehni signal-e train ro alude mikonan.
3. **Validator ba'd az LLM** — faghat mahdudiat-e prompt kafi nist (bargardandan-e polarity, gati-e batch).
4. **Hargez target-e khali naferest** — `needs_llm`-e baghimande drop mishe.
5. **Metadata-ye elmi** — consensus, shomaresh-e hazf-ha, model va version-e prompt baraye reproducibility-e payan-name.
