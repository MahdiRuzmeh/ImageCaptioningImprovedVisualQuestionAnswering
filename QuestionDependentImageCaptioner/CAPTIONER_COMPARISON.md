# Question-dependent captioner comparison: BUTD + Transformer vs BLIP

Stage 1 of the thesis pipeline: given an image and a question, write a caption that answers the
question. Three captioners were trained on the same data and split, and tested on the same 9,997 rows:

| Model | Visual input | Language model | Trained params |
|---|---|---|---:|
| **A** `a_full_butd` | BUTD Faster R-CNN regions (frozen, cached) | Transformer encoder–decoder + copy, from scratch, 3 layers | 23.5M |
| **A-large** `a_full_butd_large` | same | same, 6 layers | 51.8M |
| **B** `b_blip_full` | BLIP ViT-B/16 patches (frozen) | BLIP BERT decoder, pretrained, fine-tuned with the question as prompt | 137.9M |

**Summary.**
- **BLIP is the best captioner on every metric:** AnswerHit 58.9% vs 46.3%, yes/no polarity 76.4% vs 59.2%, CIDEr 6.38 vs 5.94.
- **The bigger A model gives no gain** (6 vs 3 layers: CIDEr 5.96 vs 5.94, AnswerHit 46.5% vs 46.3%). Capacity is not the limit for A.
- **The data is not the bottleneck:** the ground-truth captions contain the answer in 98.0% of the test rows.
- **Model A depends strongly on both inputs.** Without the right image, AnswerHit falls from 46% to 14–16%, the same as a no-model answer prior. With another question about the same image, it answers the new question as well as its own question. Its weak spot is yes/no questions, where it barely uses the image.
- BLIP's grounding test is pending (Kaggle run, section 5).

---

## 1. Architectures

### Model A: BUTD regions + multimodal Transformer with copy (`models/captioner_mmtransformer_copy.py`)

```
image ──► BUTD Faster R-CNN (ResNet-101 C4, Visual Genome 1600 objects / 400 attributes, frozen)
          36 detected regions + 1 mean token = 37 × 2048 features + box (x1, y1, x2, y2, area)
          ──► LayerNorm + Linear 2048→512  (+ Linear box 5→512, + type embedding "region")
question ──► word embedding 512 (question vocab 4,763) + position + type embedding "question"

[ 37 region tokens ; question tokens ]
          ──► JOINT ENCODER: N pre-LN Transformer layers (d = 512, 8 heads, FFN 1024 / 2048)
              self-attention over regions AND question words together: in every layer each region
              attends to the question words and each question word attends to the regions (early fusion)
          ──► memory (regions + question, both question-aware)

caption so far ──► DECODER: N layers, each: causal self-attention → cross-attention to the memory → FFN
          ──► output = tied word embedding (caption vocab 7,029)
          ──► COPY HEAD (pointer-generator): attention over the question part of the memory;
              p_gen · P_vocab + (1 − p_gen) · P_copy(question word)
```

| | A | A-large |
|---|---|---|
| Encoder / decoder layers | 3 / 3 | 6 / 6 |
| d_model, heads, FFN | 512, 8, 1024 | 512, 8, 2048 |
| Trained params | 23.5M | 51.8M |
| Training | from scratch, 15 epochs, AdamW 3e-4, batch 128 (2× T4), label smoothing 0.1 | same |
| Best epoch (val CIDEr) | 15 | 15 |

The detector is frozen: its features were extracted once into a packed cache.

### Model B: BLIP-base with the question as a decoder prompt (`train_blip.py`)

```
image (384×384) ──► BLIP ViT-B/16 image encoder, FROZEN
                    12 layers, d = 768, 12 heads → 577 tokens (576 patches + CLS) × 768
                    (the image encoder never sees the question)

decoder input:  [DEC] question: <question> caption: <caption> [SEP]
                ──► BLIP text decoder (BERT-base, pretrained, fine-tuned)
                    12 layers, each: causal self-attention over the prompt + caption so far
                                     → cross-attention to the 577 image tokens → FFN 3072
                ──► LM head (WordPiece vocab 30,524); loss only on the caption tokens
```

| | B |
|---|---|
| Start point | `Salesforce/blip-image-captioning-base` (pretrained on image–text data, fine-tuned for COCO captioning) |
| Trained | text decoder only: 137.9M params (the ViT, ~86M, is frozen) |
| Training | 4 epochs, AdamW 3e-5, cosine schedule, batch 96 (2× T4), label smoothing 0.1 |
| Best epoch (val CIDEr) | 3 |

### The main design difference

| | A (BUTD + Transformer) | B (BLIP) |
|---|---|---|
| Visual units | 36 object regions with attribute-trained features | 576 grid patches |
| Where image and question meet | **in the encoder**: regions and question attend to each other in both directions in every layer, then the decoder cross-attends | **in the decoder only**: the question is a causal prefix, and each decoder layer cross-attends to image features that don't depend on the question |
| Language knowledge | none: learned from 264k QD captions | pretrained on large image–text data |
| Copying question words | explicit copy head | none needed (WordPiece covers the vocabulary) |

*Caveat on pretraining:* both visual front-ends were pretrained on data that overlaps COCO, so some test images may have been seen before:
- **BLIP:** its COCO captioning fine-tuning used COCO images, with generic human captions.
- **A's BUTD detector:** trained on Visual Genome, which contains about half of the COCO images, with object and attribute labels.

Neither has seen these questions or answers. Only the captioners trained here saw the QD captions, and their test images are disjoint from their training images.

---

## 2. Dataset and split

**Source:** `QuestionDependentCaptionGenerator/outputs/vqa_v2_question_dependent_captions_train2014.json`, built from VQA v2 train2014 questions and answers.
- 443,757 VQA questions → 283,707 kept as directly visual. The rest were not directly visual or needed OCR (15,824), plus 88 duplicates and 8 empty.
- A rule fills the caption from the question and answer (what_color 37,788, how_many 23,903, what_is_doing 4,940, who 628). The other 216,448 were written by an LLM (qwen2.5-3B).
- An LLM judge checked every caption:

| Judge verdict | ok | GRAMMAR_ISSUE_ONLY | WRONG |
|---|---:|---:|---:|
| First pass | 250,987 | 4,567 | 28,153 |
| After regenerating the 28,153 WRONG (26,476 fixed) | **277,318** | 4,712 | 1,677 |

**Used for training:** only rows with `caption_quality_verdict == "ok"`, i.e. **277,318 rows on 79,568 images**. Captions are short (5.7 words on average), and 32% of the answers are yes/no.

**Split** (`scripts/prepare_full_qd.py`): rows sorted by `image_id`; the last ~10k rows are test and the ~3k before them val. Boundaries are moved to image changes, so **no image appears in two splits**.

| Split | Rows | Share | Images |
|---|---:|---:|---:|
| Train | 264,318 | 95.3% | 75,872 |
| Val (checkpoint choice) | 3,003 | 1.1% | 860 |
| Test (all numbers below) | 9,997 | 3.6% | 2,836 |

### Answer in the ground-truth captions

| Rows | AnswerHit (non-yes/no) | YesNoPolarity (yes/no) |
|---|---:|---:|
| All 283,707 (any verdict) | 98.0% | 97.5% |
| **ok (277,318)** | **98.1%** | **97.5%** |
| GRAMMAR_ISSUE_ONLY (4,712) | 96.2% | 99.5% |
| WRONG (1,677) | 93.6% | 97.3% |
| Train / val / test | 98.1% / 97.8% / **98.0%** | 97.5% / 97.5% / **96.7%** |

The answers are in the data. The few misses are mostly the strict string rule, e.g. answer "none" vs "This person is not wearing a hat".

---

## 3. Metrics

| Metric | What it measures |
|---|---|
| BLEU-4, METEOR, ROUGE-L, CIDEr-D | how close the caption is to the ground-truth caption (wording). One reference per row; CIDEr on the ×10 scale used throughout this repo. METEOR = NLTK `meteor_score`. |
| **AnswerHit** | non-yes/no rows (6,639 test rows): share whose caption contains every word of the VQA answer (number words = digits, "0" = no/none) |
| **YesNoPolarity** | yes/no rows (3,358 test rows): share where the caption is negated (not / no / none / n't …) exactly when the answer is "no" |

AnswerHit and YesNoPolarity are the metrics that matter for VQA: they measure whether the caption carries the answer.

---

## 4. Results on the test split (9,997 rows, beam 3)

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | **AnswerHit** | **YesNoPolarity** |
|---|---:|---:|---:|---:|---:|---:|
| Ground-truth captions | — | — | — | — | 98.0% | 96.7% |
| A (3 layers) | 0.611 | 0.775 | 0.777 | 5.94 | 46.3% | 59.2% |
| A-large (6 layers) | 0.612 | 0.774 | 0.776 | 5.96 | 46.5% | 60.1% |
| **B (BLIP)** | **0.641** | **0.803** | **0.800** | **6.38** | **58.9%** | **76.4%** |
| Answer prior (no model, no image) | 0.000 | — | 0.091 | 0.13 | 14.1% | 49.2% |

### AnswerHit by question type

| Type | Rows | GT | A | A-large | BLIP |
|---|---:|---:|---:|---:|---:|
| LLM-written (all other questions) | 4,432 | 97.2% | 42.8% | 43.2% | **56.7%** |
| what color | 1,217 | 100.0% | 64.4% | 65.6% | **71.8%** |
| how many | 794 | 98.7% | 36.4% | 36.0% | **49.2%** |
| what is … doing | 171 | 100.0% | 52.0% | 48.0% | **66.1%** |
| who | 25 | 100.0% | 56.0% | 44.0% | **64.0%** |

| Yes/no polarity | answer = yes | answer = no | captions negated |
|---|---:|---:|---:|
| A | 54.8% | 62.7% | 54.5% |
| A-large | 59.7% | 60.8% | 51.2% |
| BLIP | **78.0%** | **75.1%** | 50.2% |

(53% of the yes/no test answers are "no".)

**Reading the results:**
- **BLIP leads everywhere**, by about +12 points of AnswerHit and +17 points of yes/no polarity. The largest gains are in counting (+13) and in the LLM-written open questions (+14).
- **The wording metrics are close** (CIDEr 5.9 vs 6.4): all models write in the dataset's style. They differ in **whether the stated answer is right**, which only AnswerHit and polarity show.
- **The extra capacity of A-large gives nothing.** Both A runs were still improving slowly at epoch 15 (val CIDEr +0.05 over the last 3 epochs). BLIP peaked at epoch 3 of 4.
- **Counting is the hardest type for all three models** (36–49%).

Per-row tables: `scripts/answerhit_table.py` → `kaggle_Image_captioner_BLIP_full/answerhit_table.csv`
(question, answer, GT caption, answer in GT, predicted caption, answer in prediction).

---

## 5. Visual and question grounding

How much does each model depend on the image and on the question? Every test row is decoded again with one input changed (`scripts/grounding_eval.py`, beam 3):

| Condition | Input | Meaning |
|---|---|---|
| normal | own image + own question | the test result above |
| **just question** (`no_image`) | uninformative image (A: mean of all test region features; BLIP: mean-gray image) + own question | what the model says from the question alone |
| **just image** (`no_question`) | own image + empty question | what the model says from the image alone |
| **same question, different image** (`swap_image`) | another test image + own question | does the caption change with the image? |
| **same image, different question** (`swap_question`) | own image + another question about the same image (9,524 rows on 2,363 images) | does the caption follow the question? |

Scores are against the original question's answer ("own"). For `swap_question` they are also against the new question's answer and caption ("follow"). "Changed" is the share of captions that differ from the normal caption.

### 5.1 Model A and A-large

| Condition | A AnswerHit | A Polarity | A CIDEr | A changed | A-large AnswerHit | A-large Polarity | A-large CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | **46.3%** | **59.2%** | 5.95 | — | **46.5%** | **60.1%** | 5.96 |
| just question (no image) | 15.5% | 53.0% | 4.74 | 67.0% | 16.2% | 51.2% | 4.78 |
| same question, different image | 14.1% | 51.4% | 4.80 | 66.6% | 14.2% | 48.7% | 4.75 |
| just image (no question) | 7.9% | 47.6% | 0.44 | 99.8% | 9.6% | 49.1% | 0.47 |
| same image, other question: own answer | 7.4% | 49.5% | 0.33 | 99.7% | 7.4% | 49.9% | 0.33 |
| same image, other question: **follows new question** | **46.5%** | — | — | — | **46.6%** | — | — |
| answer prior (no model) | 14.1% | 49.2% | 0.13 | | 14.1% | 49.2% | 0.13 |

**Dependence** (normal minus condition, AnswerHit points):

| | A | A-large |
|---|---:|---:|
| Image (normal − different image) | **32.2** | **32.3** |
| Image (normal − no image) | 30.8 | 30.3 |
| Question (normal − no question) | **38.4** | **36.9** |
| Image, yes/no polarity (normal − different image) | 7.8 | 11.4 |
| Question, yes/no polarity (normal − no question) | 11.6 | 11.0 |

**Diversity of the normal captions:**

| | A | A-large |
|---|---:|---:|
| Same image, different questions (2,363 images): distinct captions per image | 99.6% | 99.6% |
| Same question, different images (949 questions on ≥ 2 images): distinct captions | 72.8% | 73.2% |
| … questions whose answer differs between images (768): distinct captions | 74.5% | 75.3% |
| … AnswerHit on those rows | 58.0% | 58.8% |

**What it shows for model A:**
1. **It uses the image for open questions.** With the wrong image or no image, AnswerHit drops from 46% to 14–16%. That's the level of the no-model answer prior (14.1%), so all of A's gain over guessing comes from the image. The captions keep the question's form (CIDEr stays 4.7–4.8), but the content changes, e.g. *"they are playing tennis"* → *"they are playing polo"* with another image.
2. **It follows the question completely.** Given another question about the same image, it answers the *new* question as well as its own (46.5% vs 46.3%), and almost every caption changes (99.7%). Example: same image, *"What is the sport?"* → *"the sport is tennis"*; *"What color is the man's shirt?"* → *"the man's shirt is white"*. Without a question it writes a generic scene caption (*"a bus can be seen"*, *"people are present"*), which almost never contains the answer (7.9%).
3. **For yes/no questions it barely uses the image.** Polarity is 59% normally and 51% with another image, close to chance (49%). A mostly repeats the question as a statement and picks the polarity almost at random. Example: *"Can you see the person's face?"* (no) → *"you can see the person's face"* with or without the right image. This is model A's biggest weakness, and BLIP's largest advantage (76%).
4. **Same question, different images:** 73% of the captions differ between images, and they differ where the answers differ, e.g. *"What is in the background?"* → tree / bus / mountains / trees on different images. When the same question has different answers on different images, A gets 58% of them right.
5. **A and A-large behave the same** in every condition.

### 5.2 Model B (BLIP): pending

Run `scripts/kaggle_grounding_b.sh` on Kaggle (one T4, ~40 min), then put `grounding.json` in `kaggle_Image_captioner_BLIP_full/`. This section and the comparison below get the same tables as 5.1.

### 5.3 A vs BLIP grounding: pending

---

## 6. Conclusions so far

1. **Captioner choice: BLIP (model B).** It has the highest AnswerHit and polarity on every question type, it wins on all wording metrics, and it has already plateaued, so more training wouldn't change the ranking.
2. **The bottleneck is the model, not the data.** The ground-truth captions carry the answer 98% of the time; the best model carries it 59% of the time. The 2,631 test rows where the reference caption has the answer but BLIP's caption doesn't are mostly wrong counts, colors and objects.
3. **Model A is properly grounded but limited:** both inputs matter (image dependence 32 points, question dependence 38). It doesn't improve with more layers, and it can't judge yes/no from the image. It is a good from-scratch baseline for the thesis.
4. **Next, before VQA:**
   - finish BLIP's grounding run;
   - plan how captions are generated for the VQA training images so the VQA model never sees captions written by a captioner that was trained on those same images (cross-fitting).

---

### Files

| Model | Results folder | Contents |
|---|---|---|
| A | `a_full_butd/` | `best.pt`, `history.json`, `test_metrics.json`, `grounding.json` |
| A-large | `a_full_butd_large/` | same |
| B | `kaggle_Image_captioner_BLIP_full/` | `best.pt`, `train.log`, `test_metrics.json.json`, `answerhit_table.csv`, `grounding.json` (pending) |

Code is on branch `feat/captioner-comparison`:
- `scripts/grounding_eval.py`, `scripts/answerhit_table.py`, `scripts/results_table.py`, `metrics.py` (YesNoPolarity);
- `configs/eval_local.yaml`: model A evaluation on the 3070, with BUTD test features in `outputs/butd_packed_test`.
