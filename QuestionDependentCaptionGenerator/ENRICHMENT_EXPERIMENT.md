# Experiment: COCO-context enriched QD captions (step 1)

Branch: `feat/qd-captions-coco-context` · Date: 2026-10-29

## Question

The question-dependent (QD) captioner trained on ~10k rows did not improve VQA
(CaptionVQA model C: 45.64 with captions vs 46.29 without). Human COCO captions
*do* help (48.89). Is the problem the **caption targets**?

## Diagnosis

Most QD captions are the question and its answer rewritten as one sentence
("What color are the dishes? / pink and yellow" → "The dishes are pink and yellow.").
They contain nothing beyond the VQA label: only **10%** of their content words are
not in the question. A captioner trained on them learns to guess the answer, and
the VQA model already does that from the question.

## Method — enrichment (`enrich_captions.py`)

For every QD row, a local LLM rewrites the caption, adding scene detail taken
**only** from the 5 human COCO captions of the same image.

- LLM: `qwen2.5:3b-instruct-q4_K_M` via Ollama (`OLLAMA_NUM_PARALLEL=3`, `--llm-parallel 3`),
  prompt `enrich_v1_coco_context` (system prompt + 3 few-shot turns).
  In a 300-row pilot qwen 3B did better than `mistral:latest`.
- Two attempts per row (temperature 0.0, then 0.7). The output is kept only if all deterministic checks pass:
  - the answer is still in the caption
  - yes/no polarity is unchanged
  - ≥ 60% of the base caption's content words are kept
  - every added content word appears in the COCO captions (no hallucination)
  - no conflicting number for counting questions
  - no meta words ("image", "photo", …).
- Otherwise the original caption is kept (`base_fallback` / `base_no_detail`).
- Resumable: progress is written to `<out>.progress.jsonl` one row at a time.
- Output: `src/dataset/vqa_v2_qd_captions_enriched_train2014.json`, with the same rows and
  order as the original file plus the fields `caption_base`, `enrich_status`, `enrich_added` and `enrich_rejects`.
  Stats: `audit/score_enrichment.py`.

Examples:

| Question / answer | Base caption | Enriched caption |
|---|---|---|
| What color are the dishes? / pink and yellow | The dishes are pink and yellow. | Pink and yellow dishes sit on a colorful tray. |
| Are both people wearing jeans? / yes | Both people are wearing jeans. | Two people wearing jeans sit on a bench with their legs crossed. |
| What is the player holding in his hand? / bat | A bat is being held in the player's hand. | A baseball player holds a bat in his hand ready to swing. |

### Dataset result (11,626 rows)

| | Value |
|---|---|
| enriched | 7,842 (67.5%) |
| base_fallback / base_no_detail | 3,648 / 136 |
| enriched rate by type | color 70%, number 60%, other 68%, yes/no 70% |
| content words not from question | 0.10 → 0.48 |
| mean length (words) | 5.6 → 10.4 |

## Captioner training

`QuestionDependentImageCaptioner/configs/kaggle_mini_enriched.yaml` is an exact copy of `kaggle_mini`
except for the caption file and `max_caption_len: 32`. It uses the `MMTransformerCopyCaptioner`
(21.7M params) with the image-disjoint split:

- train rows 0–9500 (9,502 rows, 2,895 images)
- val 9500–10000 (504 rows)
- test 10000–10500

Training: 20 epochs, batch 32, lr 3e-4. The best epoch by val CIDEr was 16.
Checkpoint: `outputs/kaggle_mini_enriched/best.pt`.

## Results

### Captioner probe (test rows, `scripts/captioner_probe.py`)

| Metric | Old QD captioner | Enriched captioner |
|---|---|---|
| AnswerConsistency (all) | 0.362 | 0.360 |
| AnswerHit | 0.281 | **0.300** |
| PolarityAcc (yes/no) | **0.582** | 0.522 |
| CountAcc (number) | **0.293** | 0.217 |
| AC color / other | 0.247 / 0.287 | **0.274 / 0.349** |
| caption words from question | 0.71 | 0.60 |
| CIDEr vs COCO human captions | 0.21 | **0.30** |

Generated captions are short: 7.1 words on average, vs 10.4 in the enriched targets.

### Answer coverage on the CaptionVQA test set (`CaptionVQA/caption_stats.py`)

% of questions whose answer appears in the caption (beam 1 / any of the 3 beams):

| | Old | Enriched |
|---|---|---|
| all | 17.8 / 31.8 | 17.2 / 26.6 |
| yes/no | 10.1 / 22.9 | 7.2 / 14.0 |
| number | 27.2 / 51.6 | 20.7 / 35.5 |
| other | 21.0 / 33.2 | 23.6 / 33.6 |

### VQA accuracy (CaptionVQA small, test, mean ± std over 3 seeds)

Config `CaptionVQA/configs/small_enriched.yaml`, script `CaptionVQA/scripts/run_enriched.sh`.

| Model | No caption | Old QD captions | **Enriched** | Human COCO |
|---|---|---|---|---|
| A (gated attention) | 46.29 ± 0.16 | 45.89 ± 0.58 | 46.14 ± 0.22 | 46.70 ± 0.48 |
| B (single-stream) | 45.36 ± 0.21 | 45.61 ± 0.21 | 45.32 ± 0.13 | 46.32 ± 0.38 |
| C (classifier + pointer) | 46.29 ± 0.16 | 45.64 ± 0.49 | **46.22 ± 0.41** | 48.89 ± 0.19 |

Model C by answer type (yes/no · number · other): no caption 67.2 · 31.9 · 34.6;
old 66.8 · 31.2 · 33.8; **enriched 66.9 · 33.1 · 34.4**; human 66.5 · 39.2 · 38.4.

Accuracy drop when captions are shuffled between questions (how much the model relies on them):

| Model | Old | Enriched | Human |
|---|---|---|---|
| C | 1.98 | 0.85 | 5.73 |
| B | 11.12 | 4.11 | 7.70 |

## What we learned

1. **Enrichment removes the harm, but adds no gain.** Model C goes from −0.65 to −0.07 vs no
   captions, which is within noise. Number questions improve slightly (+1.1 vs no caption, +1.9 vs old).
2. **Better targets alone are not enough.** With 9.5k training rows the captioner cannot learn
   the added scene detail. Its captions stay short (7 words) and generic, and the VQA model
   barely uses them (shuffle drop 0.85 vs 5.73 with human captions).
3. **There is a trade-off inside the captioner.** Enriched targets improve "other"/color grounding and
   COCO-likeness, but hurt yes/no polarity and counting. The answer is now a small part of a
   longer sentence, and the 3 beams vary less in the answer they give.
4. **The VQA side can use good captions.** Human captions give +2.6 (+7.3 on numbers) with the
   same models, so the captioner is the bottleneck, not the VQA architecture.
5. **Practical notes:**
   - LLM enrichment of 11.6k rows takes hours locally (≈0.8 rows/s with qwen 3B, parallel 3).
     For 300k rows, run it on more GPUs or in chunks with `--start/--max-items`.
   - The deterministic checks rejected about 1/3 of LLM outputs, so they are essential.
   - Do not compare CIDEr across different reference sets: the enriched captioner's val CIDEr (1.65)
     cannot be compared with the old one.

## Next: step 2 (COCO pretraining)

Pretrain the captioner on COCO human captions (81k train2014 images, 406k captions, QD val/test images
excluded), then fine-tune on the same QD split using the **original** QD captions. Enriched captions are
not part of step 2 for now; `configs/kaggle_mini_ft_enriched.yaml` stays available for later.
Files:
- `QuestionDependentImageCaptioner/scripts/build_coco_pretrain.py`
- `configs/kaggle_coco_pretrain.yaml`
- `configs/kaggle_mini_ft_base.yaml`
- `scripts/kaggle_step2.sh`.
