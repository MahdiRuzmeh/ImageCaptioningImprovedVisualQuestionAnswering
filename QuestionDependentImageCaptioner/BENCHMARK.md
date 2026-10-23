# QD captioner — architecture benchmark

Comparison of the current question-dependent captioner (v1) with two new architectures, on a 3000-item chunk of `vqa_v2_question_dependent_captions_train2014.json`.

**Recommendation: arch B, `MMTransformerCopyCaptioner`** (branch `feat/qd-captioner-transformer-copy`). It ties A for the best caption quality, has the best Answer-Hit of the two copy models, and trains fastest. However, no architecture solves the main problem: the answer word is still mostly guessed (see [What the numbers mean](#what-the-numbers-mean)).

## Setup

| | |
|---|---|
| Data | Rows sorted by `image_id`, split so no image appears in two splits: train rows 0–2999 (905 images), val 503 rows (153 images), test 499 rows (145 images) |
| Features | Same frozen Faster R-CNN, 32 regions per image, cached once for all models |
| Budget | 30 epochs, batch 16, grad clip 2.0, label smoothing 0.1, AMP |
| Selection | Checkpoint with the best **val CIDEr-D** (greedy decode every 2 epochs) |
| Test decode | Beam 3, length penalty 0.7, trigram blocking |
| Seeds | 42, 43, 44; tables show mean ± std |
| Optimizer | Adamax 5e-4 for the LSTM models (as in v1); AdamW 3e-4 with 400 warmup steps for the Transformer |

Branches: `feat/qd-captioner-benchmark` (shared harness), `feat/qd-captioner-updown-copy` (A), `feat/qd-captioner-transformer-copy` (B). B and the harness are merged into `feat/question-dependent-img-captioner-v2`, where B is the default `model_class`.

**The old smoke "val" set was training data.** `configs/smoke.yaml` uses val = the first 500 training rows. `outputs/smoke/best.pt` is included below only as a reference. Its test images are unseen, but it was selected on training rows.

## Architectures

- **v1 `QuestionDependentImageCaptioner`:** GRU question vector, one question attention, concat fusion, and an LSTM with per-step attention whose query is only `h`.
- **v1 + question vocab (`configs/bench_v1_qvocab.yaml`):** the same v1 model, but its caption vocab is built from captions + question words, like A and B. It still can't copy. This isolates how much of A and B's gain comes from the vocab change.
- **A `UpDownCopyCaptioner`:**
  - BiGRU per-word question states.
  - Two-layer Top-Down LSTM.
  - Additive region attention whose query includes both the decoder state and the question.
  - Attention over individual question words.
  - Pointer-generator copy from the question.
- **B `MMTransformerCopyCaptioner`:**
  - Region tokens and question tokens go through a 3-layer joint Transformer encoder.
  - 3-layer causal decoder.
  - Output layer shares weights with the word embeddings.
  - Copy head over the encoded question tokens.

Loss is the same for all: cross-entropy with label smoothing 0.1. For the copy models it's applied to the mixed (vocab + copy) distribution.

## Results (test, 499 items)

| Model | Params | BLEU-1 | BLEU-4 | **CIDEr-D** | ROUGE-L | **Answer-Hit** | Train s/epoch | Decode items/s |
|---|---|---|---|---|---|---|---|---|
| v1 (smoke `best.pt`, reference) | 18.3M | 0.466 | 0.150 | 1.306 | 0.470 | 0.155 | – | 96 |
| v1 (retrained) | 16.2M | 0.482 ± 0.008 | 0.180 ± 0.008 | 1.528 ± 0.062 | 0.486 ± 0.005 | 0.171 ± 0.015 | 7.1 | 93 |
| v1 + question vocab | 16.3M | 0.474 ± 0.008 | 0.178 ± 0.006 | 1.489 ± 0.028 | 0.484 ± 0.002 | 0.158 ± 0.005 | 6.7 | 107 |
| A UpDown + copy | 23.6M | **0.676 ± 0.007** | **0.427 ± 0.016** | **4.078 ± 0.128** | **0.672 ± 0.003** | 0.138 ± 0.018 | 11.7 | **440** |
| **B Transformer + copy** | 21.1M | 0.666 ± 0.007 | 0.418 ± 0.008 | **4.073 ± 0.082** | 0.656 ± 0.010 | **0.179 ± 0.008** | **6.7** | 181 |

- Answer-Hit is the share of captions that contain the gold answer. Number words count as digits ("two" = "2"), and "no"/"none" count as "0". It's measured on the 341 test items whose answer isn't yes/no (a caption doesn't state yes/no).
- v1's decode is slow because its beam search blocks repeated trigrams in a Python loop. A and B use the vectorized version in `models/beam_utils.py`.
- Best epochs: v1 18–30, v1 + question vocab 12–30, A 16–30, B 12–18. B converges fastest.

### By question type

| Model | CIDEr how_many (49) | CIDEr what_color (64) | CIDEr other (377) | Answer-Hit how_many | Answer-Hit what_color | Answer-Hit other |
|---|---|---|---|---|---|---|
| v1 | 0.42 | 1.73 | 1.55 | 0.143 | 0.188 | 0.164 |
| v1 + question vocab | 0.63 | 1.43 | 1.52 | 0.143 | 0.182 | 0.148 |
| A | 2.19 | 6.00 | 4.03 | 0.129 | 0.156 | 0.135 |
| B | **2.44** | **6.21** | 3.93 | **0.156** | 0.182 | **0.180** |

"Other" is the `llm_fallback` rule. `what_is_doing` has only 9 test items and is left out. CIDEr is computed within each group.

## What the numbers mean

0. **The vocab change alone does nothing.** Giving v1 the same caption + question vocab as A and B leaves it at the same level: CIDEr-D 1.49 ± 0.03 vs 1.53 ± 0.06, and Answer-Hit 15.8% vs 17.1%. Both differences are within seed noise. Without a way to copy, v1 still drifts off the question (e.g. "Are there any people in the stands?" → "there is one ball in the background"). So the jump to about 4.07 comes from the copy mechanism, not the vocab.

1. **Copying is the big win.** CIDEr-D goes from 1.53 to 4.07, and BLEU-4 from 0.18 to 0.42, for both copy models. That's more than 20 standard deviations of the seed noise. About 69% of caption tokens appear in the question, and the copy models now restate the question's subject reliably. v1 often drifts into an unrelated scene:

   | Question | v1 | A | B |
   |---|---|---|---|
   | Are there any people in the stands? (gt: There are no people in the stands.) | there are no people | no people are in the stands | no people are in the stands |
   | What is the clock made of? (metal) | the building is made of wood | the clock is made of wood | the clock is made of brick |
   | Is there tall buildings in the picture? (no) | a tree is in front of the picture | there is no buildings in the picture | there is no buildings in the picture |

2. **No model reliably produces the answer.** Answer-Hit is 14–18% for every architecture, and the differences from v1 are within about 2 standard deviations.
   - On counting questions, all models say "two" in 60–70% of captions.
   - Always saying "two" would score 22% on counting. All the models score below that.
   - So the answer is a learned guess from common answers, not read from the image.
   - The likely limits are the visual features and the data size: torchvision COCO Faster R-CNN box features with no attribute training, and 905 training images. The decoder isn't the limit.

3. **A vs B.** CIDEr-D is the same within noise (4.078 vs 4.073). B is ahead on Answer-Hit (+4.1 points, about 2 standard deviations), converges in fewer epochs, trains about 1.7× faster per epoch, and has fewer parameters. A sometimes produces broken copies ("the all one is house", "a scissors is in be cut with the scissors"). A decodes faster, but decoding speed doesn't matter for Stage 2 at this size.

## Recommendation and next steps

Use **B (`MMTransformerCopyCaptioner`)** as the Stage-1 architecture. It keeps the same interface as v1 (`generate_caption`, `encode_caption`, `get_caption_embedding`), so VQA Stage 2 can load it through `models.build_captioner`.

To raise Answer-Hit, which is what VQA actually needs from the caption, the next experiments should change the visual input and the data size, not the decoder:
- Use visual features trained with attributes (Visual Genome bottom-up features) or CLIP grid/patch features in place of the COCO box-head features.
- Train on all 11.6k rows (or more), which is ~4× the data.
- Consider rewarding Answer-Hit directly (SCST with a CIDEr + Answer-Hit reward) once the features can carry the answer.

## Caveats

- A and B build their caption vocab from captions + question words (needed for copying), and the main v1 row uses captions only. The "v1 + question vocab" row tests this directly: the vocab change alone gives no gain.
- Optimizer and dropout follow each model family's usual settings (Adamax with dropout 0.3 for the LSTMs; AdamW with dropout 0.2 for the Transformer), so they aren't identical across models.
- The test set is 499 items from 145 images, with one reference caption each. CIDEr with a single reference is noisy, which is why the tables report the spread over 3 seeds.

## Reproduce

Run from `QuestionDependentImageCaptioner/`:

```bash
python scripts/cache_regions.py --config configs/bench_v1.yaml
bash scripts/run_bench.sh configs/bench_v1.yaml v1                              # on feat/question-dependent-img-captioner-v2
bash scripts/run_bench.sh configs/bench_v1_qvocab.yaml v1_qvocab                # on feat/question-dependent-img-captioner-v2
bash scripts/run_bench.sh configs/bench_transformer_copy.yaml transformer_copy  # on feat/question-dependent-img-captioner-v2
bash scripts/run_bench.sh configs/bench_updown_copy.yaml updown_copy            # only on feat/qd-captioner-updown-copy
python scripts/bench_compare.py --root outputs/bench --out outputs/bench/summary.md
```

Outputs are written to `outputs/bench/<arch>/seed<k>/` (`best.pt`, `history.json`, `run_config.yaml`, `test_metrics.json` with all predictions).
