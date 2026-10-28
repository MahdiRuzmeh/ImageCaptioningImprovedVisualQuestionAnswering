# CaptionVQA on data the captioner was trained on ("seen" benchmark)

[benchmark.md](benchmark.md) tested A/B/C on images the captioner **never saw** (captions contain the answer only ~18% of the time). This report repeats the test on the images (and image–question pairs) the captioner **was trained on**. There the captions contain the answer much more often. This is the leakage condition the base papers may have had (benchmark.md §6), and it shows what question-dependent captions do for VQA when they carry the answer.

Date: 2026-10-28. Captioner: the same frozen `QuestionDependentImageCaptioner/outputs/kaggle_mini/best.pt`. Models, training and metric are unchanged (benchmark.md §1–2). 3 seeds, mean ± std.

## 1. Setup

| | |
|---|---|
| Image pool | The 2,895 COCO train2014 images of the captioner's training rows (QD JSON rows 0–9500, `kaggle_mini.yaml`) |
| Questions | **All** VQA v2 questions on these images, with their 10 human answers |
| Split (by image) | train 11,149 q / 2,045 img · dev 1,437 q / 250 img · test 3,107 q / 600 img |
| `qd_pair` | The (image, question) pair is one of the 9,502 rows the captioner was trained on. 61% of test questions (1,903). The other 1,204 (`other_q`) are questions on the same images that the captioner never saw (mostly questions the QD caption generator filtered out) |
| Region features | Copied from the captioner's region cache (the exact features it was trained on) |
| Unseen test | The test split of benchmark.md: 10,038 questions on val2014 images the captioner never saw |

Split files: `outputs/seen/split/*.json`. Reproduce: §7.

## 2. How often does the caption contain the answer?

Same measure as benchmark.md §5 (VQA soft accuracy of the best caption n-gram, n ≤ 3), seen test:

| Subset | n | Beam-1 caption | Any of 3 beams |
|---|---|---|---|
| **qd_pair (captioner trained on this pair)** | 1,903 | **33.9** | **51.6** |
| ┣ number | 329 | 38.9 | 66.7 |
| ┣ other | 1,021 | 40.4 | 55.7 |
| ┗ yes/no | 553 | 18.7 | 35.0 |
| other_q (same images, new question) | 1,204 | 6.4 | 12.6 |
| all seen test | 3,107 | 23.2 | 36.5 |
| *unseen test (benchmark.md §5)* | *10,038* | *17.8* | *31.8* |

On its own training pairs, the caption contains the answer about twice as often as on unseen images (33.9 vs 17.8). It still does not memorize: beam 1 is wrong for two out of three trained pairs (the Stage-1 AnswerHit on these rows is 0.36, `QuestionDependentImageCaptioner/VISUAL_GROUNDING.md`). On new questions about the same images the captions are nearly useless (6.4), because they describe what the captioner was trained to say, not what the question asks.

## 3. Experiments

| | VQA trained on | Tested on | Question |
|---|---|---|---|
| **E1** | unseen images (the 40k benchmark.md runs, no retraining) | seen test | Do models trained with weak captions profit when the captions get good? |
| **E2** | seen train (captions often contain the answer) | seen test | Leakage on both sides, like the base papers |
| **E3** | seen train (E2 models) | unseen test | What happens when such a model meets honest captions? |

E2/E3 models are trained on 11k questions instead of 40k, so their absolute accuracy is about 5–7 points lower than the 40k runs. Compare **with vs without caption inside one row of experiments**, not across them.

## 4. Results

### E2 — trained and tested on captioner-seen data (main result)

| Model | Caption | Overall | Yes/No | Number | Other | qd_pair | other_q | Shuffled caption |
|---|---|---|---|---|---|---|---|---|
| A | yes | 39.35 ± 0.19 | 63.03 | 26.35 | 25.92 | 39.79 ± 0.03 | 38.66 ± 0.49 | 39.01 (−0.3) |
| B | yes | 40.26 ± 0.29 | 64.90 | 26.75 | 26.29 | 40.76 ± 0.83 | 39.49 ± 0.58 | 32.24 (−8.0) |
| **C** | yes | **41.96 ± 0.93** | 64.72 | **29.79** | **28.98** | **44.59 ± 1.65** | 37.82 ± 0.33 | 32.49 (**−9.5**) |
| A = C | no | 39.36 ± 0.14 | 63.55 | 24.36 | 26.07 | 39.99 ± 0.57 | 38.37 ± 0.59 | – |
| B | no | 39.37 ± 0.90 | 63.15 | 26.65 | 25.79 | 39.34 ± 1.25 | 39.40 ± 0.39 | – |

Gain from the caption (caption − no caption):

| Model | Overall | qd_pair | ┣ number | ┣ other | ┗ yes/no | other_q |
|---|---|---|---|---|---|---|
| A | −0.01 | −0.20 | +2.45 | −0.56 | −1.14 | +0.29 |
| B | +0.89 | +1.42 | +0.39 | +1.13 | +2.56 | +0.09 |
| **C** | **+2.60** | **+4.60** | **+7.07** | **+5.13** | **+2.13** | **−0.55** |

C's pointer copies the answer from a caption for 30% of test questions (13% in the unseen benchmark). Its fusion gate falls to 0.22 (0.31 unseen).

### E1 — models trained on unseen images, tested on seen data

| Model | Caption | Overall | qd_pair | other_q | Shuffled caption |
|---|---|---|---|---|---|
| A | yes | 44.60 ± 0.60 | 45.15 | 43.73 | 44.52 (−0.1) |
| B | yes | 44.15 ± 0.19 | 44.32 | 43.87 | 33.40 (−10.8) |
| C | yes | 44.98 ± 0.12 | 45.09 | 44.81 | 42.33 (−2.7) |
| A = C | no | 44.71 ± 0.29 | 45.21 | 43.93 | – |
| B | no | 44.29 ± 0.83 | 44.71 | 43.63 | – |

Gain from the caption: A −0.11, B −0.14, C +0.27 overall. On qd_pair: A −0.06, B −0.39, **C −0.12**. The captions now contain the answer twice as often, but these models gain nothing.

### E3 — models trained on seen data, tested on unseen images

| Model | Caption | Overall | Yes/No | Number | Other | Shuffled caption |
|---|---|---|---|---|---|---|
| A | yes | 41.23 ± 0.46 | 65.53 | 28.63 | 26.55 | 41.12 |
| B | yes | 40.53 ± 0.26 | 65.20 | 27.66 | 25.64 | 33.90 |
| C | yes | 40.45 ± 0.54 | 65.81 | 29.51 | 24.54 | 34.37 |
| A = C | no | 41.07 ± 0.35 | 65.26 | 29.66 | 26.16 | – |
| B | no | 40.30 ± 0.46 | 65.06 | 28.66 | 25.03 | – |

Gain from the caption: A +0.16, B +0.23, **C −0.62** (Other −1.62).

## 5. What this shows

1. **When captions contain the answer, question-dependent captions do improve VQA, but only through C's pointer.** Trained and tested on captioner-seen data (E2), C gains **+2.6 points overall** (about 3× the seed std) and **+4.6 on the exact pairs the captioner was trained on**: number +7.1, other +5.1. On new questions about the same images, where the captions rarely contain the answer (6.4%), the gain disappears (−0.55). The gain follows caption answer coverage, not the image.
2. **The size of the gain matches the human-caption upper bound.** With COCO human captions on unseen images, C gained +2.6 (benchmark.md §8). Here C gains the same +2.6 from the leaked captions. In both cases the pointer does the work.
3. **A and B do not exploit the leaked answers.** A ignores the caption (shuffle −0.3, gain 0.0). B depends on the caption heavily (shuffle −8.0) but gains only +0.9, as in the unseen benchmark. An attention-pooled caption vector does not pass on a specific answer word, and a pointer does.
4. **A model must be trained with informative captions to use them (E1).** The 40k-trained models see captions with twice the answer coverage, and C still gains nothing (+0.27 overall, −0.12 on trained pairs). During training their captions were right only 18% of the time, so they learned not to trust them (C's pointer rate 0.13 vs 0.30 in E2). Better captions only at test time are not enough. The VQA model must learn on captions of the same quality.
5. **Leakage-trained models over-trust captions on honest data (E3).** C trained on seen data loses 0.6 points on unseen images (Other −1.6), and its shuffle drop is 6.1 points. It learned to copy from the caption, and on new images the caption is usually wrong. A similar train/test mismatch would make a published caption gain disappear on truly new images.
6. **This matters for reading the base papers.** A gain of +2.6 to +4.6 appears here only because the captioner had seen the VQA images and questions. On unseen images the same captioner and model give −0.65 (benchmark.md §3). Part of the +5 to +7 points reported by Sharma & Jalal and IC-VQA may come from this overlap (their captioners are trained on COCO images that overlap with VQA v2). benchmark.md §9.5 suggests reporting both conditions. This report gives that number for our pipeline: **about +3.3 points of C's caption gain (−0.65 → +2.6) comes from the captioner having seen the data.**

## 6. Caveats

- E2/E3 train on 11k questions (vs 40k), so the no-caption baseline is weaker. The caption gain may be smaller with more VQA data, as the baseline improves.
- The seen test has 3,107 questions (unseen: 10,038). The C gain on number questions (+7.1 on 329 qd_pair questions) has a large seed std (4.3).
- 61% of seen questions are exact captioner training pairs. In the base papers' setting (a captioner trained on generic COCO human captions) the leak is weaker: the captioner has seen the image, not the question.

## 7. Reproduce

```bash
cd CaptionVQA
python make_seen_split.py --config configs/seen.yaml                                 # split + region cache copy
python prepare_features.py --config configs/seen.yaml --splits train,dev,test        # 3-beam captions (~3 min)
python caption_stats.py --config configs/seen.yaml --split test                      # table in §2
bash scripts/run_seen.sh      # E2 (18 runs, ~35 min) + E1 + E3 evals
python eval_runs.py --runs-config configs/seen.yaml --data-config configs/seen.yaml --split test   # E2 with qd_pair breakdown
```

Outputs: `outputs/seen/runs/<model>[_nocap]_s<seed>/` (E2), `outputs/seen/evals/{small,seen}/*.json` (E1, E2 breakdown), `outputs/small/evals/seen/*.json` (E3).
