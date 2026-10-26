# v1 vs v2 — evaluation on train / val / test

Comparison of the two Stage-1 captioners on all three splits of the benchmark protocol, to decide the next step.

| | |
|---|---|
| **v1** | `QuestionDependentImageCaptioner` (GRU question + one attention + LSTM decoder) |
| **v2** | `MMTransformerCopyCaptioner` (joint region+question Transformer + copy from question) |
| Checkpoints | `outputs/bench/{v1,transformer_copy}/seed{42,43,44}/best.pt` (best val CIDEr-D) |
| Data | rows of `vqa_v2_question_dependent_captions_train2014.json`, image-disjoint: train 0–3000 (3000 rows / 905 images), val 3000–3500 (503 / 153), test 3500–4000 (499 / 145) |
| Decoding | beam 3, the same scoring code as `benchmark.py` |
| Numbers | mean ± std over 3 seeds |

AnswerHit = the answer word appears in the predicted caption. It is computed on non-yes/no questions only (train 2157, val 365, test 341 rows), because yes/no answers rarely appear as words in the caption.

## 1. Overall results

| Split | Model | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | AnswerHit |
|---|---|---|---|---|---|---|
| train | v1 | 0.719 ± 0.038 | 0.484 ± 0.055 | 4.429 ± 0.568 | 0.732 ± 0.038 | 0.514 ± 0.052 |
| train | **v2** | **0.902 ± 0.036** | **0.835 ± 0.063** | **8.096 ± 0.616** | **0.895 ± 0.036** | **0.699 ± 0.093** |
| val | v1 | 0.481 ± 0.008 | 0.188 ± 0.008 | 1.585 ± 0.044 | 0.476 ± 0.005 | 0.161 ± 0.016 |
| val | **v2** | **0.672 ± 0.006** | **0.438 ± 0.010** | **4.202 ± 0.049** | **0.656 ± 0.007** | **0.220 ± 0.003** |
| test | v1 | 0.482 ± 0.008 | 0.180 ± 0.008 | 1.528 ± 0.062 | 0.486 ± 0.005 | 0.171 ± 0.015 |
| test | **v2** | **0.666 ± 0.007** | **0.418 ± 0.008** | **4.073 ± 0.082** | **0.656 ± 0.010** | **0.179 ± 0.008** |

| | v1 | v2 |
|---|---|---|
| Trainable params | 16.2M | 21.1M |
| Best epoch (seeds 42/43/44) | 18 / 30 / 20 | 14 / 18 / 12 |
| Decode speed, test (items/s, beam 3) | ~137 | ~209 |

### Generalization gap (train → test)

| Model | CIDEr-D train → test | AnswerHit train → test |
|---|---|---|
| v1 | 4.43 → 1.53 (−65%) | 0.51 → 0.17 |
| v2 | 8.10 → 4.07 (−50%) | 0.70 → 0.18 |

Val and test scores are close for both models, so selecting `best.pt` on val doesn't bias the test numbers.

## 2. Per question type (CIDEr-D / AnswerHit, mean over 3 seeds)

| Type | n (train/val/test) | Split | v1 CIDEr | v2 CIDEr | v1 AnswerHit | v2 AnswerHit |
|---|---|---|---|---|---|---|
| how_many | 279 / 46 / 49 | train | 4.22 | 6.17 | 0.60 | 0.55 |
| | | val | 0.82 | 2.93 | 0.24 | 0.20 |
| | | test | 0.42 | 2.44 | 0.14 | 0.16 |
| what_color | 464 / 93 / 64 | train | 4.26 | 8.82 | 0.59 | 0.70 |
| | | val | 2.04 | 6.10 | 0.16 | 0.20 |
| | | test | 1.73 | 6.21 | 0.19 | 0.18 |
| what_is_doing | 69 / 7 / 9 | train | 6.11 | 9.24 | 0.62 | 0.90 |
| | | val | 3.82 | 6.53 | 0.24 | 0.52 |
| | | test | 3.20 | 4.84 | 0.37 | 0.26 |
| llm_fallback (all other questions) | 2184 / 357 / 377 | train | 4.38 | 8.17 | 0.46 | 0.72 |
| | | val | 1.48 | 3.87 | 0.14 | 0.22 |
| | | test | 1.55 | 3.93 | 0.16 | 0.18 |

`what_is_doing` has only 7–9 val/test items, so its numbers are noise. `who` has 4 rows, all in train.

## 3. What the models actually generate (test, seed 42)

| Question | Answer | Reference | v1 | v2 |
|---|---|---|---|---|
| What color is the computer keyboard? | white | The computer keyboard is white. | the flower is white | the computer keyboard is black |
| Is that a big scissor? | yes | A big scissor is present. | a tree is in front of the giraffes | a big scissor is present |
| What is the guy riding? | horse | The guy is riding a horse. | the man is riding a truck | the guy is riding a bike |
| What colors are the bike? | red and white | The bike are red and white. | the boys are black | the bike are red |
| Are there any people in the stands? | no | There are no people in the stands. | there are no people | no people are in the stands |
| Is the sky clear? | no | The sky is not clear. | the bench is not | the sky is clear |
| What is the weather like? | sunny | The weather is sunny. | the person is moving | the weather is not sunny |
| What is the kid doing on the bench? | climbing | The kid is climbing on the bench. | a tree is on the ground | the kid is cooking on the bench |
| How many screws are on the stop sign? | 4 | There are four screws. | there is one clock | there are two on the stop sign |
| Is there more than one car visible? | no | Not both cars are visible. | two people are pictured | a tree is in front of the giraffes |

Statistics on the test predictions (seed 42):

| | v1 | v2 |
|---|---|---|
| Share of predicted words that also appear in the question | 0.50 | 0.73 |
| Distinct captions / items | 0.83 | 0.93 |
| Most frequent caption | "a tree is in front of the giraffes" (16×) | same (13×) |
| AnswerHit when the answer word isn't in the question (n = 329) | 0.109 | 0.137 |

## 4. Findings

1. **Overall, v2 is better than v1 on every split and metric, except AnswerHit on test, where the two are tied (0.18 vs 0.17).** By question type, v1 still has a slightly higher AnswerHit on a few small groups (e.g. test `what_color`, `what_is_doing`). On unseen images it scores 2.7× v1 on CIDEr-D and 2.3× on BLEU-4. It also trains to its best epoch faster and decodes faster.
2. **The gain comes from the caption's form, not its content.** v2 copies the question's subject and structure ("the computer keyboard is …", "the guy is riding …"), and 73% of its words come from the question. v1 often loses the subject entirely ("the flower is white" for a keyboard question).
3. **Neither model gets the answer from the image.** The answer word (color, count, action, object) is right on unseen images only ~14–18% of the time. v2 has the right sentence with the wrong slot ("keyboard is black", "riding a bike", "two" for 4). It also flips yes/no answers ("the sky is clear" when the answer is no).
4. **Both models overfit strongly.** v2 reaches AnswerHit 0.70 on train but 0.18 on test, so it memorizes answers for training images rather than learning to read them from regions. The training set is small: 3000 rows from 905 images.
5. **Both models fall back to a memorized training caption** when they're unsure ("a tree is in front of the giraffes" for scissors, cars, …).
6. **Counting is the weakest type** (test CIDEr 2.44 for v2, 0.42 for v1). The frozen detector's 32 region features don't carry a usable count.

## 5. Recommendation for the next step

**Keep v2 as the Stage-1 captioner and stop changing the architecture for now.** The v1 → v2 change already fixed the caption form. What's left is grounding the answer word in the image, and the train/test gap points to data size and visual features rather than the decoder.

Suggested order:

1. **Train v2 on much more data (most important, cheapest).** Use 30k–100k rows, or the full train2014 JSON with the val2014 JSON for validation (`configs/default.yaml`). Re-check the test AnswerHit and the train/test gap. If AnswerHit rises clearly, the bottleneck was data. If it stays near 0.2, the bottleneck is the visual features.
2. **If AnswerHit stays low, improve the visual features.** Options: bottom-up-attention (Visual Genome) region features instead of torchvision's COCO Faster R-CNN, or adding a global CLIP/ViT image feature. Count and color are the types to watch.
3. **Measure the effect on Stage 2 (SimpleVQA).** Caption metrics are only a proxy. Train VQA with the v2 checkpoint (and v1 for reference) and compare VQA accuracy. That accuracy is the thesis result.
4. **Lower priority:** add an answer-word auxiliary loss (predict the answer from the encoder), or penalize repeated fallback captions. Try these only once steps 1–3 show where the gap is.

A 1000-item run is fine as a quick sanity check of the pipeline. It has fewer rows than this benchmark, though, so expect lower scores and a larger train/test gap, not a better model.

## Reproduce

`benchmark.py` accepts only `--split val|test`. The train-split numbers were produced by the same `benchmark.main()` with `train` added to the allowed splits. For val/test:

```bash
python benchmark.py --config configs/bench_transformer_copy.yaml --ckpt outputs/bench/transformer_copy/seed42/best.pt --name transformer_copy --split val --out outputs/v2_val.json
python benchmark.py --config configs/bench_v1.yaml --ckpt outputs/bench/v1/seed42/best.pt --name v1 --split val --out outputs/v1_val.json
```
