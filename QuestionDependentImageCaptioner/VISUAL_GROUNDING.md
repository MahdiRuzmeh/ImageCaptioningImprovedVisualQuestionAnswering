# Is the QD captioner visually grounded? (same question, different image)

Checkpoint: `outputs/kaggle_mini/best.pt` (`MMTransformerCopyCaptioner`, best epoch 17).
Splits: `configs/kaggle_mini.yaml`, image-disjoint rows of the QD caption JSON: train 0–9500 (9,502 rows), val 9500–10000 (504), test 10000–10500 (494). Decoding: beam 3.

An earlier check showed that captions change when the **question** changes and the image stays the same. This report covers the other direction: keep the **question** fixed and change the **image**. If the captions stay the same, the model is visually blind. If they follow the image, it is grounded.

## 1. Tests

| Test | What changes | What a grounded model does |
|---|---|---|
| **normal** | nothing (real image, real question) | reference |
| **shuffle** | the image is replaced by another image from the same split. The question stays the same | caption changes, answer word mostly lost |
| **mean** | all 32 regions = the split's mean region vector (an "average image") | answer word mostly lost |
| **mask** | regions masked out (no image at all) | answer word mostly lost |
| **prior** | no model: the most frequent train answer for (question type, first 2 question words) | what a question-only guess gets |
| **same question** | natural pairs: the *same question text* asked on two different images in the split | different answers → different captions, each with its own answer |

**AnswerHit** = the answer word appears in the caption (non-yes/no questions only: train 6,761, val 361, test 360). It is the measure of grounding here: BLEU/CIDEr mostly reward the sentence form, and that form comes from the question.

## 2. Image swap — results

| Split | Mode | AnswerHit | CIDEr-D | BLEU-4 | Captions changed vs normal |
|---|---|---|---|---|---|
| train | normal | **0.357** | 5.45 | 0.572 | – |
| | shuffle | 0.159 | 4.62 | 0.507 | 54.6% |
| | mean | 0.178 | 4.65 | 0.502 | 50.3% |
| | mask | 0.190 | 4.17 | 0.401 | 66.5% |
| | prior (question only) | 0.180 | – | – | – |
| val | normal | **0.230** | 4.41 | 0.455 | – |
| | shuffle | 0.147 | 4.11 | 0.432 | 59.5% |
| | mean | 0.172 | 4.15 | 0.431 | 51.6% |
| | mask | 0.139 | 3.67 | 0.347 | 69.4% |
| | prior | 0.141 | – | – | – |
| test | normal | **0.281** | 4.90 | 0.516 | – |
| | shuffle | 0.125 | 4.26 | 0.459 | 55.5% |
| | mean | 0.158 | 4.44 | 0.472 | 52.4% |
| | mask | 0.139 | 3.85 | 0.349 | 67.8% |
| | prior | 0.169 | – | – | – |

Summary:

| | train | val | test |
|---|---|---|---|
| ImageDep = AnswerHit(normal) − AnswerHit(shuffle) | **+0.198** | **+0.083** | **+0.156** |
| AnswerHit(normal) − question-only prior | +0.177 | +0.089 | +0.112 |
| Correct answers that survive the image swap | 35.2% | 41.0% | 31.7% |
| Share of caption words copied from the question (normal / shuffle / reference) | 0.72 / 0.71 / 0.46 | 0.70 / 0.70 / 0.44 | 0.71 / 0.70 / 0.46 |

## 3. Per question type (AnswerHit, normal → shuffle)

| Type | train | val | test |
|---|---|---|---|
| how_many (n = 875 / 44 / 48) | 0.314 → 0.225 | 0.295 → 0.250 | 0.333 → 0.292 |
| what_color (1473 / 74 / 66) | 0.377 → 0.229 | 0.230 → 0.189 | 0.273 → 0.197 |
| llm_fallback, objects/places/sports… (4202 / 232 / 236) | 0.348 → 0.124 | 0.211 → 0.116 | 0.275 → 0.076 |
| what_is_doing (195 / 10 / 9) | 0.569 → 0.087 | 0.300 → 0.000 | 0.222 → 0.000 |

## 4. Same question, different image (natural pairs)

All pairs of items in a split with the identical question text and different images.

| | train | val | test |
|---|---|---|---|
| Question groups (≥ 2 images) / items | 927 / 3,498 | 21 / 50 | 25 / 56 |
| Pairs with **different** human answers | 9,297 | 28 | 29 |
| → caption differs between the two images | **58.6%** | 75.0% | 69.0% |
| → both captions contain their own answer (non-yes/no pairs) | 22.7% | 15.4% | 34.6% |
| → exactly one does | 48.9% | 42.3% | 50.0% |
| Pairs with the **same** human answer | 2,880 | 13 | 9 |
| → identical caption | 84.6% | 69.2% | 88.9% |

Val/test have few pairs (26–29), so treat those rates as rough. Train has 9k pairs.

Test examples (same question, each row a different image):

| Question | Answers (image 1 / 2 / 3) | Captions |
|---|---|---|
| What sport is being played? | tennis / baseball / frisbee | "tennis is being played" / "baseball is being played" / "tennis is being played" |
| What type of animal is this? | bird / zebra / cow | "this is a bird" / "this is a zebra" / "this is a cow" |
| What is the man holding? | frisbee / dog | "the man is holding a frisbee" / "the man is holding a dog" |
| What is the child holding? | tennis racket / umbrella | "…a tennis racket" / "…an umbrella" |
| How many people are present? | 2 / 15 | "two people are present" / "two people are present" |
| What color is the sky? | gray ×3 / blue | "the sky is blue" ×4 |
| Is it raining? | yes / no | "it is not raining" / "it is not raining" |

Image-swap examples (test; S = caption with another image, same question):

| Question | Answer | Real image | Swapped image |
|---|---|---|---|
| What room is this? | kitchen | this is the kitchen | this is a frisbee |
| What sport is being played? | tennis | tennis is being played | a fork is being played |
| What color are the dogs? | brown | the dogs are brown | the dogs are blue |
| What color is the tile on the wall? | white | the tile on the wall is white | the tile on the wall is brown |
| How many dogs are in the picture? | 2 | there are two dogs | there are two dogs (unchanged) |

## 5. Findings

1. **The captioner is not visually blind. It is partly grounded.** On unseen test images, swapping the image drops AnswerHit from 0.281 to 0.125 (−0.156), and about 55% of captions change. The real image puts the model 11 points above a question-only guess (0.281 vs 0.169). The same pattern holds on train and val.
2. **A wrong image pulls the answer below the question-only prior** (test 0.125 shuffle vs 0.169 prior). The model really reads the regions: a different image gives a different, wrong answer word ("a fork is being played"), not a generic guess.
3. **The sentence form comes from the question, and the answer slot comes from the image.** About 71% of caption words are copied from the question, both with the real and with a swapped image (references: 46%). CIDEr only drops 4.90 → 4.26 under shuffle, so CIDEr is a poor measure of grounding. AnswerHit is the one to watch.
4. **Grounding depends strongly on the question type.**
   - **Objects, places, sports, animals, actions (llm_fallback, what_is_doing): well grounded.** Test 0.275 → 0.076 and 0.222 → 0.000 when the image is swapped. Same-question pairs ("what type of animal is this" → bird/zebra/cow) follow the image.
   - **Colors: weakly grounded** (test 0.273 → 0.197). The model falls back to frequent colors ("the sky is blue" for gray skies).
   - **Counting: almost blind** (test 0.333 → 0.292, captions often identical with another image). "two" is written regardless of the image. The 32 Faster R-CNN region features do not carry a usable count.
   - **Yes/no: blind.** "it is not raining" for both a rainy and a dry image.
5. **About a third of correct answers survive an image swap** (test 31.7%). These are answers the model gets from the question (frequent answers like "two", "white", "blue"), not from the image.
6. **Train vs test:** ImageDep is larger on train (+0.198) than on test (+0.156), and AnswerHit is 0.357 vs 0.281. The model relies on the image somewhat more on images it was trained on, but the gap is moderate. It does not simply memorize (question → answer): on train, a swapped image also drops AnswerHit to the prior level (0.159 vs 0.180).

**Answer to the question:** the captioner *is* visually grounded for "what object / what sport / what animal / what is X doing" questions, weakly grounded for colors, and visually blind for counting and yes/no. Overall, about half of its correct answer words on unseen images come from the image, the rest from question priors.

Implications for next steps: the counting and color weakness points to the visual features (frozen torchvision Faster R-CNN, 32 RPN proposals, no box coordinates), as already suspected in `EVAL_V1_VS_V2.md` §5. Box geometry or bottom-up (Visual Genome) features would be the first thing to try for counting.

## 6. Reproduce

```bash
cd QuestionDependentImageCaptioner
python scripts/image_ablation.py --config configs/kaggle_mini_local.yaml --ckpt outputs/kaggle_mini/best.pt \
  --split test --examples -1 --out outputs/ablation/kaggle_mini_ep17_test.json      # also --split val / train
```

`configs/kaggle_mini_local.yaml` holds the `kaggle_mini.yaml` split ranges with local paths. The val/test numbers reproduce the earlier `kaggle_mini_ep17_{val,test}.json` runs exactly. `--examples -1` stores every item (all modes + donor image id). The same-question block is the `same_question` entry of each output JSON.
