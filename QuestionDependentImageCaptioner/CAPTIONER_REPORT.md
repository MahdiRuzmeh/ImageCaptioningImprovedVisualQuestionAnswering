# QD captioner report: question vs image dependence (kaggle_mini)

| | |
|---|---|
| Checkpoint | `outputs/kaggle_mini/best.pt`: `MMTransformerCopyCaptioner`, best val CIDEr-D at epoch 17 |
| Splits | `configs/kaggle_mini.yaml`, image-disjoint rows of `vqa_v2_question_dependent_captions_train2014.json`: **train** 0–9500 (9,502 items / 2,895 images), **val** 9500–10000 (504 / 146), **test** 10000–10500 (494 / 165). Only train was used for training |
| Decoding | beam 3, length penalty 0.7, no-repeat 3-gram (same as `benchmark.py`) |
| Script | `scripts/captioner_probe.py` (§8). Every caption of every condition is saved in `outputs/probe/kaggle_mini_{split}.json` |

Contents: §1 metrics · §2 main results · §3 same image, different questions · §4 same question, different images · §5 question only · §6 image only · §7 conclusions · §8 reproduce.

## 1. Metrics

Question types (from the answer and the QD rule):

| Type | Definition | train / val / test |
|---|---|---|
| yes/no | answer is "yes" or "no" | 2,741 / 143 / 134 |
| number | numeric answer, or `how_many` rule | 1,747 / 99 / 92 |
| color | `what_color` rule or "what color…" | 1,564 / 80 / 73 |
| other | everything else (objects, places, sports, actions, …) | 3,450 / 182 / 195 |

| Metric | Measured on | Definition |
|---|---|---|
| BLEU-1 / BLEU-4, CIDEr-D, ROUGE-L | all | Against the single QD reference caption. These measure mostly the **sentence form** |
| **AnswerHit** | non-yes/no | Every answer word (numbers normalized: "two" = 2, "no/none" = 0) appears in the caption |
| **PolarityAcc** | yes/no | The caption is negative (no, not, n't, nothing, none, nobody, never, …) exactly when the answer is "no" |
| **CountAcc** | number | The first number in the caption equals the answer |
| **AnswerConsistency (AC)** | all | The caption agrees with the human answer: PolarityAcc for yes/no, CountAcc for number, AnswerHit otherwise. This is the one-number summary of "can the answer be read from the caption alone" |
| Question-word share | all | Share of caption words that also occur in the question |
| Fallback | all | Share of captions equal to the memorized caption "a tree is in front of the giraffes" |

Reference points: 47–49% of yes/no answers are "yes", so always answering "no" gives a PolarityAcc of 0.51 (train) / 0.53 (val) / 0.52 (test). The **prior** row in the tables is a model-free question-only baseline: the most frequent train answer for (question type, first two question words).

## 2. Main results: all conditions

Conditions (same checkpoint, same items):

| Condition | Image | Question | Section |
|---|---|---|---|
| **normal** | real | real | reference |
| **shuffle** | another image of the same split | real | §4 same question, different image |
| **mean** | split-average region features ("average image") | real | §5 question only |
| **mask** | none (all regions masked) | real | §5 question only |
| **noq** | real | empty (`[BOS, EOS]`) | §6 image only |
| **prior** | – | first 2 words + type | §5 baseline without a model |

### 2.1 Overall

| Split | Condition | CIDEr-D | BLEU-1 | BLEU-4 | ROUGE-L | AnswerHit | PolarityAcc | CountAcc | **AC** | Changed vs normal |
|---|---|---|---|---|---|---|---|---|---|---|
| train | **normal** | **5.45** | **0.763** | **0.572** | **0.749** | **0.357** | **0.598** | **0.295** | **0.427** | – |
| | shuffle | 4.62 | 0.723 | 0.507 | 0.705 | 0.159 | 0.574 | 0.241 | 0.279 | 54.6% |
| | mean | 4.65 | 0.717 | 0.502 | 0.704 | 0.178 | 0.583 | 0.249 | 0.295 | 50.3% |
| | mask | 4.17 | 0.634 | 0.401 | 0.682 | 0.190 | 0.557 | 0.219 | 0.296 | 66.5% |
| | noq | 0.30 | 0.242 | 0.037 | 0.211 | 0.050 | 0.488 | 0.005 | 0.175 | 97.5% |
| | prior | – | – | – | – | 0.178 | 0.550 | 0.249 | 0.285 | – |
| val | **normal** | **4.41** | **0.689** | **0.455** | **0.671** | **0.230** | **0.587** | 0.313 | **0.331** | – |
| | shuffle | 4.11 | 0.678 | 0.432 | 0.652 | 0.147 | 0.552 | 0.273 | 0.262 | 59.5% |
| | mean | 4.15 | 0.682 | 0.431 | 0.657 | 0.172 | 0.538 | 0.323 | 0.276 | 51.6% |
| | mask | 3.67 | 0.594 | 0.347 | 0.626 | 0.139 | 0.566 | 0.232 | 0.258 | 69.4% |
| | noq | 0.26 | 0.225 | 0.028 | 0.203 | 0.036 | 0.510 | 0.000 | 0.161 | 96.4% |
| | prior | – | – | – | – | 0.141 | 0.531 | **0.333** | 0.252 | – |
| test | **normal** | **4.90** | **0.739** | **0.516** | **0.718** | **0.281** | **0.582** | 0.293 | **0.362** | – |
| | shuffle | 4.26 | 0.696 | 0.459 | 0.672 | 0.125 | 0.507 | 0.250 | 0.229 | 55.5% |
| | mean | 4.44 | 0.703 | 0.472 | 0.684 | 0.158 | 0.507 | 0.272 | 0.253 | 52.4% |
| | mask | 3.85 | 0.594 | 0.349 | 0.646 | 0.139 | 0.545 | 0.228 | 0.249 | 67.8% |
| | noq | 0.34 | 0.248 | 0.042 | 0.219 | 0.058 | 0.470 | 0.000 | 0.170 | 97.6% |
| | prior | – | – | – | – | 0.169 | 0.507 | **0.304** | 0.261 | – |

### 2.2 AnswerConsistency per question type

| Split | Type (n) | normal | shuffle | mean | mask | noq | prior |
|---|---|---|---|---|---|---|---|
| train | yes/no (2,741) | **0.598** | 0.574 | 0.583 | 0.557 | 0.488 | 0.550 |
| | number (1,747) | **0.294** | 0.240 | 0.249 | 0.219 | 0.005 | 0.249 |
| | color (1,564) | **0.377** | 0.230 | 0.280 | 0.324 | 0.000 | 0.183 |
| | other (3,450) | **0.379** | 0.086 | 0.095 | 0.114 | 0.091 | 0.139 |
| val | yes/no (143) | **0.587** | 0.552 | 0.538 | 0.566 | 0.510 | 0.531 |
| | number (99) | 0.313 | 0.273 | 0.323 | 0.232 | 0.000 | **0.333** |
| | color (80) | 0.212 | 0.175 | **0.263** | 0.237 | 0.000 | 0.113 |
| | other (182) | **0.192** | 0.066 | 0.049 | 0.038 | 0.044 | 0.049 |
| test | yes/no (134) | **0.582** | 0.507 | 0.507 | 0.545 | 0.470 | 0.507 |
| | number (92) | 0.293 | 0.250 | 0.272 | 0.228 | 0.000 | **0.304** |
| | color (73) | 0.247 | 0.178 | **0.260** | **0.260** | 0.000 | 0.205 |
| | other (195) | **0.287** | 0.046 | 0.067 | 0.051 | 0.108 | 0.092 |

(For yes/no AC = PolarityAcc, for number AC = CountAcc, for color/other AC = AnswerHit.) CIDEr-D of the normal captions per type, test: yes/no 5.25, number 3.95, color 6.56, other 4.59.

## 3. Same image, different questions

All pairs of different questions asked about the same image (normal decoding). "Different type" means for example a color question vs a counting question on the same image.

| | train | val | test |
|---|---|---|---|
| Images with ≥ 2 questions | 2,344 | 119 | 131 |
| Question pairs (all / different type / same type) | 25,364 / 17,721 / 7,643 | 1,491 / 1,013 / 478 | 1,000 / 695 / 305 |
| **Caption differs**: different type | **99.8%** | **99.8%** | **99.7%** |
| Caption differs: same type | 98.7% | 99.2% | 97.7% |
| **Question alignment**: different type | **95.5%** | **95.7%** | **96.0%** |
| Question alignment: same type | 92.6% | 91.2% | 90.2% |
| Own answer in caption (different type, non-yes/no) | 33.4% | 19.2% | 24.8% |
| *Other* question's answer in caption (different type) | 2.7% | 2.0% | 1.4% |
| Own / other answer (same type) | 31.3% / 12.4% | 18.0% / 12.0% | 22.5% / 13.6% |
| Both captions of the pair consistent (AC) | 19.7% | 9.9% | 13.4% |

*Question alignment* = the caption for question *i* is closer (bag-of-words F1) to the reference of *i* than to the reference of the other question *j* on the same image (ties = ½). Random = 50%.

Examples (test, one block per image):

| Image | Type | Question | Answer | Caption |
|---|---|---|---|---|
| 23096 | number | How many cars are there? | 2 | there are two cars ✓ |
| | yes/no | Is the parking meter gray? | yes | the parking meter is gray ✓ |
| | yes/no | Is it raining? | no | it is not raining ✓ |
| | yes/no | Is there a red car in this picture? | no | there is no red car in the picture ✓ |
| | other | What is on the pole? | parking meter | a car is on the pole ✗ |
| 23098 | other | What type of sport are they playing? | tennis | they are playing tennis ✓ |
| | other | What is the woman doing? | playing tennis | the woman is playing tennis ✓ |
| | yes/no | IS this person serving the ball? | no | the person is not serving the ball ✓ |
| | other | What is the little girl wearing on her head? | hat | the little girl is wearing a ball ✗ |
| | number | How many hands is on the racket? | 2 | there is one person ✗ |
| 23047 | color | What color is the blender? | white | the blender is blue ✗ |
| | color | What color is the lit up light? | white | the lit up is red ✗ |
| | number | How many blue cakes are visible? | 0 | two blue cakes are visible ✗ |
| | yes/no | Is the child eating yogurt? | yes | the child is not eating eating ✗ |
| | other | Where is the watermelon? | bowl | a tree is in front of the giraffes ✗ (fallback) |
| 22954 | color | What color is the tile on the wall? | white | the tile on the wall is white ✓ |
| | number | How many pots are shown? | 3 | two pots are shown ✗ |
| | other | What is hanging on the back of the stove? | oven mitts | a <unk> is hanging on the back of the stove ✗ |
| | yes/no | Is there more than one metal pan? | yes | a tree is in front of the giraffes ✗ (fallback) |

**Result: the captioner is strongly question-dependent.** Different questions on the same image almost always give different captions (99.7–99.8% for different types). 96% of captions match their own question's reference better than the other question's. Answers almost never leak from one question to another (1.4–2.7% for different types). The form is always right for the question. Whether the answer slot is right is a separate issue (§4–§5).

## 4. Same question, different images

### 4.1 Image swap (shuffle): every item, question kept, image replaced

| | train | val | test |
|---|---|---|---|
| Caption changed | 54.6% | 59.5% | 55.5% |
| Extracted answer changed (polarity / count / caption) | 50.2% | 53.2% | 51.0% |
| AC normal → shuffle | 0.427 → 0.279 | 0.331 → 0.262 | 0.362 → 0.229 |
| AnswerHit normal → shuffle | 0.357 → 0.159 | 0.230 → 0.147 | 0.281 → 0.125 |
| PolarityAcc normal → shuffle | 0.598 → 0.574 | 0.587 → 0.552 | 0.582 → 0.507 |
| CountAcc normal → shuffle | 0.295 → 0.241 | 0.313 → 0.273 | 0.293 → 0.250 |
| CIDEr-D normal → shuffle | 5.45 → 4.62 | 4.41 → 4.11 | 4.90 → 4.26 |
| Question-word share (normal / shuffle / reference) | 0.72 / 0.71 / 0.46 | 0.70 / 0.70 / 0.44 | 0.71 / 0.70 / 0.46 |

Per type (§2.2): **other** collapses with another image (test 0.287 → 0.046), while **color** (0.247 → 0.178), **number** (0.293 → 0.250) and **yes/no** (0.582 → 0.507) change much less.

Examples (test; same question, real vs swapped image):

| Question | Answer | Real image | Swapped image |
|---|---|---|---|
| What room is this? | kitchen | this is the kitchen ✓ | this is a frisbee |
| What sport is being played? | tennis | tennis is being played ✓ | a fork is being played |
| What sport is the woman playing? | tennis | the woman is playing tennis ✓ | the woman is playing <unk> |
| What color are the dogs? | brown | the dogs are brown ✓ | the dogs are blue |
| What color is the tile on the wall? | white | the tile on the wall is white ✓ | the tile on the wall is brown |
| How many dogs are in the picture? | 2 | there are two dogs ✓ | there are two dogs (unchanged) |
| How many boats are in the picture? | 1 | there are two boats ✗ | there are two boats (unchanged) |
| Is it sunny? | no | it is not sunny ✓ | it is not sunny (unchanged) |

### 4.2 Natural pairs: the same question text asked about different images

| | train | val | test |
|---|---|---|---|
| Question groups (≥ 2 images) / items | 927 / 3,498 | 21 / 50 | 25 / 56 |
| Pairs with **different** human answers | 9,297 | 28 | 29 |
| → caption changed | 58.6% | 75.0% | 69.0% |
| → extracted answer changed | 58.7% | 67.9% | 69.0% |
| → both captions consistent | 21.0% | 14.3% | 31.0% |
| Pairs with the **same** human answer | 2,880 | 13 | 9 |
| → identical caption | 84.6% | 69.2% | 88.9% |
| → both captions consistent | 65.9% | 53.8% | 33.3% |

Val and test have only 26–29 different-answer pairs, so their rates are rough. Train has 9k pairs.

Examples (test; each cell is a different image):

| Question | Answers | Captions |
|---|---|---|
| What type of animal is this? | bird / zebra / cow | "this is a bird" / "this is a zebra" / "this is a cow" ✓✓✓ |
| What is the man holding? | frisbee / dog | "…holding a frisbee" / "…holding a dog" ✓✓ |
| What is the child holding? | tennis racket / umbrella | "…a tennis racket" / "…an umbrella" ✓✓ |
| What sport is being played? | tennis / baseball / frisbee | "tennis…" / "baseball…" / "tennis…" ✓✓✗ |
| How many people are present? | 2 / 15 | "two people are present" ×2 ✓✗ |
| What color is the sky? | gray ×3 / blue | "the sky is blue" ×4 ✗✗✗✓ |
| Is it raining? | yes / no | "it is not raining" ×2 ✗✓ |
| What color is the bus? | yellow and green / yellow | "the bus is red" ×2 ✗✗ |

**Result: the caption follows the image for about half of the items.** For objects, animals, sports and actions the answer word comes from the image. For counts, colors and yes/no it mostly does not.

## 5. Question only (image removed)

Two ways to remove the image: **mask** (no regions) and **mean** (the split's average region). **prior** is a question-only guess without the model.

| | train | val | test |
|---|---|---|---|
| AC normal / **mask** / mean / prior | 0.427 / **0.296** / 0.295 / 0.285 | 0.331 / **0.258** / 0.276 / 0.252 | 0.362 / **0.249** / 0.253 / 0.261 |
| CIDEr-D normal / mask / mean | 5.45 / 4.17 / 4.65 | 4.41 / 3.67 / 4.15 | 4.90 / 3.85 / 4.44 |
| BLEU-4 normal / mask / mean | 0.572 / 0.401 / 0.502 | 0.455 / 0.347 / 0.431 | 0.516 / 0.349 / 0.472 |
| Caption identical to normal (mask) | 33.5% | 30.6% | 32.2% |

Per type, test (normal → mask → prior): yes/no 0.582 → 0.545 → 0.507, number 0.293 → 0.228 → **0.304**, color 0.247 → **0.260** → 0.205, other 0.287 → 0.051 → 0.092.

Examples (test):

| Question | Answer | Normal | Question only (mask) |
|---|---|---|---|
| What color is the tile on the wall? | white | the tile on the wall is white | the tile on the wall is white |
| How many pots are shown? | 3 | two pots are shown | two pots are shown |
| Is anyone riding a bicycle? | yes | no one is riding a bicycle | no one is riding a bicycle |
| What type of sport are they playing? | tennis | they are playing tennis | they they are playing baseball |
| What room is this? | kitchen | this is the kitchen | this is the bathroom bathroom |
| What is this person riding on? | dirt bike | this person is riding a motorcycle | this person is riding a hat |
| What is the woman doing? | playing tennis | the woman is playing tennis | the woman is walking walking |

What this shows:

1. **Without an image the model still writes the right sentence for the question** (test CIDEr-D 3.85 vs 4.90; a third of the captions are identical). Fluency degrades ("they they are playing", "front front front"), because the decoder was never trained without regions.
2. **Counting and color are answered from the question.** On val/test, question-only accuracy for number and color is as good as with the image (number: prior 0.304 ≥ normal 0.293; color: mask 0.260 ≥ normal 0.247). Only on the *training* images does the image help for color (0.377 vs 0.324), which is memorization.
3. **Yes/no is close to the question prior.** PolarityAcc with image 0.58–0.60, without 0.55–0.57, always-"no" 0.51–0.53.
4. **"Other" questions need the image.** Without it AC falls from 0.287 to 0.051 (test), below the prior (0.092).

## 6. Image only (question removed)

"Caption only" is read here as the captioner's **image-only** output: the real image with an empty question (`[BOS, EOS]`). The model was never trained without a question. Whether the *answer* can be read from the caption alone is what AC measures in §2–§5.

| | train | val | test |
|---|---|---|---|
| **Fallback "a tree is in front of the giraffes"** | **59.4%** | **55.8%** | **66.4%** |
| Distinct captions / items | 1.7% | 7.5% | 8.5% |
| AC / AnswerHit / PolarityAcc / CountAcc | 0.175 / 0.050 / 0.488 / 0.005 | 0.161 / 0.036 / 0.510 / 0.000 | 0.170 / 0.058 / 0.470 / 0.000 |
| vs QD references: CIDEr-D / BLEU-4 | 0.30 / 0.037 | 0.26 / 0.028 | 0.34 / 0.042 |
| vs **COCO human captions** of the image: CIDEr-D, image-only / normal | 0.107 / 0.222 | 0.106 / 0.229 | 0.107 / 0.209 |
| vs COCO human captions: BLEU-1, image-only / normal | 0.377 / 0.261 | 0.333 / 0.247 | 0.381 / 0.262 |
| Non-fallback captions with a content word from the image's own COCO captions (vs another image's) | 63.7% (5.9%) | 59.6% (3.8%) | 59.3% (3.7%) |

Examples (test; image-only caption vs the first COCO human caption):

| Image | Image only | COCO human caption |
|---|---|---|
| 22928 | a dog is shown | two tan colored dogs sitting face to face |
| 22974 | a sandwich is present | … a man about to bite into a sandwich |
| 23047 | a kitchen is present | a young boy eating at a counter with a blender … |
| 23050 | the man is skateboarding | a man on a skateboard at the top of a ramp |
| 22964 | a tennis tennis is shown | a woman standing on a tennis court holding a racquet |
| 22953 | no one is present | a kitchen with people eating food … |
| 22924 / 22926 / 22940 / … | a tree is in front of the giraffes | (not giraffe images) |

What this shows:

1. **The captioner cannot write a caption without a question.** For 56–66% of images it outputs one memorized training caption. It is a question-conditioned model, not a general captioner.
2. **When it does write something else, it describes the real image.** 59–64% of those captions name an object from the image's COCO captions, against 4–6% for a random image ("a dog is shown", "the man is skateboarding"). The visual features carry the main object, and the question is what unlocks a sentence about it.
3. Against COCO human captions, CIDEr-D is low in both conditions (≈ 0.1–0.2). QD captions are short, question-shaped sentences, not scene descriptions, so this comparison only confirms that image-only output is not a usable generic caption.

## 7. Conclusions

| Dependence | Verdict | Evidence |
|---|---|---|
| On the question | **Strong** | 99.8% different captions for different-type questions on one image; 96% question alignment; 1.4–2.7% answer leakage between questions; without a question 56–66% fallback |
| On the image: objects, places, sports, animals, actions ("other") | **Grounded** | Test AC 0.287 → 0.046 with another image and 0.051 without one; same-question pairs follow the image (bird / zebra / cow) |
| On the image: color | **Weak / prior** | Unseen images: question-only (0.260) ≥ with image (0.247); "the sky is blue" for gray skies. Helps only on training images (memorized) |
| On the image: counting | **Blind** | CountAcc ≈ prior (test 0.293 vs 0.304); "two" written regardless of the image |
| On the image: yes/no | **Mostly prior** | PolarityAcc 0.58–0.60 vs 0.51–0.55 without the image |
| Generalization | Moderate gap | AC train 0.427 → test 0.362; AnswerHit 0.357 → 0.281 |

1. **The captioner is question-driven, partly grounded and not visually blind.** The question fixes the sentence (71% of caption words come from the question; references 46%). The image fills the answer slot only for object/scene questions.
2. **CIDEr and BLEU overstate grounding.** They stay high without the image (test CIDEr-D 3.85 vs 4.90) because the sentence form comes from the question. AnswerHit, PolarityAcc, CountAcc and AC are the metrics to report for this model.
3. **Weakest parts, in order:** counting (blind), yes/no (near prior), color (prior on unseen images), and fallback to a memorized caption when the model is unsure (2.6–3.2% of normal captions, 56–66% without a question).
4. **Next steps** (consistent with `EVAL_V1_VS_V2.md` §5): more training data first. Then better visual features for counting and color (box geometry, bottom-up Visual Genome features or a global CLIP feature), and a polarity/count auxiliary loss on the encoder. Use the per-type AC table (§2.2) and the question-only gap (§5) to measure progress.

## 8. Reproduce

```bash
cd QuestionDependentImageCaptioner
python scripts/captioner_probe.py --config configs/kaggle_mini_local.yaml --ckpt outputs/kaggle_mini/best.pt \
  --split test --out outputs/probe/kaggle_mini_test.json        # also --split val / --split train (~15 min)
```

`configs/kaggle_mini_local.yaml` = the split ranges of `configs/kaggle_mini.yaml` with local paths and the local region cache. The output JSON holds all metrics (`results.<condition>.<type>`), the pair analyses (`results.same_image`, `results.same_question`), the COCO comparison (`results.coco_reference`) and every caption of every condition (`items`). The shuffle/mean/mask numbers match `outputs/ablation/kaggle_mini_ep17_*.json` (`scripts/image_ablation.py`).
