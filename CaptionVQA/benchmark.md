# Caption-aware VQA — benchmark of architectures A / B / C

Stage 2 of the thesis pipeline. We test whether the question-dependent captioner from Stage 1 (`MMTransformerCopyCaptioner`, checkpoint `QuestionDependentImageCaptioner/outputs/kaggle_mini/best.pt`) improves VQA accuracy. We compare three VQA architectures that read its captions, the same architectures without captions, and the existing `SimpleVQA` model.

Date: 2026-10-27. Hardware: one RTX 3070 Laptop GPU (8 GB).

## 1. Setup

| | |
|---|---|
| Data | **VQA v2 only** (questions, annotations, 10 human answers). The question-dependent caption dataset is **not** used in this stage |
| Train | 40,011 questions / 7,577 images from `train2014` (9% of the VQA v2 train split). All 3,521 images of the QD caption dataset are excluded, because the captioner was trained on them (see §5) |
| Dev | 5,006 questions / 984 images from `val2014`, used for model selection (best epoch) |
| Test | 10,038 questions / 1,893 images from `val2014`, image-disjoint from dev. **All reported numbers are on this split** |
| Image features | Faster R-CNN (torchvision, COCO) top-32 proposals, 1024-d ROI features. These are the same features the captioner was trained on |
| Captioner (frozen) | For every (image, question) pair: 3 beam-search captions (beam 3, max 20 tokens), their length-normalized log-probabilities, and the decoder hidden states of beam 1 |
| Answer head | Classifier over the 1,144 answers that are the majority answer ≥ 3 times in train. Loss = soft-score BCE (Teney et al. 2018) |
| Metric | Official VQA v2 accuracy (answer normalization + `min(1, #humans/3)` averaged over the 10 leave-one-out subsets), per answer type |
| Training | AdamW, lr 1e-3 (B: 1e-4 for layers copied from the captioner), 1 warm-up epoch + cosine decay, 15 epochs, batch 128, dropout 0.2, AMP. Best dev epoch → test |
| Seeds | 3 (0, 1, 2); values are mean ± std |

Split files: `outputs/small/split/{train,dev,test}.json`. Reproduce: see §8.

## 2. Architectures

All three receive the same inputs (question, 32 region features, captioner outputs). Each one is also trained **without the caption** (`--no-caption`) to measure what the caption adds.

| | Idea | Caption input | Based on |
|---|---|---|---|
| **A** — gated top-down attention | Question-guided attention over regions (`v̂`) and over caption words (`ĉ`). A learned gate `g = σ(W[q; v̂; ĉ])` mixes them: `h = q ⊙ (g·v̂ + (1−g)·ĉ)` | Beam-1 caption: frozen captioner word embeddings + decoder states → BiGRU | Bottom-Up/Top-Down (Anderson 2018), caption attention of Wu & Mooney 2019 |
| **B** — single-stream Transformer, initialised from the captioner | Sequence `[CLS] + regions + question + caption`. The 3 encoder layers, region projection, position/type embeddings and caption word embeddings are **copied from the captioner** (fine-tuned at 0.1× lr). `CLS` → classifier | Beam-1 caption tokens + decoder states | VisualBERT / ViLT style; warm start from Stage 1 |
| **C** — classifier + pointer into the caption | A's fusion, plus a pointer over every 1–3-word n-gram of the 3 beam captions, scored jointly with the classifier (one BCE over `answers ∪ caption n-grams`). A confidence gate (beam scores, negation flags) shifts all pointer scores; negation flags also feed the classifier | All 3 beams | Pointer-generator, M4C (TextVQA) |

Without a caption, A and C are the **same network** (C's pointer and gate need the caption), so they share one "no caption" row below.

| | Trainable params | Train time / epoch |
|---|---|---|
| A (caption / no caption) | 9.8M / 6.8M | 10 s / 5 s |
| B (caption / no caption) | 14.7M (10.8M copied from captioner) | 16 s / 12 s |
| C (caption) | 11.9M | 22 s |

## 3. Main results (test, 10,038 questions)

| Model | Caption | Overall | Yes/No | Number | Other | Dev (selection) |
|---|---|---|---|---|---|---|
| A | yes | 45.89 ± 0.58 | 67.02 ± 0.47 | **32.18 ± 0.28** | 33.84 ± 0.97 | 45.30 ± 0.20 |
| B | yes | 45.61 ± 0.21 | **67.19 ± 0.11** | 30.26 ± 0.72 | 33.64 ± 0.33 | 44.50 ± 0.45 |
| C | yes | 45.64 ± 0.49 | 66.76 ± 0.55 | 31.15 ± 1.25 | 33.80 ± 0.89 | 45.51 ± 0.56 |
| A = C | no | **46.29 ± 0.16** | 67.18 ± 0.32 | 31.91 ± 0.72 | **34.59 ± 0.14** | 45.56 ± 0.13 |
| B | no | 45.36 ± 0.21 | 66.66 ± 0.59 | 31.19 ± 0.15 | 33.29 ± 0.02 | 44.64 ± 0.46 |
| SimpleVQA (current Stage-2 model, generative LSTM decoder), 1 seed | yes | 41.79 | 65.33 | 29.80 | 27.50 | 41.18 |

**Takeaways**

1. **No architecture gains accuracy from the caption on this benchmark.** Each caption model is within about one standard deviation of its no-caption version: A −0.40, B +0.25, C −0.65 points overall. The best single configuration is the no-caption A/C network (46.29).
2. The three architectures are close to each other (45.6–46.3). At this data size the fusion design matters less than what the caption contains (§4, §5).
3. **All three new architectures beat the current SimpleVQA by 3.6–4.5 points** (41.79 → 45.4–46.3 across all A/B/C variants), mostly on Other (+5.8 to +7.1) and Yes/No (+1.3 to +1.9). SimpleVQA was trained on the same questions with its `kaggle_mini` hyper-parameters (batch 32, lr 1e-3 step decay, 15 epochs; best epoch = 15, still improving slowly; ~5.5 min/epoch vs ≤ 22 s for A/B/C). The gain comes from the classifier head with soft-score BCE and better fusion, not from the caption (point 1).
4. Per answer type, the caption changes Number accuracy by +0.27 (A), −0.93 (B) and −0.76 (C), and Other by −0.75 (A), +0.35 (B) and −0.79 (C). None of these is clearly larger than the seed noise.

## 4. Do the models actually read the caption?

At test time, each question's caption is replaced by one generated for a **different image** (same model weights). If accuracy drops, the model relies on the caption.

| Model | Real caption | Shuffled caption | Drop | Diagnostic |
|---|---|---|---|---|
| A | 45.89 | 45.78 ± 0.49 | **0.12** | Gate share for the image branch = 0.52–0.59 |
| B | 45.61 | 34.48 ± 0.89 | **11.12** | Yes/No drops 67.2 → 54.1, Other 33.6 → 22.3 |
| C | 45.64 | 43.66 ± 0.40 | **1.98** | Pointer is used for 9–14% of answers (6–7% with shuffled captions) |

- **A learns to ignore the caption.** The gate still gives it weight, but the caption branch carries no information the classifier uses. Shuffling changes almost nothing.
- **B depends on the caption heavily** (−11 points when shuffled) but **does not gain from it** compared with B without a caption. The caption is computed by the captioner from the same 32 regions and the same question, so B uses it as a shortcut to information it can otherwise get from the regions directly. It adds no new information.
- **C uses the pointer moderately.** It copies the answer from a caption for about 1 in 8 questions, but the copied answers are wrong about as often as the classifier's answers would have been.

## 5. Why: caption quality on images the captioner never saw

How often the frozen captioner's output contains a correct answer, scored like VQA accuracy (best n-gram, n ≤ 3), on the test split. Produced by `caption_stats.py`:

| Answer type | n | Correct answer in beam-1 caption | Correct answer in any of the 3 beams |
|---|---|---|---|
| all | 10,038 | 17.8 | 31.8 |
| yes/no | 3,712 | 10.1 | 22.9 |
| number | 1,310 | 27.2 | 51.6 |
| other | 5,016 | 21.0 | 33.2 |

(Yes/no is shown only for completeness: those answers are rarely written as "yes"/"no" in a sentence.)

The VQA models already reach 32–35% on Number/Other **without** captions. A caption that contains the right answer only 21–27% of the time (beam 1) cannot add much on top. The captioner was trained on only ~9.5k caption rows from ~2.9k images. The same architecture is about twice as good on its own training images as on unseen ones (Stage-1 benchmark: CIDEr-D 8.1 on train vs 4.1 on unseen test images, `QuestionDependentImageCaptioner/EVAL_V1_VS_V2.md`), which is why those images were excluded from VQA training here. Training VQA on them would teach the model to trust captions far more than it should on new images.

Typical test captions (`outputs/small/features/captions_test_sample.json`):

| Question | Human answer | Beam 1 / 2 / 3 |
|---|---|---|
| How many sinks are there? | 1 | there are two sinks / there is one sink / there are three sinks |
| What sport are they playing? | tennis | they are playing baseball / they are playing tennis / they are playing soccer |
| What color is the umbrella? | red | the umbrella is blue / the umbrella is red / the umbrella is yellow |
| What is on the ground? | luggage, suitcases | a suitcase is on the ground / … / snow is on the ground |
| What's hanging from the ceiling? | lamp | the ceiling is hanging from the ceiling / … |

The captioner produces well-formed, question-shaped sentences, and the correct answer is often among the 3 beams. But beam 1 is frequently a plausible guess rather than grounded in the image. This is why C's multi-beam pointer was the most promising of the three designs, yet it still does not outperform the classifier.

## 6. Comparison with the paper (Sharma & Jalal 2021, *Image captioning improved visual question answering*)

The paper was read only after A/B/C were designed, implemented and run.

**What the paper reports on VQA v2 (Table 6, test-standard):**

| Paper model | Number | Yes/No | Other | Overall |
|---|---|---|---|---|
| Baseline | 44.21 | 78.30 | 53.19 | 61.21 |
| Baseline + Att | 44.06 | 79.21 | 56.29 | 63.32 |
| Baseline + Cap | 46.98 | 85.31 | 59.18 | 69.91 |
| **Baseline + Att + Cap (concat)** — proposed model | **48.23** | **86.63** | **59.98** | **70.67** |
| Anderson et al. (Bottom-Up/Top-Down), as listed in the paper | 43.90 | 82.20 | 56.26 | 65.32 |

Validation ablation (paper Table 3): Baseline 57.12 → +Att 62.91 → +Cap 63.63 → full model 66.82 (concat) / 66.36 (sum).

**Side by side with this benchmark:**

| | Paper (full model) | This benchmark (best caption model, A) | This benchmark (no caption, A=C) |
|---|---|---|---|
| Overall | 70.67 | 45.89 | 46.29 |
| Yes/No | 86.63 | 67.02 | 67.18 |
| Number | 48.23 | 32.18 | 31.91 |
| Other | 59.98 | 33.84 | 34.59 |
| Gain from adding captions | **+7.35** (63.32 → 70.67, Table 6); +3.91 on val (Table 3) | **−0.40** | — |

**The absolute numbers are not comparable**, and the gap of about 25 points comes mainly from the setup, not the architecture:

| Factor | Paper | This benchmark |
|---|---|---|
| VQA training data | Full VQA v2 train (~443k questions; 248k for VQA 1.0) | 40k questions (9%) |
| Evaluation | VQA v2 test-standard (official server) | 10k-question subset of val2014 |
| Captioner training | Starts from MS-COCO human captions (5 per image), then question-based captions generated for every image–question pair of train/val/test (paper §5) | 9.5k LLM-generated question-dependent captions from 2.9k images |
| Image features | ResNet-101 global feature + 32 Faster R-CNN region features, bottom-up/top-down style (paper §3; detector pre-training data not stated) | torchvision COCO Faster R-CNN, top-32 RPN proposals, no global feature, no box coordinates |
| Answer output | Generative LSTM answer decoder | Classifier over 1,144 answers |

What **is** comparable is the *relative* effect of the caption. The paper reports +7.35 points (test-std) from adding captions to the attention model. Here the caption adds nothing measurable, across three different fusion designs and 3 seeds. Points to keep in mind when citing the paper:

- In the paper's Table 6, the caption branch alone (Baseline + Cap, 69.91) beats Bottom-Up/Top-Down (65.32) by 4.6 points, while attention alone adds only 2.1 over the baseline. That is a very large effect for a caption feature and worth treating with care.
- The paper's full model scores 66.82 on validation (Table 3) but 70.67 on test-standard (Table 6). A 3.9-point improvement from val to test is unusual unless the final model was also trained on val. The paper does not say whether it was.
- The paper's captioner sees every COCO image with 5 human captions. If the VQA test images (COCO) overlap with the captioner's training images, the caption carries ground-truth-like information. §5 shows how much caption quality drops on unseen images; we excluded seen images to avoid exactly this.

## 7. Conclusions and next steps

1. **On 40k VQA v2 questions, the current QD captioner does not improve VQA accuracy** with any of the three proposed fusions. The best model is the caption-free A/C network (46.29 ± 0.16). All A/B/C variants are about 4 points above the current SimpleVQA (41.79), so for the thesis pipeline any of them is a better Stage-2 base than SimpleVQA.
2. **The bottleneck is the captioner, not the fusion.** B proves a VQA model can be made to depend strongly on the caption (−11 points when shuffled). C proves answers can be copied from it. But on unseen images the caption contains the correct answer only ~18% of the time (beam 1).
3. To give captions a real chance, the most promising steps are, in order:
   - Train the captioner on far more QD caption data. The generator has so far covered only the first 20k VQA train questions.
   - Re-run this benchmark with the improved captioner. The scripts take any captioner checkpoint (`captioner_ckpt`), and features are recomputed in about 15 minutes.
   - Scale VQA training data (the full train split). The no-caption baseline will rise too, so the caption gain should be measured as with/without at the same scale, as done here.
4. If you continue with one architecture: **C** is the most informative, because its pointer rate and gate show directly whether captions become useful. **B** stands to benefit most from better captions, since it already reads them closely.

## 8. Reproduce

```bash
cd CaptionVQA
python make_split.py --config configs/small.yaml          # split (excludes captioner images)
python prepare_features.py --config configs/small.yaml    # regions + 3-beam captions (~15 min)
python train.py --config configs/small.yaml --model A [--no-caption] --seed 0   # A | B | C
bash scripts/run_all.sh                                   # all 18 runs (~70 min)
python summarize.py --config configs/small.yaml           # tables in §3/§4
python caption_stats.py --config configs/small.yaml       # table in §5
# SimpleVQA baseline on the same split (from SimpleVQA/)
python ../CaptionVQA/precache_simplevqa.py --config configs/caption_vqa_small.yaml
python train.py --config configs/caption_vqa_small.yaml --fresh
python ../CaptionVQA/score_simplevqa.py --config configs/caption_vqa_small.yaml
```

Per-run outputs: `outputs/small/runs/<model>[_nocap]_s<seed>/{results.json, history.json, test_predictions.json, best.pt}`.
