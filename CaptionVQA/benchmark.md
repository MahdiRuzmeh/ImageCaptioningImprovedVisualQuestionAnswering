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

Split files: `outputs/small/split/{train,dev,test}.json`. Reproduce: see §10.

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
| Captioner training mode | Pre-trained together with the VQA module for 25 epochs (paper §5) | Frozen |
| What VQA receives from the captioner | Captioner LSTM hidden state at the last word, average-pooled over 5 captions (a 512-d learned feature, not text), fused by element-wise product | Caption tokens + decoder states (A, B), n-grams of 3 beams (C) |
| Can the captioner have seen the evaluated images? | Likely for the val ablation (Table 3): COCO human captions exist for val2014 | No, all captioner images excluded |
| Seeds / variance | 1 run, no std | 3 seeds, mean ± std |

What **is** comparable is the *relative* effect of the caption. The paper reports +7.35 points (test-std) from adding captions to the attention model. Here the caption adds nothing measurable, across three different fusion designs and 3 seeds. Points to keep in mind when citing the paper:

- In the paper's Table 6, the caption branch alone (Baseline + Cap, 69.91) beats Bottom-Up/Top-Down (65.32) by 4.6 points, while attention alone adds only 2.1 over the baseline. That is a very large effect for a caption feature and worth treating with care.
- The paper's full model scores 66.82 on validation (Table 3) but 70.67 on test-standard (Table 6). A 3.9-point improvement from val to test is unusual unless the final model was also trained on val. The paper does not say whether it was.
- The paper's captioner sees every COCO image with 5 human captions. If the VQA test images (COCO) overlap with the captioner's training images, the caption carries ground-truth-like information. §5 shows how much caption quality drops on unseen images; we excluded seen images to avoid exactly this.
- The paper's "Baseline + Att" (63.32 test-std) is below Bottom-Up/Top-Down (65.67), although it is built the same way. Part of the caption gain may come from adding a branch to an under-tuned baseline. Here the no-caption model is the strongest model.
- Table 6 lists Anderson et al. (65.32) and Teney et al. (65.67) with identical per-type numbers.

### 6.1 Second base paper (Zhou, Liu & Zhao 2024, *A visual question answering model based on image captioning*, IC-VQA)

A question-guided Transformer captioner (Faster R-CNN objects + box geometry + ResNet grid features) produces a caption that is multiplied element-wise with the fused question/visual feature of an Attention+LSTM answer decoder.

| Paper result | Number | Yes/No | Other | Overall |
|---|---|---|---|---|
| VQA 1.0, IC-VQA (Table 2) | 42.53 | 84.12 | 57.81 | 70.10 |
| VQA 2.0, IC-VQA (Table 3) | 46.83 | 87.20 | 61.98 | 70.85 |
| VQA 2.0 ablation (Table 5): no Obj / Att-obj / Cap / Att-w → only Cap → all four | | | | 62.80 → 64.72 → 70.85 |

Differences from this benchmark, and points to keep in mind:

- The captioner is trained on MS-COCO with 118k training images × 5 human captions. The COCO 2017 train split contains most val2014 images, i.e. most VQA v2 val images. The paper does not say which VQA split its numbers are on.
- The paper does not explain how a question-specific captioner is supervised when COCO captions are question-agnostic.
- Full VQA v2 train, 1 run, no std, no code.
- The same model has Yes/No = 84.12 in Table 2 but 82.3 in Table 4 (VQA 1.0).
- Table 3 compares against BLIP-2's **zero-shot** score (65.0); fine-tuned BLIP-2 is above 80.

### 6.2 Consistency check of the reported numbers

VQA "Overall" is the question-weighted mean of Number, Yes/No and Other, so it is fixed by the per-type numbers. We fitted the type weights by least squares on the rows both papers copy from other work (VQA 2.0 test-std: Goyal et al. prior / language-only / LSTM+CNN / MCB, Teney et al.; VQA 1.0 test-std: 7 rows from Table 5 / Table 2). The fit reproduces those rows to within 0.00 (VQA 2.0) and 0.03 (VQA 1.0) points, and also reproduces the Up-Down ensemble (70.34) and Wu & Mooney (69.66) rows of IC-VQA Table 3 exactly. Fitted weights (Number, Yes/No, Other): VQA 2.0 0.109 / 0.415 / 0.476, VQA 1.0 0.119 / 0.405 / 0.484. Applied to the papers' own rows:

| Row | Reported overall | Implied by per-type numbers | Gap |
|---|---|---|---|
| P1 VQA 2.0 Baseline | 61.21 | 62.62 | −1.41 |
| P1 VQA 2.0 Baseline + Att | 63.32 | 64.46 | −1.14 |
| P1 VQA 2.0 Baseline + Cap | 69.91 | 68.68 | +1.23 |
| P1 VQA 2.0 Baseline + Att + Cap | 70.67 | 69.75 | +0.92 |
| P1 VQA 1.0 Baseline + Att + Cap | 70.35 | 68.39 | +1.96 |
| P2 VQA 2.0 IC-VQA | 70.85 | 70.78 | +0.07 |
| P2 VQA 1.0 IC-VQA (Table 2) | 70.10 | 67.12 | +2.98 |

- In Sharma & Jalal, rows without captions are below the overall their per-type numbers imply, and rows with captions are above it. From the per-type numbers, the caption gain on VQA 2.0 is about +5.3 (64.46 → 69.75), not +7.35.
- IC-VQA's VQA 1.0 overall (70.10) is higher than CAM's (69.29), although IC-VQA is 2.4 points lower on Yes/No and 2.6 lower on Other. This is not possible with a weighted mean.

These checks do not show the papers are wrong, but both should be cited with care.

## 7. Related work: captions in the VQA pipeline

| Work | Setup | Effect of captions |
|---|---|---|
| Wu, Hu & Mooney, ACL 2019 ([arXiv 1906.00513](https://arxiv.org/abs/1906.00513)), cited by IC-VQA | Question-relevant captioner trained **jointly** with Up-Down VQA. For each question, the caption loss is applied only to the human COCO caption whose gradient agrees most with the predicted answer. Then 5 captions are generated per question for train/val/test and the model is fine-tuned | VQA 2.0 val (Table 2): Up-Down 63.2 → 64.6 (generic Up-Down captions) → **65.8** (generated question-relevant) → 67.1 (human captions). Test-std 65.3 → 68.4. Human captions alone, without the image: 59.6 |
| Kim & Bansal, ACL 2019 ([arXiv 1906.06216](https://arxiv.org/abs/1906.06216)) | Generated paragraph captions + image (Visual Genome QA) | Test 44.68 → 46.86. Human paragraphs only about 0.6 higher than generated ones (val) |
| Li et al., *Tell-and-Answer*, EMNLP 2018 ([D18-1164](https://aclanthology.org/D18-1164/)) | Attributes + generic captions replace the image | Comparable to baselines. The gain is explainability, not accuracy |
| Yang et al., PICa, AAAI 2022 ([arXiv 2109.05014](https://arxiv.org/abs/2109.05014)) | Captions/tags + in-context examples → GPT-3 | +8.6 over supervised SOTA on OK-VQA. The caption is the LLM's only view of the image |
| Tiong et al., PNP-VQA, 2022 ([arXiv 2210.08773](https://arxiv.org/abs/2210.08773)) | Question-guided captions → frozen QA language model, zero-shot | SOTA zero-shot VQAv2/GQA at the time. VQA accuracy correlates with the captions' **answer hit rate**, the measure of §5 |
| Özdemir & Akagündüz, CVPRW 2024 ([arXiv 2404.08589](https://arxiv.org/abs/2404.08589)) | Question-keyword-driven captions as LLM prompts | Competitive zero-shot results on GQA |

What follows for this benchmark:

1. With a classic VQA model that already sees the regions, the credible gain from captions is about **2–3 points**. It requires a captioner trained on large human caption data (≈600k COCO captions), preferably jointly with the VQA loss. Gains of 6–7 points (both base papers) are outliers.
2. Question-relevant captions help more than generic ones (Wu & Mooney: 65.8 vs 64.6), which supports the QD captioner design.
3. Captions help much more when the answering model is a text-only LLM (PICa, PNP-VQA), because there they are the only visual input. That is a different setting from A/B/C.
4. The useful signal is the extra caption supervision, not the caption as such. A caption computed from the same regions and question adds no new information (model B, §4). Our captioner adds 9.5k LLM captions (beam-1 hit rate 17.8%, §5), about 60× less human-quality supervision than these papers.

## 8. Upper bound: COCO human captions

Same split, models and training as §3. The only change is the caption input: the frozen QD captioner's beams are replaced by COCO human captions (5 per image) of the same image (`prepare_human_captions.py`).

- **Selection:** the 3 captions with the most overlap with the question's non-stopword words. Ties keep COCO order. The answer is never used.
- **Encoding:** captions use the captioner vocabulary (6.5% of tokens map to `<unk>`). Each caption is teacher-forced through the frozen captioner to get the token hidden states (used by B and C) and a length-normalized log-prob score (the same normalization as beam search).
- **Answer coverage on test** (§5 metric, first caption / any of 3): all 14.5 / 21.6, yes/no 0.6 / 1.0, number 11.1 / 18.4, other 25.6 / 37.8. Generated captions reach a higher "number" coverage (27.2 / 51.6), but many of those numbers are wrong for the question.

These captions were written by people looking at the test images. This is therefore an **upper bound** on what a better captioner could give, not a comparable result. Wu & Mooney (2019) report the same condition (human 67.1 vs generated 65.8, §7).

Test accuracy, mean ± std over seeds 0/1/2:

| Model | No caption | Generated | Human | Δ human − none | Δ shuffle (generated → human) |
|---|---|---|---|---|---|
| A | 46.29 ± 0.16 | 45.89 ± 0.58 | 46.70 ± 0.48 | +0.41 | 0.12 → 0.91 |
| B | 45.36 ± 0.21 | 45.61 ± 0.21 | 46.32 ± 0.38 | +0.96 | 11.12 → 7.70 |
| **C** | 46.29 ± 0.16 | 45.64 ± 0.49 | **48.89 ± 0.19** | **+2.60** | 1.98 → 5.73 |

Per answer type, C with human captions vs. no caption: yes/no 67.18 → 66.46, **number 31.91 → 39.20**, **other 34.59 → 38.42**. Dev agrees (48.55 ± 0.24 vs 45.56 ± 0.13).

What this shows:

1. **The fusion can use good captions.** C gains +2.6 points, about 15× the seed std. The gain comes entirely from number and other questions, where captions can contain the answer.
2. **C gains through the pointer, not the attention path.** With human captions the fusion gate falls to about 0.03 (0.29–0.32 with generated captions), while the pointer stays active (0.14–0.19, vs 0.08–0.13 on shuffled captions). Copying answer words from the caption is what helps.
3. **A barely gains** (+0.4, within noise). Attending over a mean caption vector does not extract a specific answer word.
4. **B reads both caption types closely but gains only +1.0.** Its shuffle drop is large for generated captions too (11.1), so dependence on the caption is not the same as useful information.
5. **The gap between the current captioner and human captions is 3.25 points for C** (45.64 → 48.89). Retraining the captioner (§9) should be measured against this range.

## 8.1 Captioner-seen data (leakage condition)

Same models, trained and/or tested on the images and (image, question) pairs the captioner was trained on, where captions contain the answer about twice as often (beam 1: 33.9% on trained pairs vs 17.8% unseen). Full report: [benchmark_seen.md](benchmark_seen.md).

- Trained and tested on seen data, **C gains +2.6 overall and +4.6 on the captioner's own training pairs** (A 0.0, B +0.9).
- Models trained on unseen images gain nothing on seen data (C +0.3): they learned not to trust captions.
- Models trained on seen data lose on unseen images (C −0.6): they over-trust captions.

## 9. Conclusions and next steps

1. **On 40k VQA v2 questions, the current QD captioner does not improve VQA accuracy** with any of the three proposed fusions. The best model is the caption-free A/C network (46.29 ± 0.16). All A/B/C variants are about 4 points above the current SimpleVQA (41.79), so for the thesis pipeline any of them is a better Stage-2 base than SimpleVQA.
2. **The bottleneck is the captioner, not the fusion.** B proves a VQA model can be made to depend strongly on the caption (−11 points when shuffled). C proves answers can be copied from it. But on unseen images the caption contains the correct answer only ~18% of the time (beam 1). With COCO human captions, the same model C gains +2.6 points (§8), which confirms that caption quality is the limit.
3. To give captions a real chance, the most promising steps are, in order:
   - Train the captioner on far more QD caption data. The generator has so far covered only the first 20k VQA train questions.
   - Re-run this benchmark with the improved captioner. The scripts take any captioner checkpoint (`captioner_ckpt`), and features are recomputed in about 15 minutes.
   - Scale VQA training data (the full train split). The no-caption baseline will rise too, so the caption gain should be measured as with/without at the same scale, as done here.
4. If you continue with one architecture: **C** is the most informative, because its pointer rate and gate show directly whether captions become useful. **B** stands to benefit most from better captions, since it already reads them closely.
5. To get closer to the setting where captions are known to help (§7): train the captioner jointly with the VQA loss instead of freezing it (Wu & Mooney 2019), and add COCO human captions as caption supervision. When comparing with the base papers, report both conditions: captioner images overlapping the VQA evaluation images vs. disjoint (this benchmark). The difference measures how much of the published gain is leakage.

## 10. Reproduce

```bash
cd CaptionVQA
python make_split.py --config configs/small.yaml          # split (excludes captioner images)
python prepare_features.py --config configs/small.yaml    # regions + 3-beam captions (~15 min)
python train.py --config configs/small.yaml --model A [--no-caption] --seed 0   # A | B | C
bash scripts/run_all.sh                                   # all 18 runs (~70 min)
python summarize.py --config configs/small.yaml           # tables in §3/§4
python caption_stats.py --config configs/small.yaml       # table in §5
# Upper bound with COCO human captions (§8)
python prepare_human_captions.py --config configs/small.yaml   # captions_{split}_human.pt
bash scripts/run_human.sh                                 # 9 runs (~60 min)
python caption_stats.py --config configs/small.yaml --captions human
# SimpleVQA baseline on the same split (from SimpleVQA/)
python ../CaptionVQA/precache_simplevqa.py --config configs/caption_vqa_small.yaml
python train.py --config configs/caption_vqa_small.yaml --fresh
python ../CaptionVQA/score_simplevqa.py --config configs/caption_vqa_small.yaml
```

Per-run outputs: `outputs/small/runs/<model>[_nocap|_human]_s<seed>/{results.json, history.json, test_predictions.json, best.pt}`.
