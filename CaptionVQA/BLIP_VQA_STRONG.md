# Stage A + B: stronger VQA model on train2014 + val2014, baseline vs + BLIP captions

Goal: a stronger baseline and the same model with question-dependent captions, compared on the **same 10,001
val2014 test questions** as all earlier experiments.

| | Before (VQA_REPORT.md) | Now |
|---|---|---|
| VQA model | Bottom-Up Top-Down | **MCAN-style co-attention transformer** (Yu et al. 2019, "small": 6 + 6 layers, d = 512) |
| Word vectors | random | **GloVe 300-d** |
| Answers | 5,862 (≥ 3 times) | **3,092** (≥ 9 times, ≈ the standard 3,129) |
| Training data | train2014: 443,757 questions | **train2014 + val2014 outside dev/test: 643,110 questions** |
| Schedule | AdamW, cosine, 12 epochs | MCAN: Adam (0.9, 0.98), lr 1e-4, 2 warm-up epochs, ×0.2 at epochs 10 and 12, 13 epochs |

## Models (`models_vqa.py`, class `MCANVQA`)

```
question ──► GloVe (300) ──► LSTM 512 ──► question tokens ──► ENCODER: 6 × self-attention
image ──► 37 BUTD regions × 2048 ──► Linear 512 (+ type "region")
[caption ──► same GloVe table ──► LSTM 512 ──► caption tokens (+ type "caption")]      ← caption model only
DECODER: 6 × ( self-attention over [regions ; caption tokens]
               + guided attention to the question tokens + FFN )
AttFlat(question) + AttFlat(regions ; caption) ──► LayerNorm ──► classifier (3,092 answers)
```

Baseline: 53.1M parameters; with captions: 54.8M. The caption tokens enter the decoder **as a sequence**: every
image region can attend to every caption word, and the question guides both.

## Data

- **Train:** all of train2014 (443,757 questions) plus the 199,353 val2014 questions on images outside dev/test, so 643,110 questions on 120,508 images.
- **Dev / test:** unchanged (5,000 / 10,001 val2014 questions, no shared images).
- **Captions:** cross-fitted as before. No captioner ever captions an image it trained on:

| Questions | Caption from |
|---|---|
| on fold-k train2014 images | the fold 1 − k model |
| on val2014 images (the captioners never trained on any of them) | half from each fold model (`data/crossfit_folds.json`, now covering all 40,504 val2014 images; the earlier entries are unchanged) |

The 458,758 captions from the previous crossfit run are **reused**; only the 199,353 new val2014 questions are captioned.

## Steps: 3 Kaggle notebooks

Use GPU T4 x2 and internet on for every notebook, run each as **Save Version → Save & Run All**, and never run two
GPU notebooks on the same account at once.

| # | Notebook | Needs | Time |
|---|---|---|---|
| 0 | val2014 feature cache | — | ~2 h |
| 1 | stronger baseline | output of 0 | ~6–8 h (see below) |
| 2 | stronger baseline + captions | output of 0 | ~7–9 h |

Notebooks 1 and 2 can run at the same time on your two accounts once notebook 0 is done.

### Notebook 0: BUTD features for all val2014 images

Input: `mnhlfuch/vqa-v2`.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/CaptionVQA
!bash scripts/kaggle_val2014_cache.sh
```

It ends with `val2014 cache: 40504/40504 images extracted` and `VAL2014_CACHE_DONE`. Then make its output a dataset:
Output tab → **New Dataset** (e.g. `qd-butd-val2014`). It holds `butd_val2014/` (~6 GB).

### Notebook 1: stronger baseline

Inputs:
- `mnhlfuch/vqa-v2`, `questiondependentcaptions`;
- `qd-butd-packed`, `qd-butd-val2014`;
- **a GloVe dataset:** any Kaggle dataset with `glove.6B.300d.txt` or `glove.840B.300d.txt`, e.g. search Datasets for "glove".

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/CaptionVQA
!MODE=baseline bash scripts/kaggle_vqa_strong.sh
```

### Notebook 2: stronger baseline + BLIP captions

Inputs: the same five as notebook 1, **plus**:
- the two fold notebooks' outputs (`model_b/b_blip_fold0/best.pt`, `model_b/b_blip_fold1/best.pt`);
- **`captions_crossfit.json`** (458,758 captions from the previous crossfit run), as any input: e.g. upload the local file `CaptionVQA/vqa_full_crossfit/captions/captions_crossfit.json` (21 MB) as a new dataset. Without it, all 658k questions are captioned again (+3 h).

Same cells, with `!MODE=crossfit bash scripts/kaggle_vqa_strong.sh`.

### What the log shows

- `GloVe /kaggle/input/.../glove…300d.txt` and `feature caches: …butd_packed …butd_val2014`;
- `dev/test identical to the earlier experiments; 658111 questions`;
- `copied 79568 images from …butd_packed`, `copied 37725 images from …butd_val2014`, then a small rest to extract;
- **crossfit only:** `reused 458758 captions, 199353 to generate`, and later `captions: 658111 … missing 0`;
- each seed's `[nocap_s0] TEST …` / `[blip_s0] TEST …`; at the end `VQA_STRONG_BASELINE_DONE` / `VQA_STRONG_CROSSFIT_DONE`.

### Time and seeds

The MCAN model is ~4× larger than UpDn and the data 1.45× larger, so one seed takes roughly 2.5–3.5 h on a T4.
- **Seeds 0 and 1** run in parallel, one per GPU.
- **Seed 2** runs afterwards only if enough of the 12 h is left. Otherwise the log says `skip seed 2`; run it later in its own session with `SEEDS="2"`.
- **To be safe,** start both notebooks with `SEEDS="0 1"`, check the minutes per epoch in `train_s0.log`, and add seed 2 later:

```bash
!SEEDS="0 1" MODE=baseline bash scripts/kaggle_vqa_strong.sh
```

## What to bring back

| Notebook | Folder | Files |
|---|---|---|
| 1 | `/kaggle/working/vqa_strong_baseline/` | `runs/*/results.json`, `history.json`, `test_predictions.json`, `summary.md`, `*.log` |
| 2 | `/kaggle/working/vqa_strong_crossfit/` | the same, plus `captions/captions_crossfit.json` |

`best.pt` files (~210 MB each) are not needed. The features are kept in `/tmp` and are not part of the output.

## Round 1b: batch 64 + GloVe 840B (+ caption dropout)

Config `configs/blip_strong_b64.yaml`: identical to `blip_strong.yaml` except **batch 64**. The runner prefers
`glove.840B.300d.txt` when a GloVe 840B dataset is attached. Caption dropout hides the caption for a share of the
training questions (caption model only; test questions always get their caption).

| Notebook | Inputs | Command (after the clone + `%cd /kaggle/working/repo/CaptionVQA` cells) | Time (estimate) |
|---|---|---|---|
| **1b-A** baseline | vqa-v2, questiondependentcaptions, qd-butd-packed, qd-butd-val2014, **GloVe 840B** | `!CONFIG=configs/blip_strong_b64.yaml RUN=vqa_b64_baseline SEEDS="0 1" MODE=baseline bash scripts/kaggle_vqa_strong.sh` | ~4.5 h |
| **1b-B** + captions, dropout 0.3 | the same 5 + **the 658k captions** (`captions_crossfit.json`) | `!CONFIG=configs/blip_strong_b64.yaml RUN=vqa_b64_crossfit_drop CAPTION_DROPOUT=0.3 SEEDS="0 1" MODE=crossfit bash scripts/kaggle_vqa_strong.sh` | ~5.5–6 h |
| **1b-C** + captions, no dropout | the same as 1b-B | `!CONFIG=configs/blip_strong_b64.yaml RUN=vqa_b64_crossfit SEEDS="0 1" MODE=crossfit bash scripts/kaggle_vqa_strong.sh` | ~5.5–6 h |

- **The 658k captions:** upload `CaptionVQA/vqa_strong_crossfit/captions/` (the folder) as a dataset. With all
  captions present, the fold captioner models are **not** needed: the log shows `reused 658111 captions, 0 to generate`.
- **Outputs:** `/kaggle/working/<RUN>/`, holding `runs/*/results.json`, `history.json`, `test_predictions.json` and now
  also `dev_scores.npy`, `test_scores.npy`, `answers.json` (for the ensemble), plus `summary.md` and logs.

## Ensemble (local)

`ensemble_vqa.py` averages the answer scores of several trained models. The caption-vs-baseline weight is chosen on
dev; test is only reported. Result for the 4 Stage A+B models (`vqa_ensemble.md`):

| Combination | Dev | Test |
|---|---:|---:|
| best single caption model (blip_s1) | 69.26 | 67.81 |
| baseline seeds averaged | 66.80 | 66.82 |
| caption-model seeds averaged | 69.64 | 68.57 |
| all 4 models, equal weight | 69.76 | 68.99 |
| **0.7 × caption models + 0.3 × baselines** | **70.26** | **69.11** |
