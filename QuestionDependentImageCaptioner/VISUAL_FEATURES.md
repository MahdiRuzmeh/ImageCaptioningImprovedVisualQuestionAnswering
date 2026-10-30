# Visual-feature comparison for the QD captioner (V0–V3)

Branch: `feat/qd-captions-coco-context` · Date: 2026-10-30

## Question

`VISUAL_GROUNDING.md` showed that the captioner is grounded for objects/sports/animals, weakly grounded
for colors, and blind for counting and yes/no. Is the visual feature extractor (frozen torchvision
COCO Faster R-CNN, 32 RPN regions, 1024-D box-head features, no box coordinates) the cause?

## Setup

Only the visual tokens change. Everything else is the same as `kaggle_mini`:

- `MMTransformerCopyCaptioner`
- QD captions only, no COCO pretraining
- image-disjoint split: train rows 0–9500 (9,502 rows), val 9500–10000, test 10000–10500 (494 rows)
- 20 epochs, lr 3e-4, best epoch by val CIDEr
- seeds 42/43/44
- local RTX 3070

The baseline was retrained the same way, so all five rows are comparable.

New visual tokens come from a cache (`scripts/extract_visual.py` → `models/visual_encoder.py`). Each token is
`LN/Linear(feature) + Linear(x1, y1, x2, y2, area)`. For grid features the box is the patch cell, so the
box embedding acts as a 2D position.

| Variant | Extractor (frozen) | Tokens × dim | Config |
|---|---|---|---|
| base | torchvision Faster R-CNN R50-FPN (COCO), top-32 RPN proposals, 448×448 squash, no boxes | 32 × 1024→2048 | `vis_base.yaml` |
| V0 | same detector, aspect ratio kept, 31 proposals + whole-image box, + boxes | 32 × 1024 | `vis_v0_frcnn_box.yaml` |
| V1 | bottom-up attention Faster R-CNN R101 (Visual Genome, 1,600 objects + 400 attributes), 36 detections + mean token, + boxes | 37 × 2048 | `vis_v1_butd.yaml` |
| V2 | SigLIP so400m/14-384 patch tokens, 27×27 → avg-pool 12×12, + MAP pooler token | 145 × 1152 | `vis_v2_siglip.yaml` |
| V3 | DINOv2-L/14 with registers, 448×448, 32×32 → 12×12, + CLS | 145 × 1024 | `vis_v3_dinov2.yaml` |

V1 runs the PyTorch port of the BUTD detector vendored in `models/third_party/butd` (LXMERT example,
Apache-2.0) with the `unc-nlp/frcnn-vg-finetuned` weights. Extraction uses batch size 1. The upstream
`norm_box` divided by the wrong image size for batches > 1 (fixed locally), and padding in a batch changes the detections.

## Results (test, mean ± std over 3 seeds, `scripts/vis_compare.py`)

AC = AnswerConsistency: yes/no → polarity, number → count, otherwise the answer word is in the caption.
ImageDep = AC(real image) − AC(another image, same question). It measures how much the caption uses the image.
Test sizes: yes/no 134, number 92, color 73, other 195.

| Metric | base | V0 boxes | **V1 BUTD** | V2 SigLIP | V3 DINOv2 |
|---|---|---|---|---|---|
| AnswerConsistency (all) | **0.372 ± 0.011** | 0.351 ± 0.012 | 0.370 ± 0.008 | 0.360 ± 0.014 | 0.365 ± 0.002 |
| PolarityAcc (yes/no) | 0.577 ± 0.004 | 0.547 ± 0.030 | **0.582 ± 0.049** | 0.542 ± 0.041 | 0.552 ± 0.034 |
| CountAcc (number) | 0.257 ± 0.045 | **0.264 ± 0.023** | 0.228 ± 0.038 | **0.264 ± 0.041** | 0.225 ± 0.017 |
| AC color | 0.237 ± 0.016 | 0.251 ± 0.034 | **0.406 ± 0.021** | 0.256 ± 0.034 | 0.306 ± 0.008 |
| AC other | **0.337 ± 0.008** | 0.294 ± 0.008 | 0.277 ± 0.009 | 0.320 ± 0.006 | 0.325 ± 0.021 |
| ImageDep (all) | 0.128 ± 0.007 | 0.114 ± 0.022 | **0.138 ± 0.004** | 0.123 ± 0.006 | 0.135 ± 0.013 |
| ImageDep yes/no | 0.022 ± 0.034 | −0.012 ± 0.030 | 0.010 ± 0.017 | 0.000 ± 0.039 | 0.000 ± 0.020 |
| ImageDep number | 0.025 ± 0.017 | **0.051 ± 0.017** | −0.011 ± 0.000 | 0.025 ± 0.027 | −0.014 ± 0.023 |
| ImageDep color | 0.027 ± 0.014 | 0.078 ± 0.055 | **0.274 ± 0.024** | 0.059 ± 0.070 | 0.169 ± 0.029 |
| ImageDep other | **0.287 ± 0.005** | 0.244 ± 0.026 | 0.244 ± 0.003 | 0.279 ± 0.011 | 0.285 ± 0.026 |
| CIDEr (QD references) | 4.91 ± 0.08 | 4.84 ± 0.07 | 4.85 ± 0.02 | 4.87 ± 0.10 | 4.86 ± 0.04 |

Color examples (seed 42, same test questions):

| Question → answer | base | V1 BUTD | V3 DINOv2 |
|---|---|---|---|
| What color is the boat? → yellow | the boat is red | **the boat is yellow** | the boat is white |
| What color is the dog's collar? → brown | the dog's collar is red | the dog's collar is red | **the dog's collar is brown** |
| What color is this man's shirt? → blue | … is white | … is black | … is white |

## What we learned

1. **Features matter for attributes.** BUTD, trained with Visual Genome attribute labels, improves color from
   0.237 to **0.406** (+0.17, about 8 std). Color goes from almost blind (ImageDep 0.027) to clearly grounded
   (0.274). DINOv2 gives half of that gain (0.306 / 0.169).
2. **No extractor fixes counting or yes/no.** ImageDep stays at about 0 for every variant. More boxes,
   attribute features, dense DINOv2 grids and text-aligned SigLIP tokens all give the same result. So the visual input is not
   what limits these types. Likely causes are the 9.5k training rows and the caption targets: the model learns
   the frequent answer ("two", "not") from the question. This needs another fix, such as a counting module, more data or
   an answer loss.
3. **Overall AnswerConsistency does not change** (0.36–0.37, within noise). BUTD trades its color gain for a
   loss on "other" (0.337 → 0.277): its 36 VG detections lose scene context that the RPN regions and grids keep.
4. **Box coordinates alone (V0) do not help.** V0 is slightly worse than the baseline (0.351 vs 0.372), mostly on "other".
5. **SigLIP is not better than the baseline here.** It is text-aligned, but its frozen, pooled tokens give no grounding gain with
   9.5k rows. The 12×12 avg-pool and the missing adapter may be part of the reason. This was not tested further.

## V1 + V3 combined (BUTD + DINOv2 tokens)

`configs/vis_v13_butd_dinov2.yaml`: 37 BUTD tokens + 145 DINOv2 tokens. Each source has its own LN/Linear and
source embedding, and the box embedding is shared (`MultiCachedVisualEncoder`).

| Metric (test, 3 seeds) | base | V1 BUTD | V3 DINOv2 | **V1 + V3** |
|---|---|---|---|---|
| AnswerConsistency (all) | 0.372 ± 0.011 | 0.370 ± 0.008 | 0.365 ± 0.002 | 0.371 ± 0.025 |
| CountAcc | 0.257 ± 0.045 | 0.228 ± 0.038 | 0.225 ± 0.017 | 0.268 ± 0.023 |
| AC color | 0.237 ± 0.016 | **0.406 ± 0.021** | 0.306 ± 0.008 | 0.384 ± 0.036 |
| AC other | **0.337 ± 0.008** | 0.277 ± 0.009 | 0.325 ± 0.021 | 0.285 ± 0.019 |
| ImageDep color | 0.027 | **0.274** | 0.169 | 0.251 |
| ImageDep number / yes/no | 0.025 / 0.022 | −0.011 / 0.010 | −0.014 / 0.000 | 0.022 / 0.000 |

The combination does **not** recover "other". The model behaves like BUTD alone, with more seed variance.
So **V1 BUTD alone** goes to the downstream check (seed 44, best val CIDEr 4.59).

## Downstream: CaptionVQA with BUTD captions

Captions come from V1 BUTD seed 44, which was picked by val CIDEr, not by test. They cover the CaptionVQA
train/dev/test images (BUTD features extracted for all 10.5k images). The VQA models' own image features
do not change, so only the captions differ. Config: `CaptionVQA/configs/small_butd.yaml` and
`scripts/run_butd.sh`, 3 seeds, test on val2014. Model B reuses the old captioner's region layers, so it cannot use this captioner and is not run.

| Test acc (3 seeds) | No caption | Old QD | ft_base (COCO→QD) | **BUTD (QD only)** | Human COCO |
|---|---|---|---|---|---|
| A | 46.29 ± 0.16 | 45.89 ± 0.58 | 46.29 ± 0.13 | **47.28 ± 0.11** | 46.70 ± 0.48 |
| C | 46.29 ± 0.16 | 45.64 ± 0.49 | 46.73 ± 0.12 | **47.32 ± 0.11** | 48.89 ± 0.19 |

By answer type, model C (yes/no · number · other): no caption 67.2 · 31.9 · 34.6, ft_base 67.6 · 32.3 · 35.1,
**BUTD 67.7 · 31.3 · 36.4**, human 66.5 · 39.2 · 38.4.
Shuffle drop (test acc minus test acc with another image's caption): C 2.78 and A 1.84 for BUTD, against 1.78 / 0.33 for ft_base.
So the VQA model really uses the BUTD captions.

- BUTD captions give **+1.0** over no caption for both A and C (about 8 std). That is the best generated-caption result so far,
  and it beats the COCO-pretrained ft_base captioner without any COCO pretraining. A with BUTD is even above A with human captions.
- The gain is on "other" (+1.8) and yes/no; number does not improve, which matches the probe (counting is not grounded).
- Human captions are still 1.6 higher for C, mainly on number and other.

### Oracle captions (upper bound, cheat)

`CaptionVQA/prepare_oracle_captions.py`: the gold answer (majority, normalized) is put in front of every BUTD
caption ("tennis they are playing tennis"), in train, dev and test. 88% of the answers are in the captioner vocab;
the rest become `<unk>`. This uses the answer at test time, so it only measures the ceiling.

| Test acc (3 seeds) | No caption | BUTD | Human | **Oracle** | yes/no · number · other (oracle) |
|---|---|---|---|---|---|
| A | 46.29 | 47.28 | 46.70 | **82.89 ± 0.10** | 100.0 · 82.7 · 70.3 |
| C | 46.29 | 47.32 | 48.89 | **87.41 ± 0.06** | 100.0 · 83.4 · 79.1 |

With another image's oracle caption, accuracy falls to 10–18%. So the models trust the caption almost completely.

- **The VQA models can use a caption that states the answer**, so fusion is not the bottleneck. **What the
  captioner says is the bottleneck**: the answer is in the beam-1 BUTD caption for only 20% of test questions (33% in any beam).
  C reaches 79% on "other", against 70% for A, because the pointer copies the open-vocabulary answers that are not in the A answer list.
- Caveat: a caption that contains the answer means the captioner has answered the question. So the remaining
  gain needs better answer-relevant grounding in the captioner (counting, yes/no), not just better caption fluency.

## Suggested next steps

- **COCO pretraining → QD fine-tuning with BUTD features** (the plan for the chosen extractor). This needs BUTD features
  for COCO train2014 (~83k images, about 10–11 h at batch size 1 locally, ~12 GB of cache) or a Kaggle run.
- **Counting and yes/no:** these are not visual-feature problems. Options: a count module on the box features
  (Zhang et al., ICLR 2018), an auxiliary answer loss, or more data (COCO pretraining → QD fine-tuning on the chosen extractor).

## Reproduce

```bash
cd QuestionDependentImageCaptioner
python scripts/extract_visual.py --config configs/vis_v1_butd.yaml --batch-size 1      # V0: vis_v0_frcnn_box
HF_HOME=E:/hf_cache python scripts/extract_visual.py --config configs/vis_v3_dinov2.yaml  # V2: vis_v2_siglip
bash scripts/run_vis.sh v1_butd configs/vis_v1_butd.yaml     # 3 seeds train + test probe
python scripts/vis_compare.py base v0_frcnn_box v1_butd v2_siglip v3_dinov2
```

Weights: BUTD `outputs/butd_weights/` (`unc-nlp/frcnn-vg-finetuned`, from the old Hugging Face S3 bucket),
SigLIP/DINOv2 from the Hugging Face Hub. On Windows, `HF_HOME` must be a short path (MAX_PATH).
