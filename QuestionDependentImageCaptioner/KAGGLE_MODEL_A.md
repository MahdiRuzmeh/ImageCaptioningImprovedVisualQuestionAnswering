# Model A on Kaggle — MMTransformerCopy + BUTD, full QD captions

Model A is the from-scratch transformer: BUTD regions and question words go through one joint
encoder, then a decoder with a copy head writes the caption. It is trained on the full QD set
(`caption_quality_verdict == "ok"`: 277,318 rows, 79,568 images). Two sizes run:

| Run | Layers | FFN | Params |
|---|---:|---:|---:|
| `a_full_butd` | 3 | 1024 | ~20M |
| `a_full_butd_large` | 6 | 2048 | ~50M |

Split (`scripts/prepare_full_qd.py`, image-disjoint, rows sorted by `image_id`):
train 264,318 rows / 75,872 images · val 3,003 / 860 · test 9,997 / 2,836.
Model B (BLIP) uses the same file and split, so the test tables can be compared directly.

## Inputs

1. **QD captions**: upload `dataset/vqa_v2_qd_captions_full_train2014.json` (77 MB) as a Kaggle dataset.
   The raw generator file also works; `load_qd_json` applies the same `ok` filter.
2. **COCO train2014 images**: the `mnhlfuch/vqa-v2` dataset you already use.
3. **Internet on**: needed once, for the BUTD weights (`unc-nlp/frcnn-vg-finetuned`, 260 MB) and NLTK WordNet (METEOR).

The script finds the inputs by file name under `/kaggle/input`, so dataset slugs don't matter.

## Notebook cells

```bash
!git clone -b feat/captioner-a-mmt-butd-full --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/src/QuestionDependentImageCaptioner
!STAGES="extract" bash scripts/kaggle_model_a.sh
```

## Plan for the 12 h session / 30 h week limits

Run as **Save Version → Save & Run All**, so the output stays even if you close the browser.
Split the work so no version gets near 12 h.

| Version | `STAGES` | Attach | Expected time | Output to keep |
|---|---|---|---|---|
| 1 | `extract` | QD JSON, COCO | ~3.5–4 h (4.4 img/s on the 3070; T4 ≈ 3 img/s per GPU) | `butd_packed/` (~12 GB) → make it a dataset |
| 2 | `train_a` | QD JSON, COCO, v1 output | ~1.5 h (15 epochs) + 2 min test | `model_a/a_full_butd/` |
| 3 | `train_a_large` | same as 2 | ~3 h | `model_a/a_full_butd_large/` |

Versions 2 and 3 are independent, so you can run them at the same time on your two accounts.
The `butd_packed` dataset must then be shared with the second account.
All of model A takes about 8 GPU-hours.

Resuming: if a version stops early, attach its output to the next one. The script copies
`butd_packed/` and `model_a/<run>/*.pt` back in. Extraction skips finished images, and
training continues from `last.pt` (saved every epoch).

## What to bring back

From each `model_a/<run>/`: `best.pt`, `test_metrics.json`, `train.log`, `run_config.yaml`
(and `model_a/results.md`). `last.pt` is only needed if training must continue.

`results.md` reports BLEU-4, METEOR, ROUGE-L, CIDEr-D and AnswerHit on the test split (beam 3).
- AnswerHit: the share of non-yes/no rows whose answer is in the caption.
- METEOR: NLTK `meteor_score`, close to but not the same as the Java coco-caption version.
- Rebuild the table locally: `python scripts/results_table.py <test_metrics.json files>`.

## Local smoke test (RTX 3070)

```bash
python scripts/extract_visual.py --config configs/a_smoke_local.yaml --packed-dir outputs/butd_packed_smoke --limit 320 --batch-size 1
python train.py --config configs/a_smoke_local.yaml --fresh
python benchmark.py --config configs/a_smoke_local.yaml --ckpt outputs/a_smoke/best.pt --name a_smoke
```

## Packed feature cache

`packed_cache.py` stores every image in one memmap: `feats.f16` (N × 37 × 2048), `boxes.f16`,
`ntok.i16` and a `done.u8` flag for resuming. The old per-image `.pt` cache keeps every image in
each process's RAM (~12 GB × 2 DDP processes), which does not fit on Kaggle. The memmap is read
through the OS page cache, so both processes share one copy. `CachedVisualEncoder` switches to the
packed reader when a cache dir contains `meta.json`, so old `.pt` caches still work.
