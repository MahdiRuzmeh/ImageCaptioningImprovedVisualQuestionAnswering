# CaptionVQA

Caption-aware VQA models — Stage 2 of the thesis pipeline (*Image captioning improved visual question answering*, Sharma & Jalal, 2021).

## Why this folder exists

Stage 1 (`../QuestionDependentImageCaptioner`) produced a captioner that, given an image **and a question**, writes a sentence about the part of the image the question asks about (e.g. *"What color is the umbrella?"* → *"the umbrella is red"*). The thesis question is: **does feeding such captions into a VQA model make it answer more accurately?**

`../SimpleVQA` was the first Stage-2 model, but it only gives a partial answer to that question:
- It generates answers word by word with an LSTM, which is slow and weaker than the usual VQA approach of choosing from a list of answers.
- It only uses one averaged vector from the captioner, not the caption text.
- It has no fair "without caption" baseline to compare against.

CaptionVQA answers the question properly:

1. **Three different ways of using the caption** (A, B, C below), each also trained **without** the caption, so the caption's effect is measured directly.
2. **A clean VQA v2 benchmark.** Train, dev and test use different images. VQA training never uses images the captioner was trained on, so captions at train time are as good (or as bad) as at test time.
3. **Diagnostics that show whether a model actually reads the caption**: swapping in another image's caption at test time, gate values, and how often the answer is copied from the caption.

Inputs: question, image (32 Faster R-CNN region features), caption. Output: answer.

## Models

| | File / class | Idea |
|---|---|---|
| **A** | `models.py` / `GatedUpDnVQA` | The question attends over image regions and over caption words. A learned gate decides how much to trust the image vs the caption |
| **B** | `models.py` / `CaptionerInitTransformerVQA` | One Transformer over `[CLS] + regions + question + caption`, starting from the captioner's own pretrained encoder |
| **C** | `models.py` / `CaptionPointerVQA` | A's fusion plus a pointer that can copy a 1–3-word answer from any of the 3 beam captions. The captioner's confidence decides how much to trust the copy |

All three use a classifier over the frequent answers, trained with soft-score BCE, and are scored with the official VQA v2 accuracy (`common.py`). `--no-caption` trains the same model without the caption.

## Results (short)

Test set: 10,038 VQA v2 val questions, 3 seeds. Full details, diagnostics and the comparison with the paper are in [benchmark.md](benchmark.md).

| Model | With caption | Without caption |
|---|---|---|
| A | 45.89 ± 0.58 | **46.29 ± 0.16** |
| B | 45.61 ± 0.21 | 45.36 ± 0.21 |
| C | 45.64 ± 0.49 | 46.29 ± 0.16 (same network as A) |
| SimpleVQA (old Stage 2) | 41.79 | – |

- A/B/C are about 4 points better than SimpleVQA.
- The current captioner (trained on ~2.9k images) does **not** improve accuracy yet. On new images its caption contains the right answer only ~18% of the time.
- The paper reports 70.67 on VQA v2 test-standard with full training data. See benchmark.md §6 for why the numbers are not directly comparable.

## Pipeline

```
VQA v2 questions/answers ──► make_split.py ──► outputs/small/split/{train,dev,test}.json
                                                   │
images ──► Faster R-CNN (captioner's RegionEncoder) ┤ prepare_features.py (once, ~15 min)
frozen QD captioner (kaggle_mini/best.pt) ──────────┘   regions_*.pt  + captions_*.pt (3 beams, scores, decoder states)
                                                   │
                                          train.py --model A|B|C [--no-caption]
                                                   │
                                     runs/<name>/results.json ──► summarize.py ──► benchmark.md
```

Every frozen part (detector, captioner) runs once. Training then reads cached tensors from RAM, so one run takes 1–6 minutes on an RTX 3070.

## Run

All commands are run from `CaptionVQA/` with the project venv (`..\.venv\Scripts\activate`).

```bash
python make_split.py --config configs/small.yaml          # image-disjoint split, excludes captioner images
python prepare_features.py --config configs/small.yaml    # regions + captions (needs GPU)
python train.py --config configs/small.yaml --model A     # or B / C, add --no-caption, --seed N
bash scripts/run_all.sh                                   # all 18 benchmark runs (~70 min)
python summarize.py --config configs/small.yaml           # result tables
python caption_stats.py --config configs/small.yaml       # how often captions contain the answer
```

SimpleVQA on the same split (run from `../SimpleVQA/`):

```bash
python ../CaptionVQA/precache_simplevqa.py --config configs/caption_vqa_small.yaml   # fill its caches in small batches
python train.py --config configs/caption_vqa_small.yaml --fresh
python ../CaptionVQA/score_simplevqa.py --config configs/caption_vqa_small.yaml      # same metric + test split
```

To test a **new captioner**, set `captioner_ckpt` in `configs/small.yaml`, delete `outputs/small/features/captions_*.pt`, and rerun `prepare_features.py` and the training runs. Region features are reused.

## Files

| File | Purpose |
|---|---|
| `configs/small.yaml` | Split sizes, captioner checkpoint, training hyper-parameters |
| `common.py` | Tokenizer, config loader, VQA v2 answer normalization, accuracy and soft targets |
| `make_split.py` | Train (train2014, without captioner images) / dev / test (val2014) question lists |
| `prepare_features.py` | Faster R-CNN regions + frozen captioner outputs (3 beams, scores, beam-1 decoder states) |
| `models.py` | Models A, B, C |
| `train.py` | Training, dev selection, test evaluation, shuffled-caption ablation |
| `summarize.py` | Mean ± std tables over seeds |
| `caption_stats.py` | Share of questions whose caption contains a correct answer |
| `precache_simplevqa.py`, `score_simplevqa.py` | Run and score the SimpleVQA baseline on the same split |
| `scripts/run_all.sh` | All benchmark runs |
| `benchmark.md` | Results, diagnostics, comparison with the paper |
| `make_seen_split.py`, `configs/seen.yaml`, `scripts/run_seen.sh` | "Seen" benchmark on the captioner's training images ([benchmark_seen.md](benchmark_seen.md)) |
| `eval_runs.py` | Evaluate trained runs on a split of another work dir (cross-condition tests) |

## Config reference (`configs/small.yaml`)

Paths are relative to `CaptionVQA/`.

| Key | Meaning |
|-----|---------|
| `train_questions_json` / `train_annotations_json` | VQA v2 train2014 questions / annotations |
| `val_questions_json` / `val_annotations_json` | VQA v2 val2014 questions / annotations (dev + test come from here) |
| `train_images_dir` / `val_images_dir` | COCO image folders |
| `qd_captions_json` | QD caption dataset. Only used to **exclude** its images from VQA train |
| `num_train_questions` / `num_dev_questions` / `num_test_questions` | Split sizes (whole images are added until the count is reached) |
| `captioner_ckpt` | Frozen Stage-1 captioner |
| `caption_beam` / `caption_max_len` | Number of beam captions kept / max caption length |
| `max_question_len`, `max_regions`, `image_size` | Same as the captioner |
| `work_dir` | Output folder (`outputs/small`, git-ignored) |
| `answer_min_count` | An answer joins the answer list if it is the majority answer at least this often in train |
| `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_epochs`, `dropout`, `grad_clip` | Training (AdamW, warm-up + cosine) |
| `pretrained_lr_scale` | Model B: learning-rate factor for layers copied from the captioner |
| `hidden_dim` | Fusion width (512) |

## Outputs (`outputs/small/`, not in git)

| Path | Content |
|---|---|
| `split/*.json` | Question records for train / dev / test |
| `region_cache/{train,val}/` | Raw Faster R-CNN features per image (captioner format; SimpleVQA also writes here) |
| `features/regions_*.pt`, `features/captions_*.pt` | Per-split tensors used by `train.py` |
| `features/captions_*_sample.json` | Example captions to inspect by eye |
| `runs/<model>[_nocap]_s<seed>/` | `results.json`, `history.json`, `test_predictions.json`, `best.pt` |
| `runs/SimpleVQA/` | SimpleVQA test/dev results and predictions |

## Hardware note

The machine has 16 GB of RAM and an 8 GB GPU. Run one GPU job at a time. Keep SimpleVQA's `num_workers` small: each data-loader worker holds a copy of the VQA JSON data, and too many workers make Windows grow the page file on C:.
