# Model B on Kaggle — BLIP-base + question prompt, full QD captions

Model B is the pretrained transformer: BLIP's ViT-B/16 image encoder (frozen) plus its BERT text
decoder, fine-tuned. The question goes in as a text prompt in front of the caption:

    [DEC] question: <question> caption: <caption> [SEP]      (loss only on the caption tokens)

So in every decoder layer, self-attention sees the question and cross-attention sees the image
patches. Data, the `verdict == ok` filter, the split, the metrics and the `test_metrics.json` format
are the same as model A ([KAGGLE_MODEL_A.md](KAGGLE_MODEL_A.md)), so `scripts/results_table.py`
puts A and B in one table.

Code: `train_blip.py` (train with DDP / `--continue` / time budget; `--test` for beam-3 test metrics),
config `configs/b_blip_full.yaml`, runner `scripts/kaggle_model_b.sh`.
No BUTD and no feature extraction: images are read straight from the COCO JPEGs.

## Notebook cells

```bash
!git clone -b feat/captioner-b-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/src/QuestionDependentImageCaptioner
!bash scripts/kaggle_model_b.sh
```

Attach the QD caption JSON and the COCO images (`mnhlfuch/vqa-v2`), select GPU T4 x2, turn **internet
on**, and run as **Save Version → Save & Run All**.

## Time

| Version | What | Expected |
|---|---|---|
| 1 | 4 epochs, 264k rows, batch 48 × 2 GPUs + val each epoch, then test (beam 3) | ~4–6 h (not measured; first epoch log shows minutes/epoch) |
| 2 (only if 1 stopped) | attach version 1's output; it resumes from `last.pt` | the rest |

`max_hours: 10.5` stops training between epochs before Kaggle's 12 h limit, so the output is saved.

## What to bring back

`model_b/b_blip_full/`: `test_metrics.json`, `train.log`, `best.pt` (decoder weights only, ~0.9 GB fp32;
the frozen ViT comes from the base model), `run_config.json`, and `model_b/results.md`.

## Local smoke test (no BLIP download)

```bash
python scripts/make_tiny_blip.py --qd ../dataset/vqa_v2_qd_captions_full_train2014.json   # tiny random BLIP
python train_blip.py --config configs/b_smoke_local.yaml
python train_blip.py --config configs/b_smoke_local.yaml --test --beam 3 --name b_smoke
```
