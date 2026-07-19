# SimpleImageCaptioner

Question-dependent image captioning — Stage 1 of the thesis pipeline (*Image captioning improved visual question answering*, Sharma & Jalal, 2021).

Each training sample is `(image, question, caption)` from `v2_question_dependent_captions_*.json`. The decoder conditions on `qctx` (PAD-aware `q_gru`) for LSTM init `concat(mean_r, qctx)`, attention query `attn_query_proj([h; qctx])`, and LSTM input.

## Train

```bash
cd SimpleImageCaptioner
python train.py --config configs/default.yaml   # full run
python train.py --config configs/smoke.yaml   # quick test
```

## Eval

```bash
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split val --samples 10
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \
  --image-id 25 --split train --question "How many animals are in this photo?"
```

Checkpoint keys: `vocab` (caption words), `q_vocab` (question words for VQA `q_cap`), `model`.

## Config reference (`configs/default.yaml` / `smoke.yaml`)

All paths are relative to `SimpleImageCaptioner/` unless absolute.

### Data

| Key | Meaning |
|-----|---------|
| `train_captions_json` | Question-dependent caption JSON (train split) |
| `val_captions_json` | Question-dependent caption JSON (val split) |
| `train_images_dir` | COCO train image folder |
| `val_images_dir` | COCO val image folder |
| `train_image_filename_template` | Train image filename pattern (`{image_id}`) |
| `val_image_filename_template` | Val image filename pattern |
| `max_train_samples` | Max training rows `(image, question, caption)`; omit for all |
| `max_val_samples` | Max validation rows; omit for all |
| `max_train_images` | Optional cap on unique train image IDs (vocab + filter) |
| `max_val_images` | Optional cap on unique val image IDs |

### Training loop

| Key | Meaning |
|-----|---------|
| `seed` | Random seed (shifted per DDP rank) |
| `save_dir` | Checkpoint output directory |
| `save_model_type` | `epoch` = save each epoch; `item` = every `save_every_samples` |
| `save_every_samples` | Global samples between `last.pt` writes when `save_model_type: item` |
| `batch_size` | Batch size |
| `grad_accum_steps` | Gradient accumulation steps |
| `epochs` | Number of training epochs |
| `eval_every` | Run validation every N epochs |
| `learning_rate` | Adamax learning rate |
| `lr_decay_factor` | StepLR multiplicative decay |
| `lr_decay_interval` | StepLR period in epochs |
| `dropout` | Dropout after projections and LSTM |
| `use_amp` | Mixed precision on CUDA |
| `vocab_min_freq` | Min token frequency for caption and question vocabs |

### Sequences

| Key | Meaning |
|-----|---------|
| `max_caption_len` | Max caption length incl. BOS/EOS |
| `max_question_len` | Max question length incl. BOS/EOS |

### Model dimensions

| Key | Meaning |
|-----|---------|
| `word_dim` | Caption word embedding size |
| `hidden_dim` | LSTM hidden size (`lstm_hidden` alias) |
| `embed_dim` | Attention working dimension |
| `region_dim` | Region feature dim after `roi_to_region` |
| `question_dim` | `q_gru` hidden size |
| `max_regions` | Faster R-CNN regions kept per image |
| `use_gnn` | Enable RelationGNN on regions |
| `gnn_dim` | GNN hidden dimension |

### Image / cache / loader

| Key | Meaning |
|-----|---------|
| `image_size` | Input image size (square resize) |
| `cache_regions` | Cache raw Faster R-CNN region features to disk |
| `train_region_cache_dir` | Train region feature cache path |
| `val_region_cache_dir` | Val region feature cache path |
| `num_workers` | DataLoader worker processes |
| `pin_memory` | Pin CPU memory for faster GPU transfer |
| `persistent_workers` | Keep DataLoader workers alive between epochs |
| `prefetch_factor` | Batches prefetched per worker |

### Scheduled sampling (paper §5)

| Key | Meaning |
|-----|---------|
| `scheduled_sampling_max_prob` | Max prob. of model prediction vs GT token during train |
| `scheduled_sampling_interval` | Epochs between scheduled-sampling increases |
| `scheduled_sampling_increment` | Amount added to sampling prob each interval |

### Device / DDP

| Key | Meaning |
|-----|---------|
| `device` | `cuda` or `cpu` |
| `ddp` | Enable DistributedDataParallel (`torchrun`) |
| `ddp_backend` | DDP backend (e.g. `nccl`) |

## CLI

| Script | Flag | Meaning |
|--------|------|---------|
| `train.py` | `--config` | YAML path (default `configs/default.yaml`) |
| `eval.py` | `--config` | Same YAML as training |
| `eval.py` | `--ckpt` | `best.pt` or `last.pt` |
| `eval.py` | `--split` | `train` or `val` |
| `eval.py` | `--samples N` | Print N random pred vs GT |
| `eval.py` | `--image-id` | Single-image demo |
| `eval.py` | `--question` | Question text (required for QD single-image) |

## VQA integration (Stage 2)

```yaml
captioner_ckpt: ../SimpleImageCaptioner/outputs/smoke/best.pt
```

VQA uses `q_vocab` from this checkpoint as `q_cap` (separate from VQA’s own question vocab).

## Outputs

`save_dir`: `last.pt` (latest epoch), `best.pt` (best val token accuracy).
