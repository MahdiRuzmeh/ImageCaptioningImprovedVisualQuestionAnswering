# SimpleVQA

VQA trainer (`train.py`) and evaluator (`eval.py`) — Stage 2 of the thesis pipeline (*Image captioning improved visual question answering*, Sharma & Jalal, 2021).

Loads the frozen QD captioner from Stage 1 and trains `VQAModel` on VQA v2 `(image, question) → answer`.

## Model flow

```
image  → ResNet-101 + Faster R-CNN regions → RelationGNN → v_att
question (VQA q_vocab) → PAD-aware GRU → q_vec
image + question (captioner q_vocab as q_cap) → frozen captioner → v_cap
v = fuse(v_cap, v_att)   # mul | add | concat
dual LSTM → answer (greedy at eval)
```

## Prerequisites

```bash
cd ../SimpleImageCaptioner
python train.py --config configs/smoke.yaml    # or default.yaml
```

## Train / eval

```bash
cd SimpleVQA
python train.py --config configs/default.yaml
python train.py --config configs/smoke.yaml

python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split val --samples 20
python train.py --config configs/default.yaml --eval --ckpt outputs/default/best.pt
```

Resume: `--continue` or `--resume outputs/default/last.pt`. Fresh start: `--fresh`.

## Config reference (`configs/default.yaml` / `smoke.yaml`)

Paths relative to `SimpleVQA/`.

### VQA data

| Key | Meaning |
|-----|---------|
| `train_questions_json` | VQA v2 train questions JSON |
| `train_annotations_json` | VQA v2 train annotations JSON |
| `val_questions_json` | VQA v2 val questions JSON |
| `val_annotations_json` | VQA v2 val annotations JSON |
| `train_images_dir` | COCO train image folder |
| `val_images_dir` | COCO val image folder |
| `train_image_filename_template` | Train image filename pattern |
| `val_image_filename_template` | Val image filename pattern |
| `max_train_qids` | Cap training question IDs; vocabs built from these only |
| `max_val_qids` | Cap val question IDs for evaluation |

### Captioner (Stage 1)

| Key | Meaning |
|-----|---------|
| `use_captioner` | Fuse captioner `v_cap` with `v_att`; `false` = `v_att` only |
| `captioner_project_root` | Path to `SimpleImageCaptioner` code |
| `captioner_ckpt` | Stage-1 captioner checkpoint (`best.pt`) |
| `captioner_class` | Captioner class name in `captioner_v1.py` |
| `captioner_finetune_q` | Unfreeze captioner `q_emb` / `q_gru` during VQA train |
| `caption_repr` | How `v_cap` is built: `hidden` or `text` |

### Training loop

| Key | Meaning |
|-----|---------|
| `seed` | Random seed |
| `save_dir` | Checkpoint output directory |
| `save_model_type` | `epoch` or `item` checkpoint schedule |
| `save_every_samples` | Samples between saves when `save_model_type: item` |
| `batch_size` | Batch size |
| `grad_accum_steps` | Gradient accumulation steps |
| `epochs` | Number of training epochs |
| `eval_every` | Greedy validation every N epochs |
| `learning_rate` | Adamax learning rate |
| `lr_decay_every` | StepLR period in epochs |
| `lr_decay_factor` | StepLR multiplicative decay |
| `weight_decay` | L2 regularization on trainable weights |
| `label_smoothing` | Cross-entropy label smoothing |
| `use_amp` | Mixed precision on CUDA |
| `vocab_min_freq` | Min frequency for question tokens in `q_vocab` |
| `dropout` | Dropout on projections and LSTM outputs |

### Sequences

| Key | Meaning |
|-----|---------|
| `max_question_len` | Max question length incl. BOS/EOS |
| `max_answer_len` | Max answer length incl. BOS/EOS |

### Model dimensions

| Key | Meaning |
|-----|---------|
| `hidden_dim` | LSTM and fusion hidden size |
| `word_dim` | Question and answer embedding size |
| `question_dim` | Question GRU hidden size |
| `max_regions` | Faster R-CNN regions per image |
| `fuse_mode` | Fuse `v_cap` and `v_att`: `mul`, `add`, or `concat` |

### Image / cache / loader

| Key | Meaning |
|-----|---------|
| `image_size` | Input image size (square resize) |
| `cache_regions` | Cache Faster R-CNN region features |
| `train_region_cache_dir` | Train region cache path |
| `val_region_cache_dir` | Val region cache path |
| `cache_global` | Cache ResNet-101 global features (pre-`g_proj`) |
| `train_global_cache_dir` | Train global feature cache path |
| `val_global_cache_dir` | Val global feature cache path |
| `num_workers` | DataLoader worker processes |
| `pin_memory` | Pin CPU memory for faster GPU transfer |
| `persistent_workers` | Keep DataLoader workers alive between epochs |
| `prefetch_factor` | Batches prefetched per worker |

### Device / DDP

| Key | Meaning |
|-----|---------|
| `device` | `cuda` or `cpu` |
| `ddp` | Enable DistributedDataParallel (`torchrun`) |
| `ddp_backend` | DDP backend (e.g. `nccl`) |
| `ddp_find_unused_parameters` | DDP flag for unused parameters in backward |
| `resume_from` | Optional explicit checkpoint path to resume |

## CLI

### `train.py`

| Flag | Meaning |
|------|---------|
| `--config` | YAML path |
| `--continue` | Resume `save_dir/last.pt` |
| `--resume PATH` | Resume from file |
| `--fresh` | Ignore resume |
| `--eval` | Greedy val accuracy only |
| `--ckpt PATH` | Checkpoint for `--eval` |

### `eval.py`

| Flag | Meaning |
|------|---------|
| `--ckpt` | VQA checkpoint (required) |
| `--split` | `train` or `val` |
| `--samples N` | Random predictions with captions |
| `--image-id` + `--question` | Single inference |
| `--question-id` | Load from VQA JSON |

## Vocabs (important)

| Token stream | Source | Used for |
|--------------|--------|----------|
| `q` | `q_vocab` in VQA ckpt (train QIDs) | `VQAModel.q_emb` |
| `q_cap` | `q_vocab` in captioner ckpt | Captioner → `v_cap` |
| `a` | `a_vocab` in VQA checkpoint (train QIDs) | Answer decoder |

## Metric

VQA v2 **soft accuracy** (greedy decode at validation). `train_acc` in logs = teacher-forcing token accuracy.

## Outputs

`last.pt`, `best.pt` (best greedy `val_acc`). Contains `model`, `q_vocab`, `a_vocab`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `val_acc` stuck at 0 | Use current code (vocabs from capped train QIDs; PAD-aware GRU; EOS-aware loss) |
| Caption ignores question | `eval.py` must pass `q_cap`, not VQA `q` |
| `val_loss=0` in log | Greedy eval — use `val_acc` |
