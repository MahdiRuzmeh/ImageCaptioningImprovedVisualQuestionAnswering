# SimpleVQA

A minimal, single-file VQA trainer (`train.py`) for the two-stage pipeline in *Image captioning improved visual question answering* (Sharma & Jalal, 2021).

**Stage 1** — train the captioner in [`SimpleImageCaptioner/`](../SimpleImageCaptioner/).  
**Stage 2** — load the frozen captioner into `VQAModel`, fine-tune question layers in the captioner, and train the VQA head on VQA v2.

## Model flow (paper §3.4)

```
image  → ResNet-101 global (v_G) + Faster R-CNN regions → RelationGNN → v_att
question → GRU (q_emb) → q_vec → attend regions → v_att
image + question → frozen captioner → v_cap
v = v_cap ⊙ v_att   (or v_cap + v_att if fuse_mode: add)
dual LSTM → answer logits
```

| Component | Role |
|-----------|------|
| **ResNet-101** | Global image feature `g` for attention LSTM |
| **Faster R-CNN** | Up to 32 local regions (frozen) |
| **RelationGNN** | Message passing between regions |
| **Question GRU** | `q_emb` + GRU → `q_vec` for region attention |
| **Captioner** | Question-conditioned caption → `v_cap` (see below) |
| **Fusion** | `v = v_cap * v_att` (default) or `v_cap + v_att` |
| **Dual LSTM** | Attention LSTM + answer LSTM → answer tokens |

## Captioner integration (important)

The captioner is loaded from `SimpleImageCaptioner` via `load_captioner()` in `train.py`.

### Two separate vocabularies

| Layer | Vocabulary | Trained when |
|-------|------------|--------------|
| `word_emb`, `classifier` | MSCOCO **caption** vocab (from checkpoint) | Stage 1 (caption training) |
| `q_emb` | VQA **question** vocab | Stage 2 (VQA — fine-tuned) |

Caption and question token IDs must not share one embedding table; sizes differ (e.g. smoke: 249 vs 7650).

### What is frozen vs trainable in stage 2

| Captioner part | Stage 2 |
|----------------|---------|
| Region encoder, LSTM, attention, `word_emb`, `classifier` | **Frozen** (from caption checkpoint) |
| `q_emb`, `q_proj` | **Trainable** (random init; no question data in stage 1) |

There is no ground-truth caption for `(image, question)` pairs in VQA, so the caption decoder is not re-trained. Only `q_emb` / `q_proj` learn from the **answer loss** (indirect signal).

### `v_cap` during train vs eval

| Mode | How `v_cap` is computed |
|------|-------------------------|
| **Train** | Mean of LSTM hidden states during greedy decode (`differentiable=True`) so gradients reach `q_emb` (argmax blocks grad through token embeddings) |
| **Eval** | Mean-pool `word_emb` on generated caption tokens (paper §3.4) |

## Requirements

- Python 3.8+
- `torch`, `torchvision`, `pillow`, `tqdm`, `pyyaml`
- A trained captioner checkpoint from `SimpleImageCaptioner` (see [prerequisites](#prerequisites))

## Data

Under `src/dataset/`:

| File | Purpose |
|------|---------|
| `v2_OpenEnded_mscoco_*_questions.json` | VQA v2 questions |
| `v2_mscoco_*_annotations.json` | Answers (10 annotators per question) |
| `train2014/`, `val2014/` | COCO 2014 images |

Paths are set in `configs/default.yaml` (relative to `SimpleVQA/`).

## Prerequisites

Train the captioner first:

```bash
cd ../SimpleImageCaptioner
python train.py --config configs/smoke.yaml    # quick test
python train.py --config configs/default.yaml  # full run
```

Point `captioner_ckpt` in your VQA config at the saved checkpoint, e.g.:

```yaml
captioner_project_root: ../SimpleImageCaptioner
captioner_ckpt: ../SimpleImageCaptioner/outputs/default/best.pt
captioner_class: SimpleImageCaptioner
```

## Train

From `SimpleVQA/`:

```bash
python train.py --config configs/default.yaml
```

Smoke run (100 train + 100 val questions, captioner smoke checkpoint):

```bash
python train.py --config configs/smoke.yaml
```

Resume from last checkpoint in `save_dir`:

```bash
python train.py --config configs/default.yaml --continue
```

Resume from a specific file:

```bash
python train.py --config configs/default.yaml --resume outputs/default/last.pt
```

Start fresh (ignore resume):

```bash
python train.py --config configs/default.yaml --fresh
```

## Eval

Greedy answer decoding + VQA v2 soft accuracy:

```bash
python train.py --config configs/default.yaml --eval --ckpt outputs/default/best.pt
```

## Config keys

### Data & captioner

| Key | Default | Meaning |
|-----|---------|---------|
| `train_questions_json` | (see yaml) | VQA v2 train questions |
| `train_annotations_json` | (see yaml) | VQA v2 train answers |
| `val_questions_json` | (see yaml) | Val questions |
| `val_annotations_json` | (see yaml) | Val answers |
| `train_images_dir` | `../dataset/train2014` | COCO train images |
| `val_images_dir` | `../dataset/val2014` | COCO val images |
| `captioner_project_root` | `../SimpleImageCaptioner` | Captioner code path |
| `captioner_ckpt` | `../SimpleImageCaptioner/outputs/default/best.pt` | Stage-1 weights |
| `captioner_class` | `SimpleImageCaptioner` | Class name in `captioner_v1.py` |

### Training

| Key | Default | Paper / notes |
|-----|---------|----------------|
| `batch_size` | 4 | Effective batch = `batch_size * grad_accum_steps` |
| `grad_accum_steps` | 4 | Gradient accumulation |
| `epochs` | 25 | |
| `learning_rate` | 0.0005 | Adamax |
| `lr_decay_every` | 4 | StepLR period |
| `lr_decay_factor` | 0.6 | StepLR gamma |
| `max_question_len` | 14 | Max question tokens (incl. BOS/EOS) |
| `max_answer_len` | 6 | Max answer tokens |
| `vocab_min_freq` | 4 | Min token freq for q/a vocabs |
| `use_amp` | true | Mixed precision (CUDA) |
| `fuse_mode` | `mul` | `mul` or `add` for `v_cap` and `v_att` |
| `max_train_qids` | null | Cap train size (smoke: 100) |
| `max_val_qids` | null | Cap val size (smoke: 100) |

### Model dimensions

| Key | Default | Paper |
|-----|---------|-------|
| `hidden_dim` | 512 | Table 2 |
| `word_dim` | 512 | §5 |
| `question_dim` | 1280 | §3.1 (GRU hidden) |
| `max_regions` | 32 | §5 |

## CLI

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` | `configs/default.yaml` | YAML with all settings |
| `--continue` | off | Resume from `save_dir/last.pt` |
| `--resume PATH` | — | Resume from explicit checkpoint |
| `--fresh` | off | Ignore resume; train from scratch |
| `--eval` | off | Evaluation only (greedy decode) |
| `--ckpt PATH` | — | Checkpoint for `--eval` |

## Outputs

`save_dir` from config (e.g. `outputs/default/`):

| File | Contents |
|------|----------|
| `last.pt` | Latest epoch: model, optimizer, scheduler, vocabs, config |
| `best.pt` | Best validation accuracy checkpoint |

Checkpoint includes `q_vocab`, `a_vocab`, and full `VQAModel` state (including fine-tuned `captioner.q_emb`).

## Metric

**VQA v2 soft accuracy** — for each question, score = min(agreement_with_annotators / 3, 1), averaged over the split. Validation during training uses teacher forcing; `--eval` uses greedy decoding.

## vs `VQA/`

| | SimpleVQA | VQA (legacy) |
|--|-----------|--------------|
| Files | Single `train.py` | Multiple modules |
| Captioner load | Inline `load_captioner()` | `models/captioner_adapter.py` |
| Separate `q_emb` | Yes | Yes (via updated adapter) |
| Fine-tune `q_emb` | Yes | Yes (via updated adapter) |

## Typical workflow

```bash
# 1. Captioner (stage 1)
cd SimpleImageCaptioner
python train.py --config configs/default.yaml

# 2. VQA (stage 2)
cd ../SimpleVQA
python train.py --config configs/default.yaml

# 3. Eval
python train.py --config configs/default.yaml --eval --ckpt outputs/default/best.pt
```

## Troubleshooting

| Issue | Cause / fix |
|-------|-------------|
| `size mismatch for word_emb.weight` | Old code resized captioner to question vocab. Use current code: caption vocab from checkpoint, separate `q_emb`. |
| `question_ids provided but captioner has no q_emb` | Load captioner with `question_vocab_size` (current `load_captioner` does this). |
| `targets should not be None` (Faster R-CNN) | Detector must stay in `eval()` during VQA train — handled in `VQAModel.train()`. |
| Val loss differs from train loss | Train uses LSTM-hidden `v_cap`; eval uses word-embedding `v_cap` — expected with current design. |
