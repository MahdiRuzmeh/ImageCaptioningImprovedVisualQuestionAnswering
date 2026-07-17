# Image Captioning Improved Visual Question Answering

Implementation of the two-stage pipeline from *Image captioning improved visual question answering* (Sharma & Jalal, 2021).

| Project | Role |
|---------|------|
| [`SimpleImageCaptioner/`](SimpleImageCaptioner/) | **Stage 1** — question-dependent (QD) captioner: `(image, question) → caption` |
| [`QuestionDependentCaptions/`](QuestionDependentCaptions/) | Rule engine to build QD training JSON from VQA v2 Q+A |
| [`SimpleVQA/`](SimpleVQA/) | **Stage 2** — VQA v2 with frozen QD captioner + trainable VQA head |

Each project has `train.py`, `eval.py`, and YAML configs. See the README in each folder for details.

## Layout

```
src/
  QuestionDependentCaptions/   # optional: generate QD caption JSON from VQA
  SimpleImageCaptioner/        # stage 1 (QD captioner)
  SimpleVQA/                   # stage 2 (VQA)
  dataset/                     # MSCOCO 2014 + VQA v2 JSON (you provide)
  architecture/                # ARCHITECTURE.en.md / ARCHITECTURE.fa.md
  .venv/                       # shared virtual environment (recommended)
```

## Data

Place under `src/dataset/`:

| Path | Used by |
|------|---------|
| `v2_question_dependent_captions_{train,val}2014.json` | Captioner (generate with `QuestionDependentCaptions/`) |
| `train2014/`, `val2014/` | Both stages |
| `v2_OpenEnded_mscoco_*_questions.json` | VQA |
| `v2_mscoco_*_annotations.json` | VQA |

Configs use paths relative to each project folder (e.g. `../dataset/...`).

## Setup (Python + venv)

From `src/`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r SimpleImageCaptioner/requirements.txt -r SimpleVQA/requirements.txt
```

Requirements are the same for both projects: `torch`, `torchvision`, `pillow`, `tqdm`, `pyyaml`. Install [PyTorch](https://pytorch.org/) with CUDA if you use a GPU (`device: cuda` in configs).

## Run — full training

**Stage 0 (optional) — generate QD captions**

```powershell
cd QuestionDependentCaptions
python generate.py --help
```

**Stage 1 — QD captioner**

```powershell
cd SimpleImageCaptioner
python train.py --config configs/default.yaml
```

Checkpoint: `SimpleImageCaptioner/outputs/default/best.pt` (includes `vocab`, `q_vocab`, model).

**Stage 2 — VQA**

Point `captioner_ckpt` in `SimpleVQA/configs/default.yaml` at the Stage-1 checkpoint.

```powershell
cd ../SimpleVQA
python train.py --config configs/default.yaml
```

Checkpoint: `SimpleVQA/outputs/default/best.pt`

**Eval (greedy decode)**

```powershell
# VQA — full split metric + samples
python eval.py --config configs/default.yaml --ckpt outputs/default/best.pt --split val --samples 10

# Or metric only
python train.py --config configs/default.yaml --eval --ckpt outputs/default/best.pt
```

## Run — smoke test (quick)

Verifies the pipeline on 100 train + 100 val questions before a long run.

```powershell
cd SimpleImageCaptioner
python train.py --config configs/smoke.yaml

cd ../SimpleVQA
python train.py --config configs/smoke.yaml
```

Smoke VQA expects `SimpleImageCaptioner/outputs/smoke/best.pt` (set in `SimpleVQA/configs/smoke.yaml`).

## Kaggle 2×GPU (DDP)

Set `ddp: true` in the Kaggle YAML config, then launch with **`torchrun`** (not plain `python`):

```powershell
# Captioner (from SimpleImageCaptioner/)
torchrun --nproc_per_node=2 train.py --config configs/kaggle_mini.yaml

# VQA (from SimpleVQA/)
torchrun --nproc_per_node=2 train.py --config configs/kaggle_mini.yaml
```

Local / single-GPU testing: set `ddp: false` and run `python train.py --config ...` as usual.

Log should show `ddp=True world=2` when both GPUs are active.

## Qualitative check (pred vs ground truth)

Use `eval.py` in each project to print predictions next to ground truth.

### SimpleImageCaptioner — caption quality

```powershell
cd SimpleImageCaptioner

# QD val: token_acc + BLEU/CIDEr + samples
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split val --samples 10

# Single image + question → QD caption
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt `
  --image-id 25 --split train --question "How many animals are in this photo?"
```

| Flag | Meaning |
|------|---------|
| `--ckpt` | Captioner `best.pt` (required) |
| `--split` | `train` or `val` |
| `--samples N` | Print N captions: pred vs GT |
| `--image-id` | One COCO `image_id` |
| `--question` | Question text (QD mode) |

### SimpleVQA — answer quality

```powershell
cd SimpleVQA

# Val accuracy (greedy) + 10 random samples
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split val --samples 10

# Image + question → answer + pred caption
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt `
  --image-id 262148 --question "Where is he looking?" --split val

# question_id from VQA JSON → image, question, GT from annotations
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --question-id 262148000
```

| Flag | Meaning |
|------|---------|
| `--ckpt` | VQA `best.pt` (required) |
| `--split` | `train` or `val` |
| `--samples N` | Print N Q/A pairs with pred caption |
| `--image-id` + `--question` | Custom image + question |
| `--question-id` | VQA v2 `question_id` (loads from JSON) |

Numeric metric only (no samples):

```powershell
python train.py --config configs/smoke.yaml --eval --ckpt outputs/smoke/best.pt
```

## Resume training

```powershell
# VQA — continue from last epoch in save_dir
python train.py --config configs/default.yaml --continue

# Or explicit checkpoint
python train.py --config configs/default.yaml --resume outputs/default/last.pt
```

## Reproducibility

- Random seed: `42` in YAML configs (`seed: 42`)
- VQA metric: VQA v2 soft accuracy (greedy decode in `eval.py` and validation during training)
- Vocabs: built from **train question IDs only** (respects `max_train_qids` on smoke runs)

## Further reading

- [architecture/ARCHITECTURE.en.md](architecture/ARCHITECTURE.en.md) — full pipeline, vocabularies, diagrams
- [SimpleImageCaptioner/README.md](SimpleImageCaptioner/README.md) — QD captioner, `q_gru`, eval
- [SimpleVQA/README.md](SimpleVQA/README.md) — VQA model, `q_cap` vs `q`, `eval.py`, troubleshooting
