# Image Captioning Improved Visual Question Answering

Implementation of the two-stage pipeline from *Image captioning improved visual question answering* (Sharma & Jalal, 2021).

| Project | Role |
|---------|------|
| [`SimpleImageCaptioner/`](SimpleImageCaptioner/) | **Stage 1** — train image captioner on MSCOCO captions |
| [`SimpleVQA/`](SimpleVQA/) | **Stage 2** — VQA v2 with frozen captioner + fine-tuned question embedding |

Each project is a single `train.py` plus YAML configs. See the README in each folder for architecture details.

## Layout

```
src/
  SimpleImageCaptioner/   # stage 1
  SimpleVQA/              # stage 2
  dataset/                # MSCOCO 2014 + VQA v2 JSON (you provide)
  .venv/                  # shared virtual environment (recommended)
```

## Data

Place under `src/dataset/`:

| Path | Used by |
|------|---------|
| `captions_train2014.json`, `captions_val2014.json` | Captioner |
| `train2014/`, `val2014/` | Both |
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

**Stage 1 — captioner**

```powershell
cd SimpleImageCaptioner
python train.py --config configs/default.yaml
```

Checkpoint: `SimpleImageCaptioner/outputs/default/best.pt`

**Stage 2 — VQA**

Ensure `SimpleVQA/configs/default.yaml` points at the captioner checkpoint (default: `../SimpleImageCaptioner/outputs/default/best.pt`).

```powershell
cd ../SimpleVQA
python train.py --config configs/default.yaml
```

Checkpoint: `SimpleVQA/outputs/default/best.pt`

**Eval (greedy decode)**

```powershell
python train.py --config configs/default.yaml --eval --ckpt outputs/default/best.pt
```

## Run — smoke test (quick)

Use this to verify the pipeline on 100 images / 100 questions before a long run.

```powershell
cd SimpleImageCaptioner
python train.py --config configs/smoke.yaml

cd ../SimpleVQA
python train.py --config configs/smoke.yaml
```

Smoke VQA expects `SimpleImageCaptioner/outputs/smoke/best.pt` (set in `SimpleVQA/configs/smoke.yaml`).

## Resume training

```powershell
# VQA — continue from last epoch in save_dir
python train.py --config configs/default.yaml --continue

# Or explicit checkpoint
python train.py --config configs/default.yaml --resume outputs/default/last.pt
```

## Reproducibility

- Random seed: `42` in YAML configs (`seed: 42`)
- VQA metric: VQA v2 soft accuracy (see `SimpleVQA/README.md`)

## Further reading

- [SimpleImageCaptioner/README.md](SimpleImageCaptioner/README.md) — caption decoder, paper dimensions, config keys
- [SimpleVQA/README.md](SimpleVQA/README.md) — VQA model, captioner integration, `q_emb` fine-tuning, troubleshooting
