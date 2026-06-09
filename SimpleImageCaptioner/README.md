# SimpleImageCaptioner

A minimal, single-file image captioning trainer (`train.py`). Docstrings reference *Image captioning improved visual question answering* (Sharma & Jalal, 2021).

Caption decoder (§3.3): at each step, attention weights depend on the **previous LSTM hidden state** `m_{t-1}` (our `h_{t-1}`), not a single fixed image context.

## Dimensions from the paper

The paper does **not** use a 256-D attention vector. Relevant sizes:

| What | Size | Where in paper |
|------|------|----------------|
| Global ResNet-101 feature `v_G` | **2048** | §3.1 |
| Local region feature `v_i` (each ROI) | **L = 2048** (`v_i ∈ ℝ^L`, eq. 4) | §3.3 |
| Number of regions `K` | **32** (VQA attention setup) | §5 |
| LSTM hidden layer | **512** | Table 2, §3.4 (`v_cap`) |
| Encoding / working dimension | **512** | Table 2 |
| Word embedding, attended features (working dim) | **all converted to 512** | §5 (experimental setup) |
| Question GRU hidden | **1280** | §3.1 (VQA path, not used in this caption-only script) |

§5 quote: *"Dimensions of hidden layer of LSTM, visual features, vector representing word embedding and attended features, are all converted to 512."*

Attention (§3.3): `α_{ti} ∝ exp(f_att(v_i, m_{t-1}))`, then `z_t = Σ_i α_{ti} v_i`. Scores are computed after mapping `v_i` and `m_{t-1}` into a common space; the paper sets that working size to **512**, while the weighted sum of regions stays in **2048-D** before projection to 512 for the LSTM.

## What this code implements

1. **Regions** — Frozen Faster R-CNN → up to **32** ROIs; ROI vectors are **1024-D** from torchvision, then `roi_to_region` maps to **2048** to match paper `L`.
2. **Attention** — `h_{t-1}` (512) and each `v_i` (2048) → **512-D** for similarity; softmax over 32 regions; sum → 2048-D `z_t`; `ctx_proj` → **512-D** context (paper §5).
3. **Decoder** — `LSTMCell` input = `[word embedding (512) ; context (512)]`, hidden **512**, predict next token.

Defaults live in `configs/default.yaml` (paper-aligned: 2048 regions, 512 LSTM/embed/word, batch 10, 25 epochs).

## Requirements

- Python 3.8+
- `torch`, `torchvision`, `pillow`, `tqdm`, `pyyaml`

## Data

MSCOCO 2014 under `src/dataset/`. Set paths in your YAML config (see `configs/default.yaml`).

## Train

All settings (learning rate, batch size, epochs, paths, model dims) are read from the YAML file named on the CLI:

```bash
cd SimpleImageCaptioner
python train.py --config configs/default.yaml
```

Smoke run:

```bash
python train.py --config configs/smoke.yaml
```

Kaggle:

```bash
python train.py --config configs/kaggle.yaml
```

## Config keys (`configs/default.yaml`)

| Key | Default | Paper |
|------|---------|-------|
| `region_dim` | 2048 | §3.1 / eq. (4) |
| `embed_dim` | 512 | §5, Table 2 |
| `word_dim` | 512 | §5 |
| `lstm_hidden` | 512 | Table 2 |
| `max_regions` | 32 | §5 |
| `learning_rate` | 0.0005 | Table 2 |
| `batch_size` | 10 | Table 2 |
| `epochs` | 25 | Table 2 |

## CLI

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` | `configs/default.yaml` | YAML with all training settings |

## Outputs

`save_dir` from config (e.g. `outputs/default/`): `last.pt`, `best.pt` (model, vocab, `config`).

## vs `ImageCaptioner/`

| | SimpleImageCaptioner | ImageCaptioner |
|--|----------------------|----------------|
| Files | One (`train.py`) + `configs/` | Modules + YAML |
| Attention | Per step from `h_{t-1}` | Often one vector from question |
| Config | YAML via `--config` | YAML + AMP / resume |

Use `ImageCaptioner` for full replication and VQA; use this folder to study the §3.3 caption flow with paper dimensions spelled out.
