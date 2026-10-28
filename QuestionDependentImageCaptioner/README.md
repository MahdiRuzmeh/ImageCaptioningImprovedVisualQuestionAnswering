# QuestionDependentImageCaptioner

Question-dependent image captioning — Stage 1 of the thesis pipeline (*Image captioning improved visual question answering*, Sharma & Jalal, 2021).

Each training sample is `(image, question, caption)` from `vqa_v2_question_dependent_captions_*.json`.

## Models

The model is chosen by `model_class` in the YAML, through the registry `models.build_captioner`. `model_class` is saved in the checkpoint `config`, so `eval.py`, `benchmark.py` and SimpleVQA rebuild the right class.

| `model_class` | File | Status |
|---------------|------|--------|
| `MMTransformerCopyCaptioner` | `models/captioner_mmtransformer_copy.py` | **Default (v2)**, used in all `configs/*.yaml` |
| `QuestionDependentImageCaptioner` | `models/captioner_v1.py` | v1, kept for old checkpoints (no `model_class` → v1) |

**v2 pipeline:**
1. Faster R-CNN regions (frozen, 32 × 1024 → `roi_to_region` 2048 → d) and question tokens (+ position, + type embedding).
2. A **joint Transformer encoder**, so regions and question words attend to each other.
3. A **Transformer decoder** (causal + cross-attention).
4. A **pointer-generator** output: `p_gen · P_vocab + (1 − p_gen) · copy-from-question`.

There's no RelationGNN in Stage 1.

Why v2: see [BENCHMARK.md](BENCHMARK.md). On a test split whose images don't appear in training (3 seeds), it reaches CIDEr-D 4.07 vs 1.53 for v1. The copy mechanism is what gives the gain; adding question words to v1's vocab alone does not. Full architecture: [`../architecture/ARCHITECTURE.en.md`](../architecture/ARCHITECTURE.en.md).

## Train

```bash
cd QuestionDependentImageCaptioner
python scripts/cache_regions.py --config configs/smoke.yaml   # once: Faster R-CNN cache for all splits
python train.py --config configs/smoke.yaml     # 3000-row benchmark protocol (image-disjoint splits)
python train.py --config configs/default.yaml   # full run
python train.py --config configs/smoke.yaml --seed 43 --save-dir outputs/smoke_s43 --fresh   # CLI overrides
```

`configs/smoke.yaml` uses `images_from_cache: true`: JPEGs are not loaded, and every image must already be in the region cache (run `scripts/cache_regions.py` first).

## Eval / benchmark

```bash
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split val --samples 10
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \
  --image-id 25 --split train --question "How many animals are in this photo?"
# Test split → outputs/smoke/test_metrics.json (BLEU, CIDEr-D, ROUGE-L, Answer-Hit, per question type)
python benchmark.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --name smoke
# Image ablation: does the model use the image? (normal / shuffle / mean / mask / question-only prior)
python scripts/image_ablation.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split test \
  --out outputs/ablation/smoke_test.json
# Multi-seed comparison
bash scripts/run_bench.sh configs/bench_transformer_copy.yaml transformer_copy
python scripts/bench_compare.py --root outputs/bench --out outputs/bench/summary.md
```

Checkpoint keys: `vocab` (caption words), `q_vocab` (question words for VQA `q_cap`), `model`, `config` (incl. `model_class`), `best_epoch`, `select_metric`.

## Config reference (`configs/default.yaml` / `smoke.yaml`)

All paths are relative to `QuestionDependentImageCaptioner/` unless absolute.

### Data

| Key | Meaning |
|-----|---------|
| `train_captions_json` | QD caption JSON (train) — `vqa_v2_question_dependent_captions_train2014.json` |
| `val_captions_json` | QD caption JSON (val) |
| `train_images_dir` | COCO train image folder |
| `val_images_dir` | COCO val image folder |
| `train_image_filename_template` | Train image filename pattern (`{image_id}`) |
| `val_image_filename_template` | Val image filename pattern |
| `train_row_range` / `val_row_range` / `test_row_range` | `[start, end)` rows of `train_captions_json`, snapped to `image_id` boundaries → image-disjoint splits (overrides `max_*`) |
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
| `model_class` | Captioner class (registry): `MMTransformerCopyCaptioner` or `QuestionDependentImageCaptioner` |
| `optimizer` | `adamw` (v2) or `adamax` (v1) |
| `learning_rate` | Base learning rate |
| `weight_decay` | AdamW weight decay |
| `warmup_steps` | Linear LR warmup (optimizer steps) |
| `lr_decay_factor` | Step decay multiplier |
| `lr_decay_interval` | Step decay period in epochs |
| `grad_clip` | Max grad norm (0 = off) |
| `label_smoothing` | CE label smoothing |
| `select_metric` | `cider` (best val CIDEr-D, greedy) or `loss` for `best.pt` |
| `val_beam_size` | Beam size for val CIDEr decode (1 = greedy) |
| `save_last` | Write `last.pt` each epoch (needed for `--continue`) |
| `save_detector_weights` | `false` = don't save frozen Faster R-CNN weights (~300 MB smaller) |
| `dropout` | Dropout (Transformer layers / embeddings in v2; projections + LSTM in v1) |
| `use_amp` | Mixed precision on CUDA |
| `vocab_min_freq` | Min token frequency for caption and question vocabs |
| `cap_vocab_include_question` | Caption vocab = caption ∪ question words (required for copy, v2) |

### Sequences

| Key | Meaning |
|-----|---------|
| `max_caption_len` | Max caption length incl. BOS/EOS |
| `max_question_len` | Max question length incl. BOS/EOS |

### Model dimensions

| Key | Meaning |
|-----|---------|
| `hidden_dim` | v2: Transformer width d (= `word_emb` = `v_cap` dim). v1: LSTM / qctx / fused size |
| `num_heads` / `num_layers` / `ff_dim` | v2 Transformer heads / encoder+decoder layers / FFN size |
| `word_dim` | v1 caption word embedding size |
| `embed_dim` | v1 attention working dimension |
| `region_dim` | Region feature dim after `roi_to_region` |
| `question_dim` | reserved (YAML parity with VQA) |
| `max_regions` | Faster R-CNN regions kept per image |
| `fuse_mode` | v1 feature fusion mode (`concat` only) |

### Image / cache / loader

| Key | Meaning |
|-----|---------|
| `image_size` | Input image size (square resize) |
| `cache_regions` | Cache raw Faster R-CNN region features to disk |
| `images_from_cache` | Skip JPEG loading; features come only from the region cache |
| `test_region_cache_dir` | Test split cache (falls back to `val_region_cache_dir`) |
| `train_region_cache_dir` | Train region feature cache path |
| `val_region_cache_dir` | Val region feature cache path |
| `num_workers` | DataLoader worker processes |
| `pin_memory` | Pin CPU memory for faster GPU transfer |
| `persistent_workers` | Keep DataLoader workers alive between epochs |
| `prefetch_factor` | Batches prefetched per worker |

### Scheduled sampling (paper §5, v1 only — ignored by MMTransformerCopyCaptioner)

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
| `train.py` | `--seed` / `--save-dir` | Override `seed` / `save_dir` (multi-seed runs) |
| `train.py` | `--fresh` / `--continue` / `--resume` | Ignore / use `last.pt` / use a given checkpoint |
| `eval.py` | `--config` | Same YAML as training |
| `eval.py` | `--ckpt` | `best.pt` or `last.pt` |
| `eval.py` | `--split` | `train`, `val` or `test` (row ranges when set) |
| `eval.py` | `--samples N` | Print N random pred vs GT |
| `eval.py` | `--image-id` | Single-image demo |
| `eval.py` | `--question` | Question text (auto from JSON if omitted) |
| `benchmark.py` | `--config` / `--ckpt` / `--name` | Splits from config, model from checkpoint, label for results |
| `benchmark.py` | `--split` / `--beam` / `--out` | `test` (default) or `val` / beam size (3) / output JSON |
| `scripts/cache_regions.py` | `--config` | Cache Faster R-CNN features for every split with a row range |
| `VISUAL_GROUNDING.md` | Same question, different image: is the kaggle_mini captioner visually grounded? (train/val/test) |
| `scripts/image_ablation.py` | `--config` / `--ckpt` / `--split` / `--modes` / `--out` | Decode with real, shuffled, mean or masked regions + question-only prior; ImageDep = AnswerHit(normal) − AnswerHit(shuffle) |

## VQA integration (Stage 2)

```yaml
captioner_project_root: ../QuestionDependentImageCaptioner
captioner_ckpt: ../QuestionDependentImageCaptioner/outputs/smoke/best.pt
captioner_class: MMTransformerCopyCaptioner   # fallback/check; the checkpoint's model_class wins
```

`SimpleVQA` `load_captioner()` builds the captioner through this project's registry (class and dims come from `ckpt['config']`). VQA uses `q_vocab` from this checkpoint as `q_cap` (separate from VQA's own question vocab). Both captioners implement `get_caption_embedding` (`caption_repr: hidden`) and `_decode_caption` + `word_emb` (`caption_repr: text`). For v2, `v_cap` dim = `hidden_dim` (512).

## Outputs

`save_dir`:
- `best.pt`: best val CIDEr-D (or val loss).
- `last.pt`: latest epoch, only if `save_last: true`.
- `history.json`: per-epoch loss, accuracy, val CIDEr/BLEU/Answer-Hit and timing.
- `run_config.yaml`: the resolved config.

`benchmark.py` adds `test_metrics.json` (all predictions included).
