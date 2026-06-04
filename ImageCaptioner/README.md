# ImageCaptioner

## Setup (first step)
1. Create virtual environment (from repo root):
   - `python -m venv .venv`
2. Activate virtual environment (Windows PowerShell):
   - `.venv\Scripts\Activate.ps1`
3. Install requirements for both projects once:
   - `pip install -r ImageCaptioner/requirements.txt -r VQA/requirements.txt`

If you want to install only captioner dependencies:
   - `pip install -r requirements.txt`

## Implemented components
- ResNet101 global encoder (pretrained)
- Faster R-CNN region features for bottom-up local objects (pretrained)
- Question-guided attention over regions
- LSTM caption decoder
- Caption embedding via mean pooling of generated caption token embeddings

## Hardware-safe defaults
- Mixed precision enabled
- Batch size `4` + gradient accumulation `4`
- Max region features per image: `32`

## Assumptions
- Paths for MSCOCO images and captions (train/val) are set in ``configs/default.yaml`` (use absolute paths on Kaggle, e.g. under ``/kaggle/input/``).
- VQA stage uses official train/val question and annotation JSON files from the same config.

## Run
- Train (from scratch): `python training/train.py --config configs/default.yaml --fresh`
- Smoke test (10 images, 1 epoch): `python training/train.py --config configs/smoke.yaml --fresh`
- Limit images in any config: set `max_train_images` / `max_val_images` (unique image ids, sorted; all captions per image are kept)
- Continue training (auto-resume from `save_dir/last.pt`): `python training/train.py --config configs/default.yaml --continue`
- Continue training (resume from a specific checkpoint): `python training/train.py --config configs/default.yaml --resume outputs/last.pt`
- Evaluate: `python evaluation/evaluate.py --config configs/default.yaml --ckpt outputs/best.pt`
- Evaluate with image previews: `python evaluation/evaluate.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --num-samples 3 --save-dir outputs/smoke/previews --show`
