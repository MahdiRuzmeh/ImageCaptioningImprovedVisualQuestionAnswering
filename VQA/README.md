# VQA

## Setup (first step)
1. Create virtual environment (from repo root):
   - `python -m venv .venv`
2. Activate virtual environment (Windows PowerShell):
   - `.venv\Scripts\Activate.ps1`
3. Install requirements for both projects once:
   - `pip install -r ImageCaptioner/requirements.txt -r VQA/requirements.txt`

If you want to install only VQA dependencies:
   - `pip install -r requirements.txt`

## Implemented architecture
- ResNet101 global visual features
- Faster R-CNN local region features
- GRU question encoder
- Question-guided attention
- Graph neural network object relation modeling
- Caption feature fusion with attended visual features (element-wise multiplication default)
- Two-stage LSTM answer decoder (`LSTM_att`, `LSTM_ans`)

## Captioner plugin interface
VQA depends on `BaseImageCaptioner` only:
- `generate_caption(image, question_ids=None, max_len=20)`
- `encode_caption(caption_ids)`
- `get_caption_embedding(image, question_ids=None)`

Switch captioner through config only:
- `captioner_class: ImageCaptionerV1`
- `captioner_ckpt: ../ImageCaptioner/outputs/best.pt`

VQA core code remains unchanged when swapping captioners.

## Run
- Train (from scratch): `python training/train.py --config configs/default.yaml --fresh`
- Continue training (auto-resume from `save_dir/last.pt`): `python training/train.py --config configs/default.yaml --continue`
- Continue training (resume from a specific checkpoint): `python training/train.py --config configs/default.yaml --resume outputs/last.pt`
- Evaluate: `python evaluation/evaluate.py --config configs/default.yaml --ckpt outputs/best.pt`
