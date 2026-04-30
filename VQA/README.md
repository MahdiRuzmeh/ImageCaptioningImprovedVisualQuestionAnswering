# VQA

## Setup (first step)
1. Create environment:
   - `conda env create -f env.yml`
2. Activate environment:
   - `conda activate vqa_caption_fusion`
3. Install requirements:
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
- Train: `python training/train.py --config configs/default.yaml`
- Evaluate: `python evaluation/evaluate.py --config configs/default.yaml --ckpt outputs/best.pt`
