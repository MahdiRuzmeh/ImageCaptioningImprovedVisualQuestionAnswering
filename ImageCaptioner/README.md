# ImageCaptioner

## Setup (first step)
1. Create environment:
   - `conda env create -f env.yml`
2. Activate environment:
   - `conda activate image_captioner`
3. Install requirements:
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
- Paper includes Flickr30k results, but local dataset availability is MSCOCO val only; training is restricted to this data.
- Question-guided captioning is supported by API; standalone captioner training uses image-caption pairs.

## Run
- Train: `python training/train.py --config configs/default.yaml`
- Evaluate: `python evaluation/evaluate.py --config configs/default.yaml --ckpt outputs/best.pt`
