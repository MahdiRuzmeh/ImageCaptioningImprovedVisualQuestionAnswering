# Image Captioning Improved Visual Question Answering

This workspace contains two modular projects:
- `ImageCaptioner/`
- `VQA/`

Both are implemented from the paper architecture with deterministic split on available local data only.

## Deterministic data protocol
- Source: MSCOCO val2014 + VQA v2 val2014
- Split: 80% train / 20% test
- Seed: `42`

## Run
1. Create and activate environment for captioner
   - `cd ImageCaptioner`
   - `conda env create -f env.yml`
   - `conda activate image_captioner`
   - `pip install -r requirements.txt`
2. Train captioner
   - `cd ImageCaptioner`
   - `python training/train.py --config configs/default.yaml`
3. Create and activate environment for VQA
   - `cd ../VQA`
   - `conda env create -f env.yml`
   - `conda activate vqa_caption_fusion`
   - `pip install -r requirements.txt`
4. Train VQA
   - `cd ../VQA`
   - `python training/train.py --config configs/default.yaml`
