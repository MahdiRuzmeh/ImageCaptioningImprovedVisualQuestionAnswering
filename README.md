# Image Captioning Improved Visual Question Answering

This workspace contains two modular projects:
- `ImageCaptioner/`
- `VQA/`

Both are implemented from the paper architecture with deterministic split on available local data only.

## Deterministic data protocol
- Source: MSCOCO val2014 + VQA v2 val2014
- Split: 80% train / 20% test
- Seed: `42`

## Setup (Python + venv)
1. Create one shared virtual environment at repo root:
   - `python -m venv .venv`
2. Activate it (Windows PowerShell):
   - `.venv\Scripts\Activate.ps1`
3. Upgrade pip and install dependencies once:
   - `python -m pip install --upgrade pip`
   - `pip install -r ImageCaptioner/requirements.txt -r VQA/requirements.txt`

## Run
1. Train captioner
   - `cd ImageCaptioner`
   - `python training/train.py --config configs/default.yaml`
2. Train VQA
   - `cd ../VQA`
   - `python training/train.py --config configs/default.yaml`
