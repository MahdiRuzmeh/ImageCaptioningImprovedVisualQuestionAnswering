#!/usr/bin/env bash
# All BLIP-VQA runs: baseline and + BLIP captions with 3 seeds each, + oracle (1 seed), then the summary.
# Run from CaptionVQA/ (Git Bash):  bash scripts/run_blip_vqa.sh
# Needs: outputs/blip/features/regions_*.pt (prepare_regions.py) and outputs/blip/captions/captions_blip.json (Kaggle).
set -euo pipefail
PY=${PY:-../.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
CFG=configs/blip.yaml
for seed in 0 1 2; do
  $PY train_vqa.py --config $CFG --captions none --seed $seed | grep -E "^\[|^ep"
  $PY train_vqa.py --config $CFG --captions blip --seed $seed | grep -E "^\[|^ep"
done
$PY train_vqa.py --config $CFG --captions oracle --seed 0 | grep -E "^\[|^ep"
$PY summarize_vqa.py --config $CFG
