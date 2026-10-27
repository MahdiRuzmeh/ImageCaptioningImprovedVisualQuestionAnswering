#!/bin/bash
# Upper bound: A/B/C with COCO human captions x seeds 0,1,2
# (needs: python prepare_human_captions.py --config configs/small.yaml)
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for m in A B C; do
    echo "=== $m human seed $seed ==="
    ../.venv/Scripts/python.exe train.py --config configs/small.yaml --model $m --captions human --seed $seed 2>&1 | grep -v Warning
  done
done
echo ALL_DONE
