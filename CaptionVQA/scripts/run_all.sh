#!/bin/bash
# Full benchmark: A/B/C x caption/no-caption x seeds 0,1,2
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for m in A B C; do
    for flag in "" "--no-caption"; do
      echo "=== $m $flag seed $seed ==="
      ../.venv/Scripts/python.exe train.py --config configs/small.yaml --model $m $flag --seed $seed 2>&1 | grep -v Warning
    done
  done
done
echo ALL_DONE
