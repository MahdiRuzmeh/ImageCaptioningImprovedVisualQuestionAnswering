#!/bin/bash
# A/B/C with captions from the COCO-pretrained, QD fine-tuned captioner (configs/small_ft_base.yaml), seeds 0,1,2.
# No-caption baselines are the runs of run_all.sh (same split, same features).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for seed in 0 1 2; do
  for m in C A B; do
    echo "=== $m ft_base seed $seed ==="
    ../.venv/Scripts/python.exe train.py --config configs/small_ft_base.yaml --model $m --captions ft_base --seed $seed 2>&1 | grep -v Warning
  done
done
echo ALL_DONE
