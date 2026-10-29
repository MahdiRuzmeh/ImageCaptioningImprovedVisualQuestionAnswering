#!/bin/bash
# A/B/C with captions from the enriched captioner (configs/small_enriched.yaml), seeds 0,1,2.
# No-caption baselines are the runs of run_all.sh (same split, same features).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for seed in 0 1 2; do
  for m in C A B; do
    echo "=== $m enriched seed $seed ==="
    ../.venv/Scripts/python.exe train.py --config configs/small_enriched.yaml --model $m --captions enriched --seed $seed 2>&1 | grep -v Warning
  done
done
echo ALL_DONE
