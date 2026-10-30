#!/bin/bash
# A/C with captions from the BUTD-feature captioner (configs/small_butd.yaml), seeds 0,1,2. B needs the old region layers.
# No-caption baselines are the runs of run_all.sh (same split, same features).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for seed in 0 1 2; do
  for m in C A; do
    echo "=== $m butd seed $seed ==="
    ../.venv/Scripts/python.exe train.py --config configs/small_butd.yaml --model $m --captions butd --seed $seed 2>&1 | grep -v Warning
  done
done
echo ALL_DONE
