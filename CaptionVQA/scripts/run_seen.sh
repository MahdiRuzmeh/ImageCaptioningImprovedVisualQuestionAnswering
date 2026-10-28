#!/bin/bash
# "Seen" benchmark: A/B/C x caption/no-caption x seeds 0,1,2, trained + tested on
# captioner-train images (needs: make_seen_split.py + prepare_features.py --config configs/seen.yaml)
# then cross-condition evals (unseen-trained runs on seen test, seen-trained runs on unseen test).
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
for seed in 0 1 2; do
  for m in A B C; do
    for flag in "" "--no-caption"; do
      echo "=== seen $m $flag seed $seed ==="
      $PY train.py --config configs/seen.yaml --model $m $flag --seed $seed 2>&1 | grep -v Warning | tail -1
    done
  done
done
$PY eval_runs.py --runs-config configs/small.yaml --data-config configs/seen.yaml --split test 2>&1 | grep -v Warning
$PY eval_runs.py --runs-config configs/seen.yaml --data-config configs/small.yaml --split test 2>&1 | grep -v Warning
echo ALL_DONE
